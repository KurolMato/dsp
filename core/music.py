"""音乐分离任务与混音；pymss 仅由子进程加载。"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from uuid import uuid4

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "music"
OUTPUT_DIR = ROOT / "results" / "music"
PRESETS = {
    "人声 / 伴奏 · VR": "1_HP-UVR",
    "人声 / 伴奏 · RoFormer": "bs_roformer_voc_hyperacev2",
    "四轨 · 人声 / 鼓 / 贝斯 / 其他": "HTDemucs4",
}
STEM_LABELS = {"vocals": "人声", "instrumental": "伴奏", "drums": "鼓",
               "bass": "贝斯", "other": "其他", "original": "原曲"}


def resample_audio(audio, input_fs, output_fs, length=None):
    from math import gcd
    from scipy.signal import resample_poly
    a = np.asarray(audio, dtype=np.float32)
    if a.ndim == 1:
        a = a[:, None]
    if input_fs != output_fs:
        g = gcd(int(input_fs), int(output_fs))
        a = resample_poly(a, output_fs // g, input_fs // g, axis=0).astype(np.float32)
    if length is not None:
        a = a[:length]
        if len(a) < length:
            a = np.pad(a, ((0, length - len(a)), (0, 0)))
    return a


def mix_tracks(tracks, gains, start=0, frames=None):
    if not tracks:
        raise ValueError("没有可混音的音轨")
    first = next(iter(tracks.values()))
    end = len(first) if frames is None else min(len(first), start + frames)
    out = np.zeros((max(0, end - start), first.shape[1]), np.float32)
    for name, track in tracks.items():
        if track.shape != first.shape:
            raise ValueError("音轨长度或声道数不一致")
        gain = float(gains.get(name, 0))
        if not np.isfinite(gain) or not 0 <= gain <= 2:
            raise ValueError("音量必须在 0~2 之间")
        if gain:
            out += track[start:end] * gain
    return out


class SeparationJob:
    """单任务单进程：取消终止 worker，未完成目录保留状态但不会作为成功结果。"""
    def __init__(self, emit, python=None):
        self.emit = emit
        self.python = str(python or sys.executable)
        self.process = None
        self.cancelled = threading.Event()
        self._lock = threading.Lock()
        self.id = uuid4().hex
        self.output = OUTPUT_DIR / (time.strftime("%Y%m%d_%H%M%S") + "_" + self.id[:8])
        self.finished = threading.Event()
        self.completed = threading.Event()

    def start(self, input_path, model, device="auto", source="modelscope", download_only=False):
        if not download_only and not Path(input_path).is_file():
            raise FileNotFoundError(input_path)
        self.output.mkdir(parents=True, exist_ok=False)
        request = {"task_id": self.id, "input": str(input_path), "model": model,
                   "device": device, "source": source, "model_dir": str(MODEL_DIR),
                   "output_dir": str(self.output), "download_only": download_only}
        request_path = self.output / "request.json"
        request_path.write_text(json.dumps(request, ensure_ascii=False, indent=2), encoding="utf-8")
        threading.Thread(target=self._run, args=(request_path,), daemon=True).start()

    def _send(self, event):
        self.emit({**event, "task_id": self.id, "job": self})

    def _run(self, request_path):
        done = False
        try:
            with self._lock:
                if self.cancelled.is_set():
                    return
                self.process = subprocess.Popen(
                    [self.python, "-X", "utf8", "-u", str(ROOT / "music_worker.py"), str(request_path)],
                    cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    encoding="utf-8", errors="replace", text=True,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            with (self.output / "worker.log").open("w", encoding="utf-8") as log:
                for line in self.process.stdout:
                    log.write(line)
                    if line.startswith("DSP_EVENT "):
                        event = json.loads(line[len("DSP_EVENT "):])
                        if event.get("type") == "done":
                            done = True
                            self.completed.set()
                        if not self.cancelled.is_set():
                            self._send(event)
                    elif line.strip():
                        self._send({"type": "log", "message": line.strip()[-500:]})
            code = self.process.wait()
            if not self.cancelled.is_set() and code != 0:
                self._send({"type": "error", "message": f"分离进程退出码 {code}，详见 {self.output / 'worker.log'}"})
            elif not self.cancelled.is_set() and not done:
                self._send({"type": "error", "message": "分离进程没有返回完整结果"})
        except Exception as exc:
            if not self.cancelled.is_set():
                self._send({"type": "error", "message": str(exc)})
        finally:
            if self.cancelled.is_set():
                # 原子地移除成功标记，避免取消与进程完成的竞争。
                (self.output / "result.json").unlink(missing_ok=True)
                (self.output / "status.json").write_text('{"state":"cancelled"}', encoding="utf-8")
                self._send({"type": "cancelled", "message": "任务已取消"})
            self.finished.set()

    def cancel(self):
        if self.completed.is_set() or self.finished.is_set():
            return
        self.cancelled.set()
        with self._lock:
            if self.process is not None and self.process.poll() is None:
                self.process.terminate()


class StemPlayer:
    """音频回调内按同一游标混音，切换音轨和音量不会重新开始播放。"""
    def __init__(self):
        self.stream = None
        self.tracks = {}
        self.gains = {}
        self.fs = 44100
        self.position = 0
        self.playing = False
        self.underruns = 0
        self._lock = threading.RLock()

    def load(self, tracks, fs):
        self.stop()
        if not tracks:
            raise ValueError("没有音轨")
        normalized = {name: resample_audio(a, fs, fs) for name, a in tracks.items()}
        shapes = {a.shape for a in normalized.values()}
        if len(shapes) != 1 or not next(iter(shapes))[0]:
            raise ValueError("音轨长度或声道数不一致")
        if int(fs) <= 0 or any(not np.isfinite(a).all() for a in normalized.values()):
            raise ValueError("音轨采样率或数值无效")
        with self._lock:
            self.tracks = normalized
            self.gains = {name: float(name != "original" and name != "instrumental") for name in tracks}
            if set(tracks) == {"original", "vocals", "instrumental"}:
                self.gains.update(vocals=1.0, instrumental=1.0)
            self.fs = int(fs)
            self.position = 0

    def set_gains(self, gains):
        if any(not np.isfinite(float(g)) or not 0 <= float(g) <= 2 for g in gains.values()):
            raise ValueError("音量必须在 0~2 之间")
        with self._lock:
            self.gains = dict(gains)

    def seek(self, seconds):
        with self._lock:
            n = len(next(iter(self.tracks.values()))) if self.tracks else 0
            self.position = max(0, min(n, round(float(seconds) * self.fs)))

    def render(self, frames):
        with self._lock:
            out = mix_tracks(self.tracks, self.gains, self.position, frames)
            self.position += len(out)
        return np.clip(out, -1, 1)

    def play(self):
        import sounddevice as sd
        if not self.tracks:
            raise ValueError("先载入分离结果")
        self.pause()
        if self.position >= len(next(iter(self.tracks.values()))):
            self.position = 0
        channels = next(iter(self.tracks.values())).shape[1]

        def callback(outdata, frames, _time, status):
            if status:
                self.underruns += 1
            audio = self.render(frames)
            outdata.fill(0)
            outdata[:len(audio)] = audio
            if len(audio) < frames:
                raise sd.CallbackStop

        def finished():
            self.playing = False

        self.stream = sd.OutputStream(samplerate=self.fs, channels=channels, dtype="float32",
                                      callback=callback, finished_callback=finished, blocksize=1024)
        self.playing = True
        try:
            self.stream.start()
        except Exception:
            self.pause()
            raise

    def pause(self):
        self.playing = False
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None

    def stop(self):
        self.pause()
        with self._lock:
            self.position = 0

    def mix(self):
        with self._lock:
            return mix_tracks(self.tracks, self.gains)
