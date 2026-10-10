"""双人语音分离的轻量任务控制与分块拼接；GUI 不加载 SpeechBrain。"""
from pathlib import Path
import json
import subprocess
import sys
import threading
import time
from uuid import uuid4
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "speechbrain/sepformer-wsj02mix"
REVISION = "3a28263"
MODEL_DIR = ROOT / "models" / "speakers" / "sepformer-wsj02mix"
OUTPUT_DIR = ROOT / "results" / "speakers"
REFERENCE_DIR = ROOT / "samples" / "speaker_reference"
LABELS = {"original": "混合原音", "speaker_1": "分离音轨1", "speaker_2": "分离音轨2"}


def speaker_python():
    candidate = ROOT / ".venv-speakers" / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    if not candidate.is_file():
        raise RuntimeError("双人分离环境尚未安装。Windows 请先运行 setup_speakers_windows.bat。")
    try:
        check = subprocess.run([str(candidate), "-c", "import sys; sys.exit(0 if sys.version_info[:2] == (3,12) else 2)"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=15,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        if check.returncode != 0:
            raise RuntimeError("双人分离环境是从其他电脑复制的或已失效，请在本机运行 setup_speakers_windows.bat 重建。")
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError("双人分离环境不能运行，请在本机运行 setup_speakers_windows.bat 重建。") from exc
    return str(candidate)


def model_ready(directory=MODEL_DIR):
    directory = Path(directory)
    return (directory / "model_manifest.json").is_file() and all(
        (directory / name).is_file() and (directory / name).stat().st_size > 0
        for name in ("hyperparams.yaml", "encoder.ckpt", "decoder.ckpt", "masknet.ckpt"))


def align_overlap(previous, current):
    """仅依据同一重叠时间窗的波形匹配两路，不提取声纹、不识别真实身份。"""
    a = np.asarray(previous, np.float64)
    b = np.asarray(current, np.float64)
    if a.shape != b.shape or a.ndim != 2 or a.shape[1] != 2:
        raise ValueError("重叠音轨必须为相同长度的两列")
    a = a - a.mean(axis=0)
    b = b - b.mean(axis=0)
    norms = np.linalg.norm(a, axis=0)[:, None] * np.linalg.norm(b, axis=0)[None, :]
    correlation = np.abs(a.T @ b) / np.maximum(norms, 1e-12)
    direct = correlation[0, 0] + correlation[1, 1]
    swapped = correlation[0, 1] + correlation[1, 0]
    swap = bool(swapped > direct + .05)
    return current[:, ::-1].copy() if swap else current, swap


def separate_chunks(audio, infer, fs=8000, chunk_seconds=4, overlap_seconds=1, progress=None):
    """有限块长推理及交叉淡化；缺乏有效重叠时无法保证匿名轨道顺序。"""
    audio = np.asarray(audio, np.float32)
    if audio.ndim != 1 or not len(audio) or not np.isfinite(audio).all():
        raise ValueError("输入应为非空有限的单声道波形")
    block, overlap = int(fs * chunk_seconds), int(fs * overlap_seconds)
    if not 0 < overlap < block:
        raise ValueError("重叠时长须大于零且小于分块时长")
    result, weight_sum = np.zeros((len(audio), 2), np.float64), np.zeros((len(audio), 1), np.float64)
    previous = None
    swaps = 0
    for start in range(0, len(audio), block - overlap):
        end = min(len(audio), start + block)
        if progress:
            progress(start, len(audio))
        estimate = np.asarray(infer(audio[start:end]), np.float32)
        if estimate.shape != (end - start, 2) or not np.isfinite(estimate).all():
            raise ValueError("分离模型输出必须为原长度的两条有限音轨")
        if previous is not None:
            size = min(overlap, len(estimate), len(previous))
            _, swap = align_overlap(previous[-size:], estimate[:size])
            if swap:
                estimate = estimate[:, ::-1].copy()
                swaps += 1
        weights = np.ones(len(estimate))
        if start:
            size = min(overlap, len(estimate))
            weights[:size] *= (np.arange(size) + 1) / (size + 1)
        if end < len(audio):
            weights[-overlap:] *= np.arange(overlap, 0, -1) / (overlap + 1)
        result[start:end] += estimate * weights[:, None]
        weight_sum[start:end] += weights[:, None]
        previous = estimate
        if end == len(audio):
            break
    if progress:
        progress(len(audio), len(audio))
    return (result / weight_sum).astype(np.float32), swaps


class SpeakerJob:
    def __init__(self, emit):
        self.emit = emit
        self.id = uuid4().hex
        self.output = OUTPUT_DIR / (time.strftime("%Y%m%d_%H%M%S") + "_" + self.id[:8])
        self.process = None
        self.worker_name = "speaker_worker.py"
        self.cancelled, self.finished, self.completed = threading.Event(), threading.Event(), threading.Event()
        self.lock = threading.Lock()

    def start(self, input_path, device="auto", source="hf-mirror", download_only=False):
        python = speaker_python()
        if not download_only and not Path(input_path).is_file():
            raise FileNotFoundError(input_path)
        self.output.mkdir(parents=True, exist_ok=False)
        request = {"input": str(input_path), "task_id": self.id, "model": MODEL_ID, "device": device,
                   "source": source, "download_only": download_only, "model_dir": str(MODEL_DIR), "output_dir": str(self.output)}
        path = self.output / "request.json"
        path.write_text(json.dumps(request, ensure_ascii=False, indent=2), encoding="utf-8")
        threading.Thread(target=self._run, args=(python, path), daemon=True).start()

    def _send(self, event):
        self.emit({**event, "task_id": self.id, "job": self})

    def _run(self, python, request):
        try:
            with self.lock:
                if self.cancelled.is_set():
                    return
                self.process = subprocess.Popen([python, "-X", "utf8", "-u", str(ROOT / self.worker_name), str(request)],
                    cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                    encoding="utf-8", errors="replace", creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            with (self.output / "worker.log").open("w", encoding="utf-8") as log:
                for line in self.process.stdout:
                    log.write(line)
                    log.flush()
                    if self.cancelled.is_set():
                        continue
                    if line.startswith("DSP_EVENT "):
                        event = json.loads(line[len("DSP_EVENT "):])
                        if event["type"] == "done":
                            self.completed.set()
                        self._send(event)
                    elif line.strip():
                        self._send({"type": "log", "message": line.strip()[-300:]})
            code = self.process.wait()
            if not self.cancelled.is_set() and (code != 0 or not self.completed.is_set()):
                self._send({"type": "error", "message": f"双人分离进程退出码 {code}，详见本次 worker.log"})
        except Exception as exc:
            if not self.cancelled.is_set():
                self._send({"type": "error", "message": str(exc)})
        finally:
            with self.lock:
                if self.process is not None and self.process.poll() is None:
                    self.process.terminate()
                    self.process.wait()
            if self.cancelled.is_set():
                (self.output / "result.json").unlink(missing_ok=True)
                (self.output / "status.json").write_text('{"state":"cancelled"}', encoding="utf-8")
                self._send({"type": "cancelled", "message": "双人分离任务已取消"})
            self.finished.set()

    def cancel(self):
        if self.completed.is_set() or self.finished.is_set():
            return
        self.cancelled.set()
        with self.lock:
            if self.process is not None and self.process.poll() is None:
                self.process.terminate()


class SpeakerPostJob(SpeakerJob):
    """后处理使用当前基础环境的 ONNX 模型，沿用独立进程取消与事件协议。"""
    def __init__(self, emit):
        super().__init__(emit)
        self.worker_name = "speaker_postprocess_worker.py"

    def start(self, source_result, configs):
        from .speaker_cleanup import validate_config
        if source_result.get("kind") != "speaker_separation":
            raise ValueError("先载入双人分离结果")
        configs = {name: validate_config(configs[name]) for name in ("speaker_1", "speaker_2")}
        files = source_result.get("raw_files", source_result["files"])
        if any(not Path(path).is_file() for path in files.values()):
            raise FileNotFoundError("原始分离音轨不在磁盘，请重新载入或重新分离")
        self.output.mkdir(parents=True, exist_ok=False)
        request = {"task_id": self.id, "output_dir": str(self.output),
                   "source_result": source_result, "configs": configs}
        path = self.output / "request.json"
        path.write_text(json.dumps(request, ensure_ascii=False, indent=2), encoding="utf-8")
        threading.Thread(target=self._run, args=(sys.executable, path), daemon=True).start()
