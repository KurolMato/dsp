"""音频读写与录音/播放。

对应原 MATLAB 代码：
    pushbutton1_Callback  -> record()
    pushbutton2_Callback  -> load_audio()
    pushbutton3_Callback  -> play()

设计原则：优先使用第三方库（sounddevice / soundfile），
它们缺失时自动退化为标准库（wave / winsound）实现，保证程序仍可运行。
"""

from __future__ import annotations

import os
import sys
import threading
import time
import wave

import numpy as np

# ----------------------------------------------------------------------------
# 可选依赖
# ----------------------------------------------------------------------------
try:  # 录音与播放（跨平台，Windows 下自带 PortAudio）
    import sounddevice as sd
except Exception:  # pragma: no cover - 取决于运行环境
    sd = None

try:  # 更多音频格式（mp3 / m4a / flac ...）
    import soundfile as sf
except Exception:  # pragma: no cover
    sf = None

try:  # Windows 标准库播放（WAV）
    import winsound
except Exception:  # pragma: no cover - 非 Windows
    winsound = None


DEFAULT_FS = 16000  # 与原 MATLAB 一致


# ----------------------------------------------------------------------------
# 数据格式约定
# ----------------------------------------------------------------------------
def to_float(x: np.ndarray) -> np.ndarray:
    """把任意整型/浮点音频转成 MATLAB 风格的 float32 [-1, 1]。"""
    x = np.asarray(x)
    if x.dtype.kind == "i":
        info = np.iinfo(x.dtype)
        scale = float(max(abs(info.min), info.max))
        return (x.astype(np.float32) / scale).astype(np.float32)
    if x.dtype.kind == "u":
        info = np.iinfo(x.dtype)
        mid = (info.max + 1) / 2.0
        return ((x.astype(np.float32) - mid) / mid).astype(np.float32)
    return x.astype(np.float32, copy=False)


def to_mono(y: np.ndarray) -> np.ndarray:
    """多声道取平均（MATLAB 里用 y(:,1) 取第一声道，这里取均值更稳）。"""
    y = np.asarray(y, dtype=np.float32)
    if y.ndim == 2:
        y = y.mean(axis=1)
    return y.astype(np.float32, copy=False)


def normalize(y: np.ndarray, peak: float = 0.99) -> np.ndarray:
    """峰值归一化，防止 int16 写出时削顶。"""
    y = np.asarray(y, dtype=np.float32)
    m = float(np.max(np.abs(y))) if y.size else 0.0
    if m > peak:
        y = y * (peak / m)
    return y


# ----------------------------------------------------------------------------
# WAV 读写：标准库实现（始终可用）
# ----------------------------------------------------------------------------
def _read_wav_stdlib(path: str) -> tuple[np.ndarray, int]:
    with wave.open(path, "rb") as w:
        n_ch, width, fs, n_frames = w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()
        raw = w.readframes(n_frames)
    if width == 1:
        data = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        data = (data - 128.0) / 128.0
    elif width == 2:
        data = np.frombuffer(raw, dtype="<i2")
        data = data.astype(np.float32) / 32768.0
    elif width == 4:
        data = np.frombuffer(raw, dtype="<i4")
        data = data.astype(np.float32) / 2147483648.0
    else:  # pragma: no cover
        raise ValueError(f"不支持的 WAV 位深：{width * 8} bit")
    if n_ch > 1:
        data = data.reshape(-1, n_ch)
    return data.astype(np.float32), fs


def _write_wav_stdlib(path: str, y: np.ndarray, fs: int) -> None:
    y = normalize(y)
    data = np.clip(y * 32767.0, -32768, 32767).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(1 if data.ndim == 1 else data.shape[1])
        w.setsampwidth(2)
        w.setframerate(int(fs))
        w.writeframes(data.tobytes())


def load_audio(path: str) -> tuple[np.ndarray, int]:
    """读取音频文件，返回 (float32 数据, 采样率)。多声道返回二维数组（与 audioread 一致）。"""
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    ext = os.path.splitext(path)[1].lower()
    if sf is not None and ext not in (".wav", ".wave"):
        try:
            data, fs = sf.read(path, dtype="float32", always_2d=False)
            return np.asarray(data, dtype=np.float32), int(fs)
        except Exception:
            return _read_media_av(path)
    try:
        return _read_wav_stdlib(path)
    except Exception:
        if sf is None:
            raise
        data, fs = sf.read(path, dtype="float32", always_2d=False)
        return np.asarray(data, dtype=np.float32), int(fs)


def _read_media_av(path: str) -> tuple[np.ndarray, int]:
    """PyAV 补充 M4A 等 libsndfile 未支持的格式；保持声道数。"""
    try:
        import av
    except ImportError as exc:
        raise RuntimeError("此格式需要 PyAV，请安装 requirements-music.txt 或转换为 WAV") from exc
    with av.open(path) as container:
        if not container.streams.audio:
            raise ValueError("文件中没有音频流")
        stream = container.streams.audio[0]
        fs = int(stream.codec_context.sample_rate)
        resampler = av.AudioResampler(format="fltp", layout=stream.codec_context.layout, rate=fs)
        chunks = []
        for frame in container.decode(stream):
            chunks.extend(item.to_ndarray() for item in resampler.resample(frame))
        chunks.extend(item.to_ndarray() for item in resampler.resample(None))
    if not chunks:
        raise ValueError("文件没有可解码的音频采样")
    audio = np.concatenate(chunks, axis=1).T.astype(np.float32)
    return (audio[:, 0] if audio.shape[1] == 1 else audio), fs


def save_audio(path: str, y: np.ndarray, fs: int) -> None:
    """保存为 16-bit WAV。"""
    y = np.asarray(y, dtype=np.float32)
    if y.ndim == 2 and y.shape[1] == 1:
        y = y[:, 0]
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    if sf is not None:
        sf.write(path, normalize(y), int(fs), subtype="PCM_16")
    else:
        _write_wav_stdlib(path, y, fs)


# ----------------------------------------------------------------------------
# 播放
# ----------------------------------------------------------------------------
def play(y: np.ndarray, fs: int, blocking: bool = False) -> None:
    """播放音频。blocking=True 时播放结束才返回（对应 MATLAB 的 sound 行为）。"""
    y = np.asarray(y, dtype=np.float32)
    y = normalize(y)
    if y.size == 0:
        return
    if sd is not None:
        try:
            sd.play(y, int(fs))
            if blocking:
                sd.wait()
            return
        except Exception:
            pass
    if sys.platform.startswith("win"):
        tmp = os.path.join(os.environ.get("TEMP", "."), "dsp_py_play.wav")
        _write_wav_stdlib(tmp, y, fs)
        if blocking or winsound is None:
            if winsound is not None:
                winsound.PlaySound(tmp, winsound.SND_FILENAME)
        else:
            winsound.PlaySound(tmp, winsound.SND_FILENAME | winsound.SND_ASYNC)
        return
    raise RuntimeError("没有可用的音频播放后端（请安装 sounddevice）")


def stop_play() -> None:
    """停止当前播放。"""
    if sd is not None:
        try:
            sd.stop()
        except Exception:
            pass
    if sys.platform.startswith("win") and winsound is not None:
        try:
            winsound.PlaySound(None, winsound.SND_PURGE)
        except Exception:
            pass


# ----------------------------------------------------------------------------
# 录音（对应 pushbutton1：16 kHz / 4 s / 16 bit / 双声道）
# ----------------------------------------------------------------------------
def list_input_devices() -> list[str]:
    if sd is None:
        return []
    out = []
    for i, d in enumerate(sd.query_devices()):
        if d.get("max_input_channels", 0) > 0:
            out.append(f"[{i}] {d['name']} ({d['max_input_channels']} ch)")
    return out


def record(duration: float = 4.0, fs: int = DEFAULT_FS, channels: int = 1,
           device=None, on_progress=None) -> np.ndarray:
    """录音，返回 shape=(N,) 或 (N, ch) 的 float32 数据。

    on_progress(elapsed, total) 可选回调，用于 GUI 显示剩余秒数。
    """
    if sd is None:
        raise RuntimeError("录音需要 sounddevice：pip install sounddevice")
    if channels not in (1, 2):
        channels = 1
    fs = int(fs)
    n = int(round(duration * fs))
    done = threading.Event()
    box: dict[str, np.ndarray] = {}

    def _worker() -> None:
        try:
            rec = sd.rec(n, samplerate=fs, channels=channels, dtype="float32", device=device)
            t0 = time.time()
            # 不用 sd.get_stream()（可能抛异常），直接按时间等录音结束
            while (time.time() - t0) < duration + 0.3:
                if on_progress is not None:
                    on_progress(min(time.time() - t0, duration), duration)
                time.sleep(0.05)
            sd.wait()
            box["data"] = rec
        finally:
            done.set()

    th = threading.Thread(target=_worker, daemon=True)
    th.start()
    done.wait(timeout=duration + 10.0)
    if "data" not in box:
        raise RuntimeError("录音失败或被中断（请检查麦克风是否被占用）")
    data = np.asarray(box["data"], dtype=np.float32)
    if channels == 2:
        # 与 MATLAB nChannels=2 一致：双声道保留，主流程再取单声道
        return data
    return data.reshape(-1)


# ----------------------------------------------------------------------------
# 便捷：带噪样本（供 GUI 演示“没有干净参考时”的场景）
# ----------------------------------------------------------------------------
def default_sample_path() -> str | None:
    """项目自带的示例音频；找不到返回 None。

    查找顺序（先项目内、后项目外）：
        1. samples/demo_speech.wav —— 项目自带的干净语音（随项目分发，换电脑也有）
        2. samples/data.wav        —— 用户放进 samples/ 的素材
        3. dsp_py/data.wav         —— 「① 录音」的工作副本（可能被覆盖，故排在后面）
        4. 上级目录的 data.wav / manba.wav —— 原始 MATLAB 素材
    """
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for name in ("demo_speech.wav", "data.wav"):
        p = os.path.join(here, "samples", name)
        if os.path.isfile(p):
            return p
    for root in (here, os.path.dirname(here)):
        for name in ("data.wav", "manba.wav"):
            p = os.path.join(root, name)
            if os.path.isfile(p):
                return p
    return None
