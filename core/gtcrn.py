"""基于 GTCRN 的神经网络降噪（onnxruntime 推理，无需 PyTorch）。

模型来源与许可
--------------
GTCRN（Grouped Temporal Convolutional Recurrent Network）是 ICASSP 2024 论文
"GTCRN: A Speech Enhancement Model Requiring Ultralow Computational Resources"
的官方实现，48.2K 参数 / 33 MMACs，主打**超低算力**的实时语音增强：

    论文/代码：https://github.com/Xiaobin-Rong/gtcrn   （MIT 许可）
    模型文件：stream/onnx_models/gtcrn_simple.onnx（在 DNS3 数据上训练）
    国内镜像：https://hf-mirror.com/andrelgomes/gtcrn-stream（官方文件的逐字节镜像）

为什么用它
----------
1. **原生 16 kHz**，与本项目采样率一致，不需要重采样；
2. 纯 onnxruntime 推理，**不需要装 PyTorch**（本机 Python 3.14 装不上 torch）；
3. 只有 0.5 MB、24K 参数，CPU 上实时（比本项目自训的 MLP 还轻）；
4. 它是**在真实大规模噪声数据（DNS3）上训练的**，正好补上"我们的干净语音只有 25 秒"
   这个短板——相当于直接借用别人用几千小时数据训好的成果。

官方推理流程（本项目据此用 NumPy 复刻，见 models/gtcrn/ref/）：
    x -> STFT(512, hop 256, sqrt-Hann 窗) -> (实部, 虚部) [1,257,T,2]
      -> GTCRN 流式网络（含 conv/tra/inter 三组状态缓存）-> 复数掩蔽
      -> 复数乘法 -> ISTFT(512, 256, sqrt-Hann) -> 降噪波形
"""

from __future__ import annotations

import os
import threading

import numpy as np

try:
    import onnxruntime as ort
except Exception:  # pragma: no cover
    ort = None

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MODEL = os.path.join(HERE, "models", "gtcrn", "gtcrn_simple.onnx")

N_FFT = 512
HOP = 256
N_BINS = N_FFT // 2 + 1          # 257
SR = 16000


def _sqrt_hann(n: int) -> np.ndarray:
    """官方用的是 torch.hann_window(512).pow(0.5)；periodic Hann 与 numpy.hanning 一致。"""
    return np.sqrt(np.hanning(n)).astype(np.float32)


def _resample_poly(x: np.ndarray, fs_up: int, fs_down: int, length: int | None = None) -> np.ndarray:
    """有理数重采样（fs_up/fs_down），并保证输出长度符合预期。

    实测教训：用 scipy.signal.resample（FFT 法）时若目标长度算错，会出现
    "时长被压缩"的严重错误（48 kHz 输入变成 1/3 时长，听感像加速播放）。
    这里用 resample_poly + 显式长度对齐，上/下行严格对称。
    """
    from math import gcd

    from scipy import signal as sps

    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if fs_up == fs_down:
        out = x
    else:
        g = gcd(int(fs_up), int(fs_down))
        out = sps.resample_poly(x.astype(np.float64), fs_up // g, fs_down // g).astype(np.float32)
    if length is not None:
        if len(out) > length:
            out = out[:length]
        elif len(out) < length:
            out = np.pad(out, (0, length - len(out)))
    return out


class GTCRNDenoiser:
    """GTCRN 降噪器（流式状态管理 + 分块推理）。"""

    def __init__(self, model_path: str | None = None, chunk_frames: int = 64):
        self.model_path = model_path or DEFAULT_MODEL
        self.chunk_frames = max(1, int(chunk_frames))
        self.window = _sqrt_hann(N_FFT)
        self.session = None
        self.load_error: str | None = None
        self._caches: dict[str, np.ndarray] | None = None
        self._inference_lock = threading.RLock()
        self.reload()

    # -- 状态 ---------------------------------------------------------------
    @property
    def is_ready(self) -> bool:
        return self.session is not None

    def reload(self) -> bool:
        self.session = None
        if ort is None:
            self.load_error = "未安装 onnxruntime（pip install onnxruntime）"
            return False
        if not os.path.isfile(self.model_path):
            self.load_error = f"模型文件不存在：{self.model_path}"
            return False
        try:
            opts = ort.SessionOptions()
            opts.log_severity_level = 3
            self.session = ort.InferenceSession(self.model_path, opts,
                                                providers=["CPUExecutionProvider"])
            self.load_error = None
            return True
        except Exception as exc:
            self.session = None
            self.load_error = str(exc)
            return False

    def describe(self) -> str:
        if not self.is_ready:
            return f"GTCRN 未加载（{self.load_error}）"
        mb = os.path.getsize(self.model_path) / 1048576
        return (f"GTCRN（ICASSP 2024，DNS3 上训练）{mb:.2f} MB · 16 kHz · "
                f"onnxruntime {ort.__version__} · 分块 {self.chunk_frames} 帧")

    def reset(self) -> None:
        """清空流式状态（换成新音频时要调用）。"""
        self._caches = {
            "conv_cache": np.zeros([2, 1, 16, 16, 33], dtype=np.float32),
            "tra_cache": np.zeros([2, 3, 1, 1, 16], dtype=np.float32),
            "inter_cache": np.zeros([2, 1, 33, 16], dtype=np.float32),
        }

    # -- STFT / ISTFT（复刻官方 torch.stft / torch.istft 的用法）-------------
    def stft(self, x: np.ndarray) -> np.ndarray:
        """返回 (n_bins, n_frames, 2)，最后一维是 (实部, 虚部)。"""
        x = np.asarray(x, dtype=np.float32).reshape(-1)
        if len(x) < N_FFT:
            x = np.pad(x, (0, N_FFT - len(x)))
        n_frames = 2 + (max(0, len(x) - N_FFT) + HOP - 1) // HOP
        x = np.pad(x, (0, (n_frames - 1) * HOP + N_FFT - len(x)))
        idx = np.arange(N_FFT)[None, :] + HOP * np.arange(n_frames)[:, None]
        frames = x[idx] * self.window[None, :]
        spec = np.fft.rfft(frames, n=N_FFT, axis=1)          # (n_frames, n_bins)
        return np.stack([spec.real, spec.imag], axis=-1).transpose(1, 0, 2).astype(np.float32)

    def istft(self, spec: np.ndarray, length: int | None = None) -> np.ndarray:
        """(n_bins, n_frames, 2) -> 波形（加权重叠相加）。"""
        spec = np.asarray(spec)
        complex_spec = spec[..., 0] + 1j * spec[..., 1]
        frames = np.fft.irfft(complex_spec.T, n=N_FFT, axis=1) * self.window[None, :]
        n_frames = frames.shape[0]
        out = np.zeros((n_frames - 1) * HOP + N_FFT, dtype=np.float64)
        wsum = np.zeros_like(out)
        wsq = self.window.astype(np.float64) ** 2
        for i in range(n_frames):
            s = i * HOP
            out[s:s + N_FFT] += frames[i]
            wsum[s:s + N_FFT] += wsq
        wsum[wsum < 1e-8] = np.inf
        out = out / wsum
        out[~np.isfinite(out)] = 0.0
        if length is not None:
            out = out[:length] if len(out) >= length else np.pad(out, (0, length - len(out)))
        return out.astype(np.float32)

    # -- 推理 ---------------------------------------------------------------
    def _run(self, spec: np.ndarray) -> np.ndarray:
        """把 (n_bins, T, 2) 送进网络，返回同样形状的增强谱。

        这个导出的 ONNX 图是**逐帧流式**的：`mix` 的 rank 必须是 4 且形状为
        (1, 257, 1, 2)——时间轴恒为 1，靠 conv/tra/inter 三组状态缓存承载历史。
        所以这里按帧循环前向（实测 609 帧约 0.3 s，完全够用），
        这也正是它"超低算力、可实时"的原因。
        """
        assert self._caches is not None, "请先调用 reset()"
        n_frames = spec.shape[1]
        outs = []
        for t in range(n_frames):
            mix = spec[None, :, t, :][:, :, None, :]         # (1, n_bins, 1, 2)
            enh, cc, tc, ic = self.session.run(
                ["enh", "conv_cache_out", "tra_cache_out", "inter_cache_out"], {
                    "mix": np.ascontiguousarray(mix, dtype=np.float32),
                    "conv_cache": self._caches["conv_cache"],
                    "tra_cache": self._caches["tra_cache"],
                    "inter_cache": self._caches["inter_cache"],
                })
            self._caches["conv_cache"], self._caches["tra_cache"], self._caches["inter_cache"] = cc, tc, ic
            outs.append(enh[0, :, 0, :])                     # (n_bins, 2)
        return np.stack(outs, axis=1)                        # (n_bins, T, 2)

    def enhance(self, y: np.ndarray, fs: int = SR, chunked: bool = True) -> np.ndarray:
        # recurrent caches 属于一次音频推理，不能让多个线程交叉 reset/run。
        with self._inference_lock:
            return self._enhance_locked(y, fs, chunked)

    def _enhance_locked(self, y: np.ndarray, fs: int = SR, chunked: bool = True) -> np.ndarray:
        """对波形降噪，**返回与输入等长、且采样率仍为 fs 的波形**。

        - 输入先重采样到模型原生 16 kHz，处理后恢复原采样率。
          这里必须用 resample_poly（有理数重采样）并保证上下行严格对称——
          早期版本用 resample() 且把目标长度算错，导致 48 kHz 输入被压成 1/3 时长
          （听感上就是"加速 3 倍"），并连带污染了串联降噪的残差。
        - chunked：是否分块前向（模拟流式）。短音频可设 False 一次性前向。
        """
        if not self.is_ready:
            raise RuntimeError(f"GTCRN 不可用：{self.load_error}")
        y = np.asarray(y, dtype=np.float32).reshape(-1)
        if not len(y) or not np.isfinite(y).all() or int(fs) <= 0:
            raise ValueError("音频为空、包含非有限值或采样率无效")
        need_resample = int(fs) != SR
        if need_resample:
            y48 = _resample_poly(y, SR, int(fs))
        else:
            y48 = y
        self.reset()
        spec = self.stft(y48)
        n_frames = spec.shape[1]
        if chunked and n_frames > self.chunk_frames:
            parts = [self._run(spec[:, i:i + self.chunk_frames, :])
                     for i in range(0, n_frames, self.chunk_frames)]
            enh = np.concatenate(parts, axis=1)
        else:
            enh = self._run(spec)
        out48 = self.istft(enh, length=len(y48))
        if need_resample:
            out = _resample_poly(out48, int(fs), SR, length=len(y))
        else:
            out = out48
        return out.astype(np.float32)


# ---------------------------------------------------------------------------
# 与 test_wavs 官方配对样本复核用
# ---------------------------------------------------------------------------
def verify_against_reference(ref_dir: str | None = None) -> dict:
    """用官方 stream/test_wavs/{mix,enh}.wav 验证实现是否与官方一致。

    这是最有力的正确性证据：如果我们的 NumPy 实现正确，输出应当与官方
    enh.wav 高度吻合（相关系数接近 1）。
    """
    from core import audio_io

    ref_dir = ref_dir or os.path.join(HERE, "models", "gtcrn", "ref")
    mix_p = os.path.join(ref_dir, "stream__test_wavs__mix.wav")
    enh_p = os.path.join(ref_dir, "stream__test_wavs__enh.wav")
    if not (os.path.isfile(mix_p) and os.path.isfile(enh_p)):
        return {"ok": False, "error": f"缺少官方测试音频（{ref_dir}）"}
    mix, _ = audio_io.load_audio(mix_p)
    ref, _ = audio_io.load_audio(enh_p)
    mix, ref = audio_io.to_mono(mix), audio_io.to_mono(ref)
    den = GTCRNDenoiser()
    if not den.is_ready:
        return {"ok": False, "error": den.load_error}
    out = den.enhance(mix, chunked=False)
    n = min(len(out), len(ref))
    a, b = out[:n], ref[:n]
    corr = float(np.corrcoef(a, b)[0, 1])
    rel = float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-9))
    return {"ok": True, "corr": corr, "rel_err": rel,
            "rms_in": float(np.sqrt(np.mean(mix ** 2))), "rms_out": float(np.sqrt(np.mean(out ** 2)))}
