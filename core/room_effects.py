"""房间混响及离线 WPE 去混响；仅依赖 NumPy / SciPy。

WPE 使用延迟线性预测估计晚期混响，并非从任意录音精确恢复干声。
算法背景：https://nara-wpe.readthedocs.io/en/latest/
"""
from __future__ import annotations

from math import gcd
import numpy as np
from scipy import signal, ndimage


def _audio(audio, fs):
    y = np.asarray(audio, dtype=np.float64)
    if y.ndim not in (1, 2) or not y.size or (y.ndim == 2 and y.shape[1] not in (1, 2)):
        raise ValueError("需要非空的单声道或立体声音频")
    if not np.isfinite(y).all() or not np.isfinite(fs) or fs != int(fs) or not 8000 <= fs <= 192000:
        raise ValueError("音频必须为有限值，采样率须为 8000～192000 Hz 的整数")
    return y[:, None] if y.ndim == 1 else y, y.ndim == 1, int(fs)


def _range(value, lower, upper, name):
    if not np.isfinite(value) or not lower <= value <= upper:
        raise ValueError(f"{name}须在 {lower}～{upper} 之间")


def _finish(y, mono):
    peak = float(np.max(np.abs(y)))
    if peak > .99:
        y = y * (.99 / peak)  # 对全部声道使用同一增益，防止导出削顶。
    return (y[:, 0] if mono else y).astype(np.float32)


def room_impulse(fs, rt60=1.0, pre_delay_ms=20.0, damping=.5, seed=0):
    """确定性的早期反射 + 衰减扩散尾音，RT60 指包络降低 60 dB 的时间。"""
    _range(rt60, .1, 5, "衰减时间")
    _range(pre_delay_ms, 0, 200, "预延迟毫秒")
    _range(damping, 0, 1, "高频衰减")
    offset = round(pre_delay_ms * fs / 1000)
    n = max(2, round(rt60 * 1.2 * fs))
    rng = np.random.default_rng(seed)
    tail = rng.standard_normal(n)
    cutoff = min(fs * .45, 1000 + (1 - damping) * 9000)
    tail = signal.sosfilt(signal.butter(2, cutoff, fs=fs, output="sos"), tail)
    tail *= np.exp(-np.log(1000) * np.arange(n) / (fs * rt60))
    # 逐渐出现扩散尾音，避免一开始的随机尖峰遮住早期反射。
    tail *= np.minimum(1, np.arange(n) / max(1, fs * .02))
    tail /= max(np.linalg.norm(tail), 1e-12)
    ir = np.pad(tail * .65, (offset, 0))
    for seconds, gain in zip((.011, .019, .029, .043, .071, .113), (.65, .45, .35, .27, .18, .12)):
        index = offset + round(seconds * fs)
        if index < len(ir):
            ir[index] += gain * np.exp(-np.log(1000) * seconds / rt60)
    return ir / max(np.linalg.norm(ir), 1e-12)


def add_reverb(audio, fs, rt60=1.0, wet=.3, pre_delay_ms=20.0, damping=.5):
    """混合干声与房间响应；保留原时长、采样率、声道，末尾超出部分截断。"""
    y, mono, fs = _audio(audio, fs)
    _range(wet, 0, 1, "湿声比例")
    ir = room_impulse(fs, rt60, pre_delay_ms, damping)
    if wet == 0 or not np.any(y):
        return (y[:, 0] if mono else y).astype(np.float32)
    out = np.empty_like(y)
    for channel in range(y.shape[1]):
        channel_ir = ir if channel == 0 else room_impulse(fs, rt60, pre_delay_ms, damping, seed=1)
        reverberated = signal.fftconvolve(y[:, channel], channel_ir)[:len(y)]
        out[:, channel] = (1 - wet) * y[:, channel] + wet * reverberated
    return _finish(out, mono)


def _wpe_block(y, taps, delay, iterations):
    """Y = F×C×T；用加权协方差求延迟预测滤波器，迭代更新功率权重。"""
    _, _, spec = signal.stft(y.T, fs=16000, nperseg=512, noverlap=384,
                             boundary="zeros", padded=True, axis=-1)
    observed = spec.transpose(1, 0, 2).astype(np.complex128)
    start = delay + taps - 1
    if observed.shape[-1] - start < 2 * taps * y.shape[1] + 1:
        return y.copy()
    output = observed.copy()
    indices = np.arange(start, observed.shape[-1])[None, :] - delay - np.arange(taps)[:, None]
    # 分频段构建预测矩阵，限制长录音的临时内存。
    for first in range(0, len(observed), 16):
        target = observed[first:first + 16]
        past = target[:, :, indices].reshape(len(target), taps * y.shape[1], -1)
        enhanced = target.copy()
        for _ in range(iterations):
            power = np.mean(np.abs(enhanced) ** 2, axis=1)
            power = ndimage.uniform_filter1d(power, 3, axis=-1, mode="nearest")
            floor = np.maximum(power.max(axis=-1, keepdims=True) * 1e-6, 1e-12)
            weight = 1 / np.maximum(power[:, start:], floor)
            covariance = np.einsum("fqt,frt,ft->fqr", past, past.conj(), weight, optimize=True)
            cross = np.einsum("fqt,fct,ft->fqc", past, target[:, :, start:].conj(), weight, optimize=True)
            size = covariance.shape[-1]
            ridge = np.maximum(np.trace(covariance, axis1=-2, axis2=-1).real / size * 1e-4, 1e-12)
            covariance += ridge[:, None, None] * np.eye(size)
            coefficients = np.linalg.solve(covariance, cross)
            predicted = np.einsum("fqc,fqt->fct", coefficients.conj(), past, optimize=True)
            enhanced[:, :, start:] = target[:, :, start:] - predicted
        output[first:first + 16] = enhanced
    _, reconstructed = signal.istft(output.transpose(1, 0, 2), fs=16000, nperseg=512,
                                    noverlap=384, input_onesided=True, boundary=True)
    return reconstructed.T[:len(y)]


def dereverb(audio, fs, strength=.7, taps=10, delay=3, iterations=3):
    """分块离线 WPE，单/双声道；强度 0 原样返回，1 使用完整 WPE 输出。

    在 16 kHz 处理，每块最多 8 秒、相邻块交叉淡化 1 秒；恢复原率和原长度。
    静音或不足以求稳定预测的极短素材原样返回。
    """
    y, mono, fs = _audio(audio, fs)
    _range(strength, 0, 1, "去混响强度")
    for value, lo, hi, name in ((taps, 3, 30, "预测阶数"), (delay, 1, 10, "预测延迟"), (iterations, 1, 5, "迭代次数")):
        _range(value, lo, hi, name)
        if value != int(value):
            raise ValueError(f"{name}须为整数")
    taps, delay, iterations = int(taps), int(delay), int(iterations)
    if strength == 0 or len(y) / fs < .4 or np.max(np.abs(y)) < 1e-10:
        return (y[:, 0] if mono else y).astype(np.float32)
    g = gcd(fs, 16000)
    work = signal.resample_poly(y, 16000 // g, fs // g, axis=0) if fs != 16000 else y
    combined, weights = np.zeros_like(work), np.zeros((len(work), 1))
    block, overlap = 8 * 16000, 16000
    for first in range(0, len(work), block - overlap):
        end = min(first + block, len(work))
        piece = work[first:end]
        enhanced = _wpe_block(piece, taps, delay, iterations)
        weight = np.ones(len(piece))
        if first:
            size = min(overlap, len(piece))
            weight[:size] *= (np.arange(size) + 1) / (size + 1)
        if end < len(work):
            weight[-overlap:] *= np.arange(overlap, 0, -1) / (overlap + 1)
        combined[first:end] += enhanced * weight[:, None]
        weights[first:end] += weight[:, None]
        if end == len(work):
            break
    out = (1 - strength) * work + strength * (combined / weights)
    if fs != 16000:
        out = signal.resample_poly(out, fs // g, 16000 // g, axis=0)
    out = out[:len(y)]
    if len(out) < len(y):
        out = np.pad(out, ((0, len(y) - len(out)), (0, 0)))
    if not np.isfinite(out).all():
        raise RuntimeError("去混响输出异常，请降低强度或使用更长的素材")
    return _finish(out, mono)
