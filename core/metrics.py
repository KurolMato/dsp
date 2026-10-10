"""无参考降噪评估：手头只有一段带噪录音时，如何量化"降噪到底有没有效果"。

难点在于：`eval_compare.py` / `bench_models.py` 之所以能算 ΔSNR、对数谱失真（LSD），
靠的是合成数据里已知的**干净参考**。但真实使用场景往往只有一段录音（比如课堂演示、
现场录制），没有干净对照，此时需要"无参考指标"：

    高低能量对比 —— 高能量帧 / 低能量帧能量比的变化；不能视作真实 SNR
    LSD_self   —— 降噪前后对数谱的失真（衡量改动了多少，越小越保守）
    噪声底下降 —— 静音段（能量最低的若干帧）能量下降 dB，直接对应"底噪变小了"
    语音保留度 —— 高能量帧的能量保留比例，用来发现"降噪把人声也削掉了"

这些都是工程上常用的近似手段，比"听起来安静了"要客观，写进报告也说得清。
"""

from __future__ import annotations

import numpy as np

from . import dsp


def _frame_energy(x: np.ndarray, fs: int, frame: float = 0.02, hop: float = 0.01) -> np.ndarray:
    fl, hs = int(frame * fs), int(hop * fs)
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if len(x) < fl:
        return np.array([np.sum(x ** 2)])
    idx = np.arange(fl)[None, :] + hs * np.arange((len(x) - fl) // hs + 1)[:, None]
    return np.sum(x[idx] ** 2, axis=1)


def assess(noisy: np.ndarray, denoised: np.ndarray, fs: int = 16000,
           quiet_ratio: float = 0.2, loud_ratio: float = 0.2) -> dict:
    """返回高低能量对比和频谱改变量；长度错误会直接拒绝评估。"""
    noisy = np.asarray(noisy, dtype=np.float32).reshape(-1)
    denoised = np.asarray(denoised, dtype=np.float32).reshape(-1)
    if not len(noisy) or len(noisy) != len(denoised):
        raise ValueError("评估要求原音与处理结果等长且非空，不能自动拉伸对齐")
    if int(fs) <= 0 or not np.isfinite(noisy).all() or not np.isfinite(denoised).all():
        raise ValueError("音频数值或采样率无效")
    if not 0 < quiet_ratio < 0.5 or not 0 < loud_ratio < 0.5:
        raise ValueError("高低能量帧比例必须在 0~0.5 之间")
    length_ratio = 1.0
    n = len(noisy)
    a, b = noisy, denoised
    e_a = _frame_energy(a, fs)
    e_b = _frame_energy(b, fs)
    m = min(len(e_a), len(e_b))
    e_a, e_b = e_a[:m], e_b[:m]

    order = np.argsort(e_a)
    n_q = max(1, int(m * quiet_ratio))
    n_l = max(1, int(m * loud_ratio))
    quiet, loud = order[:n_q], order[-n_l:]

    eps = 1e-12
    quiet_drop = 10 * np.log10(max(e_a[quiet].mean(), eps) / max(e_b[quiet].mean(), eps))
    loud_keep = 10 * np.log10(max(e_b[loud].mean(), eps) / max(e_a[loud].mean(), eps))

    # 近似 SNR 改善：语音段（高能量帧）与静音段（低能量帧）之比的变化
    snr_a = 10 * np.log10(max(e_a[loud].mean(), eps) / max(e_a[quiet].mean(), eps))
    snr_b = 10 * np.log10(max(e_b[loud].mean(), eps) / max(e_b[quiet].mean(), eps))

    # 自参考对数谱失真（改动幅度，越小越保守）
    sa = np.abs(dsp.stft(a)) ** 2
    sb = np.abs(dsp.stft(b)) ** 2
    k = min(sa.shape[1], sb.shape[1])
    sa, sb = np.maximum(sa[:, :k], eps), np.maximum(sb[:, :k], eps)
    lsd = float(np.mean(np.sqrt(np.mean((10 * np.log10(sa / sb)) ** 2, axis=0))))

    reliable = bool(m >= 10 and e_a[loud].mean() > 1e-8 and snr_a >= 6.0)
    return {
        "contrast_reliable": reliable,
        "contrast_note": "高低能量对比，仅为近似值" if reliable else "没有足够可区分的高低能量帧，无法可靠估计",
        "energy_contrast_before_db": float(snr_a),
        "energy_contrast_after_db": float(snr_b),
        "snr_est_before_db": float(snr_a) if reliable else None,
        "snr_est_after_db": float(snr_b) if reliable else None,
        "snr_est_improve_db": float(snr_b - snr_a) if reliable else None,
        "noise_floor_drop_db": float(quiet_drop),
        "speech_keep_db": float(loud_keep),
        "lsd_self_db": lsd,
        "length_ratio": float(length_ratio),      # 应 ≈ 1.0；明显偏离说明时长/采样率有问题
        "peak_before": float(np.max(np.abs(a))) if n else 0.0,
        "peak_after": float(np.max(np.abs(b))) if n else 0.0,
    }


def format_report(name: str, rep: dict) -> str:
    """使用准确的指标名称，不把输入输出差异当作真实语音失真。"""
    before, after = rep.get("snr_est_before_db"), rep.get("snr_est_after_db")
    contrast = (f"高低能量对比 {before:.1f} → {after:.1f} dB（近似）"
                if before is not None and after is not None else "高低能量对比：无法可靠估计")
    return (f"{name}：{contrast}｜低能量帧下降 {rep['noise_floor_drop_db']:.1f} dB"
            f"｜高能量帧保留 {rep['speech_keep_db']:+.1f} dB"
            f"｜频谱改变量 {rep['lsd_self_db']:.1f} dB（不代表真实人声失真）")
