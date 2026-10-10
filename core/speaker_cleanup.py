"""逐轨轻度增强与带迟滞、保持、释放的软噪声门；不进行说话人识别。"""
import numpy as np

DEFAULTS = {"denoise_enabled": True, "denoise_strength": .35,
            "gate_enabled": False, "gate_threshold_db": -45., "gate_reduction_db": 18.,
            "attack_ms": 8., "hold_ms": 120., "release_ms": 160.}


def validate_config(config):
    result = {**DEFAULTS, **config}
    for name in ("denoise_enabled", "gate_enabled"):
        if not isinstance(result[name], bool):
            raise ValueError("启用状态必须是布尔值")
    ranges = {"denoise_strength": (0, 1), "gate_threshold_db": (-80, -10),
              "gate_reduction_db": (0, 48), "attack_ms": (1, 50),
              "hold_ms": (20, 500), "release_ms": (20, 1000)}
    for name, (low, high) in ranges.items():
        value = float(result[name])
        if not np.isfinite(value) or not low <= value <= high:
            raise ValueError(f"{name} 须在 {low}～{high} 之间")
        result[name] = value
    return result


def soft_noise_gate(audio, fs, threshold_db=-45, reduction_db=18,
                    attack_ms=8, hold_ms=120, release_ms=160):
    """20 ms 前向 RMS，10 ms 增益帧；3 dB 迟滞，平滑开关且保留低增益底部。"""
    x = np.asarray(audio, np.float32)
    if x.ndim != 1 or not len(x) or not np.isfinite(x).all() or fs != int(fs) or fs < 8000:
        raise ValueError("需要非空有限单声道和至少 8000 Hz 的整数采样率")
    cfg = validate_config({"gate_threshold_db": threshold_db, "gate_reduction_db": reduction_db,
                           "attack_ms": attack_ms, "hold_ms": hold_ms, "release_ms": release_ms})
    if reduction_db == 0:
        return x.copy()
    hop = max(1, round(fs * .01))
    frames = (len(x) + hop - 1) // hop
    padded = np.pad(x, (0, (frames + 1) * hop - len(x)), mode="edge")
    power = np.mean(padded.reshape(-1, hop) ** 2, axis=1, dtype=np.float64)
    rms = np.sqrt((power[:-1] + power[1:]) / 2)
    open_at = 10 ** (cfg["gate_threshold_db"] / 20)
    close_at = open_at * 10 ** (-3 / 20)
    floor = 10 ** (-cfg["gate_reduction_db"] / 20)
    attack = np.exp(-hop / (fs * cfg["attack_ms"] / 1000))
    release = np.exp(-hop / (fs * cfg["release_ms"] / 1000))
    hold_frames = int(np.ceil(cfg["hold_ms"] * fs / (1000 * hop)))
    opened, remaining = bool(rms[0] >= open_at), 0
    gain = 1. if opened else floor
    gains = np.empty(frames)
    for i, level in enumerate(rms):
        if level >= open_at:
            opened, remaining = True, hold_frames
        elif opened and level >= close_at:
            remaining = hold_frames
        elif remaining:
            remaining -= 1
        else:
            opened = False
        target = 1. if opened else floor
        coefficient = attack if target > gain else release
        gain = coefficient * gain + (1 - coefficient) * target
        gains[i] = gain
    envelope = np.interp(np.arange(len(x)), np.arange(frames) * hop + hop / 2, gains)
    return (x * envelope).astype(np.float32)


def cleanup_track(audio, fs, config, denoiser=None):
    """始终从原始分离音轨计算；不做独立峰值归一化，避免放大残留声。"""
    cfg = validate_config(config)
    x = np.asarray(audio, np.float32)
    if x.ndim != 1 or not len(x) or not np.isfinite(x).all() or not np.isfinite(fs) or fs != int(fs) or fs < 8000:
        raise ValueError("音轨或采样率无效")
    out = x.copy()
    if cfg["denoise_enabled"] and cfg["denoise_strength"] > 0:
        if denoiser is None or not denoiser.is_ready:
            raise RuntimeError("GTCRN 不可用，请安装基础依赖并保留 models/gtcrn/，或关闭降噪只使用噪声门")
        enhanced = denoiser.enhance(x, fs=int(fs))
        if enhanced.shape != x.shape or not np.isfinite(enhanced).all():
            raise ValueError("降噪输出长度或数值异常")
        out = ((1 - cfg["denoise_strength"]) * x + cfg["denoise_strength"] * enhanced).astype(np.float32)
    if cfg["gate_enabled"]:
        out = soft_noise_gate(out, fs, cfg["gate_threshold_db"], cfg["gate_reduction_db"],
                              cfg["attack_ms"], cfg["hold_ms"], cfg["release_ms"])
    return out
