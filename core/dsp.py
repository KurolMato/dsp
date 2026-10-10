"""经典数字信号处理：分帧、短时能量、端点检测、带阻滤波、频谱分析、回声、倒放、谱减法降噪。

对应原 MATLAB 代码：
    pushbutton4_Callback  -> echo()
    pushbutton5_Callback  -> design_bandstop() / classic_denoise()
    pushbutton6_Callback  -> frame_signal() / short_time_energy() / endpoint_detect()
    pushbutton7_Callback  -> reverse()
    pushbutton8/9_Callback -> spectrum()
    pushbutton10/11_Callback -> 时域绘图数据
"""

from __future__ import annotations

import numpy as np
from scipy import signal

# ----------------------------------------------------------------------------
# 全局 STFT 参数（与机器学习降噪共用，保证两套降噪结果可直接对比）
# ----------------------------------------------------------------------------
N_FFT = 512          # 窗长 32 ms @16 kHz
HOP = 128            # 帧移 8 ms
WIN = "hann"


def stft(y: np.ndarray, n_fft: int = N_FFT, hop: int = HOP, window: str = WIN) -> np.ndarray:
    """短时傅里叶变换，返回复数谱 (n_bins, n_frames)。"""
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    win = signal.get_window(window, n_fft, fftbins=True)
    if not len(y) or not np.isfinite(y).all():
        raise ValueError("音频不能为空或包含 NaN/Inf")
    # 覆盖不足一帧的尾部，另补一帧避免末端落在 Hann 窗的零点附近。
    n_frames = 2 + (max(0, len(y) - n_fft) + hop - 1) // hop
    pad = max(0, (n_frames - 1) * hop + n_fft - len(y))
    y = np.pad(y, (0, pad))
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n_frames)[:, None]
    frames = y[idx] * win[None, :]
    return np.fft.rfft(frames, n=n_fft, axis=1).T


def istft(spec: np.ndarray, n_fft: int = N_FFT, hop: int = HOP, window: str = WIN,
          length: int | None = None) -> np.ndarray:
    """短时傅里叶逆变换（加权重叠相加），与 stft() 互为逆运算。"""
    spec = np.asarray(spec)
    win = signal.get_window(window, n_fft, fftbins=True)
    frames = np.fft.irfft(spec.T, n=n_fft, axis=1) * win[None, :]
    n_frames = frames.shape[0]
    out_len = (n_frames - 1) * hop + n_fft
    out = np.zeros(out_len, dtype=np.float64)
    wsum = np.zeros(out_len, dtype=np.float64)
    wsq = win ** 2
    for i in range(n_frames):
        s = i * hop
        out[s:s + n_fft] += frames[i]
        wsum[s:s + n_fft] += wsq
    # 若窗和过小（信号首尾不足一帧），该处输出直接置零，避免除小数放大误差
    wsum[wsum < 1e-6] = np.inf
    out = out / wsum
    out[~np.isfinite(out)] = 0.0
    if length is not None:
        out = out[:length] if len(out) >= length else np.pad(out, (0, length - len(out)))
    return out.astype(np.float32)


# ----------------------------------------------------------------------------
# 分帧 / 短时能量 / 端点检测（pushbutton6）
# ----------------------------------------------------------------------------
def frame_signal(y: np.ndarray, frame_size: int, frame_step: int) -> np.ndarray:
    """按帧长、帧移切分信号，返回 (n_frames, frame_size)。"""
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    n = len(y)
    if n < frame_size:
        return y[None, :]
    n_frames = (n - frame_size) // frame_step + 1
    idx = np.arange(frame_size)[None, :] + frame_step * np.arange(n_frames)[:, None]
    return y[idx]


def short_time_energy(y: np.ndarray, frame_size: int, frame_step: int) -> np.ndarray:
    """短时能量 sum(frame.^2)，与 MATLAB 实现完全一致。"""
    frames = frame_signal(y, frame_size, frame_step)
    return np.sum(frames ** 2, axis=1)


def endpoint_detect(y: np.ndarray, fs: int, frame_length: float = 0.02, frame_shift: float = 0.01,
                    energy_threshold: float = 0.02) -> dict:
    """基于短时能量的语音端点检测。

    返回 dict：start_sample / end_sample / start_time / end_time / speech / 短时能量曲线等。
    """
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    frame_size = int(round(frame_length * fs))
    frame_step = int(round(frame_shift * fs))
    energy = short_time_energy(y, frame_size, frame_step)
    speech_frames = np.flatnonzero(energy > energy_threshold)
    res = {
        "frame_size": frame_size,
        "frame_step": frame_step,
        "energy": energy,
        "energy_threshold": energy_threshold,
        "n_frames": len(energy),
        "detected": speech_frames.size > 0,
    }
    if speech_frames.size == 0:
        res.update(start_sample=0, end_sample=0, start_time=0.0, end_time=0.0,
                   speech=np.zeros(0, dtype=np.float32))
        return res
    start_frame, end_frame = int(speech_frames[0]), int(speech_frames[-1])
    start_sample = start_frame * frame_step + 1
    end_sample = min(end_frame * frame_step + frame_size, len(y))
    res.update(
        start_frame=start_frame,
        end_frame=end_frame,
        start_sample=start_sample,
        end_sample=end_sample,
        start_time=start_frame * frame_step / fs,
        end_time=end_sample / fs,
        speech=y[start_sample - 1:end_sample].astype(np.float32),
    )
    return res


# ----------------------------------------------------------------------------
# 带阻滤波器（pushbutton5 前半段：1500-2100 Hz 阻带，滤 50 Hz 工频谐波/窄带干扰）
# ----------------------------------------------------------------------------
def design_bandstop(fs: int, fp1: float = 1500.0, fp2: float = 2100.0,
                    fs1: float = 1750.0, fs2: float = 1850.0,
                    rp: float = 1.0, rs: float = 30.0) -> dict:
    """复刻 MATLAB：buttord + butter(..., 'stop')，返回 b, a, 阶数, 频率响应曲线。"""
    nyq = fs / 2.0
    if not np.isfinite([fs, fp1, fp2, fs1, fs2, rp, rs]).all() or not 0 < fp1 < fs1 < fs2 < fp2 < nyq:
        raise ValueError("带阻频率须满足 0 < fp1 < fs1 < fs2 < fp2 < Nyquist")
    wp = [fp1 / nyq, fp2 / nyq]
    ws = [fs1 / nyq, fs2 / nyq]
    n, wc = signal.buttord(wp, ws, rp, rs)
    b, a = signal.butter(int(n), wc, btype="bandstop")
    sos = signal.butter(int(n), wc, btype="bandstop", output="sos")
    w, h = signal.sosfreqz(sos, worN=2048)
    freqs = w / np.pi * nyq
    mag_db = 20 * np.log10(np.maximum(np.abs(h), 1e-12) / np.max(np.abs(h)))
    return {"b": b, "a": a, "sos": sos, "order": int(n), "w_rad": w, "freqs": freqs, "mag_db": mag_db}


# ----------------------------------------------------------------------------
# 频谱分析（pushbutton8 / 9）
# ----------------------------------------------------------------------------
def spectrum(y: np.ndarray, fs: int, n_fft: int | None = None, one_sided: bool = True,
             half: bool = True, max_points: int = 20000) -> dict:
    """计算频谱（对应 fft(y, N) + k*D 谱线）。

    返回 freqs / mag / df(频率分辨率)。
    """
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    n_fft = int(n_fft or len(y))
    n_fft = max(1, n_fft)
    spec = np.fft.fft(y, n=n_fft)
    if one_sided:
        half_n = n_fft // 2 + 1
        spec = spec[:half_n]
    mag = np.abs(spec)
    freqs = np.arange(len(mag)) * (fs / n_fft)
    if half:  # 与原 MATLAB 一样只画到 fs/4（16 kHz 时即 0-8 kHz）
        keep = freqs <= fs / 4
        freqs, mag = freqs[keep], mag[keep]
    if len(mag) > max_points:  # 长信号抽稀，保证绘图流畅
        step = int(np.ceil(len(mag) / max_points))
        freqs, mag = freqs[::step], mag[::step]
    return {"freqs": freqs, "mag": mag, "df": fs / n_fft, "n_fft": n_fft}


def spectrogram(y: np.ndarray, fs: int, n_fft: int = N_FFT, hop: int = HOP) -> dict:
    """对数幅度谱图数据（给 matplotlib imshow 用）。"""
    spec = stft(y, n_fft=n_fft, hop=hop)
    mag = np.abs(spec)
    db = 20 * np.log10(np.maximum(mag, 1e-10))
    db -= db.max()
    times = np.arange(spec.shape[1]) * hop / fs
    freqs = np.arange(spec.shape[0]) * fs / n_fft
    return {"db": db, "times": times, "freqs": freqs}


# ----------------------------------------------------------------------------
# 回声（pushbutton4）与倒放（pushbutton7）
# ----------------------------------------------------------------------------
def echo(y: np.ndarray, fs: int, delay: int = 10000, a: float = 0.3) -> np.ndarray:
    """复刻 MATLAB：h=[1, zeros(1,R-1), a]; yr=conv(y(:,1),h)。

    delay 单位为采样点（默认 10000 点 ≈ 0.625 s @16 kHz）。
    """
    y = np.asarray(y, dtype=np.float32)
    if y.ndim == 2:
        y = y[:, 0]
    h = np.zeros(delay + 1, dtype=np.float32)
    h[0] = 1.0
    h[-1] = a
    return np.convolve(y.astype(np.float64), h).astype(np.float32)


def reverse(y: np.ndarray) -> np.ndarray:
    """倒放（对应 flip(y,1)）。"""
    y = np.asarray(y, dtype=np.float32)
    return np.flipud(y).copy()


# ----------------------------------------------------------------------------
# 经典降噪：谱减法 + 维纳平滑（作为机器学习降噪的对比基线）
# ----------------------------------------------------------------------------
def estimate_noise_psd(mag: np.ndarray, percentile: float = 5.0) -> np.ndarray:
    """噪声功率谱估计（最小值统计思想）。

    语音是"间歇"的：任何一个频点在整段信号里的大部分时间只有噪声，
    因此取该频点幅度谱的低分位数即可逼近噪声幅度，再平方得到噪声功率谱。
    这比只看开头几十毫秒稳得多（原 MATLAB 版没有噪声估计，只做带阻滤波，
    这里是新增的对比基线）。
    """
    mag = np.asarray(mag, dtype=np.float64)
    noise_mag = np.percentile(mag, percentile, axis=1, keepdims=True)
    return np.maximum(noise_mag, 1e-8) ** 2


def minstat_noise_psd(mag: np.ndarray, freq_win: int = 9, time_win: int = 11,
                      bias: float = 3.0) -> np.ndarray:
    """平滑最小值统计噪声功率谱估计（经典 min-stat 的实用版本）。

    为什么必须"先平滑再取最小"：单个周期图 bin 的功率服从指数分布，
    直接取最小值会严重低估噪声——实测估计/真实仅 0.002（偏低 539 倍！），
    过减量因此形同虚设。先做 (freq_win × time_win) 的盒式平滑提高自由度，
    最小值偏差降到约 2 倍，再乘 bias 补偿即为可用估计。

    返回 (n_bins, 1) 的噪声功率谱（整段共用），供谱减法使用。
    """
    from scipy import ndimage

    mag = np.asarray(mag, dtype=np.float64)
    p = mag ** 2
    if p.shape[1] < 2:
        return np.percentile(p, 15.0, axis=1, keepdims=True)
    ps = ndimage.uniform_filter(p, size=(max(1, freq_win), max(1, time_win)), mode="nearest")
    return np.maximum(ps.min(axis=1, keepdims=True), 1e-12) * bias


def classic_denoise(y: np.ndarray, fs: int = 16000, alpha: float = 2.0, beta: float = 0.02,
                    noise_psd: np.ndarray | None = None, n_fft: int = N_FFT,
                    hop: int = HOP, method: str = "minstat", freq_win: int = 9,
                    time_win: int = 11, bias: float = 3.0, bandstop: bool = True,
                    notch: tuple[float, float, float, float] | None = None) -> np.ndarray:
    """经典降噪（对比基线）：带阻滤波 + 平滑最小值统计噪声估计 + 过减谱减法 + 维纳平滑。

        1) 带阻滤波：滤掉 1500~2100 Hz 窄带干扰（复刻原 MATLAB 的做法；
           实测对干净语音能量损失仅 0.028 dB）
        2) 噪声估计：平滑最小值统计（method="minstat"，默认）或全局低分位
        3) 谱减法：  |S|² = max(|Y|² − α·|N|², β·|Y|²)
        4) 维纳平滑：G = |S|²/(|S|² + |N|²)，抑制谱减法残留的"音乐噪声"

    bandstop=False 只执行谱减；notch 指定原采样率上的自定义带阻频率。
    GUI 的自动/手动决策入口为 classic_denoise_with_notch，避免重复固定带阻。
    效果需要结合干净参考、语音保留和试听评估，不使用掩蔽方案作为严格上限。
    """
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    if not len(y) or not np.isfinite(y).all() or not np.isfinite(fs) or fs != int(fs) or fs < 8000:
        raise ValueError("需要非空有限音频和至少 8000 Hz 的整数采样率")
    # 自定义带阻在原采样率执行，避免 16 kHz 降采样丢失高频阻带或偷偷改变参数。
    if bandstop and notch is not None:
        info = design_bandstop(int(fs), *notch)
        y = signal.sosfiltfilt(info["sos"], y.astype(np.float64)).astype(np.float32)
    n_orig = len(y)
    # 采样率适配：本算法按 16 kHz 设计；更高采样率的输入先降采样处理，结束后再升回原率，
    # 保证"输入输出时长与采样率一致"（曾因缺少这一步导致 48 kHz 音频被当 16 kHz 处理）
    fs_orig = int(fs)
    if fs_orig > 16000:
        from math import gcd

        from scipy import signal as _sps
        g = gcd(fs_orig, 16000)
        y = _sps.resample_poly(y.astype(np.float64), 16000 // g, fs_orig // g).astype(np.float32)
        fs = 16000
    # 1) 带阻滤波（对 16 kHz 信号）
    if bandstop and notch is None and fs >= 8000:
        try:
            info = design_bandstop(fs, min(1500.0, fs / 4.0), min(2100.0, fs / 4.0),
                                   min(1750.0, fs / 4.0), min(1850.0, fs / 4.0))
            y = signal.sosfiltfilt(info["sos"], y.astype(np.float64)).astype(np.float32)
        except Exception:
            pass
    # 2) 噪声功率谱
    spec = stft(y, n_fft=n_fft, hop=hop)
    mag, phase = np.abs(spec), np.angle(spec)
    if noise_psd is not None:
        pn = np.asarray(noise_psd, dtype=np.float64)
    elif method == "minstat":
        pn = minstat_noise_psd(mag, freq_win=freq_win, time_win=time_win, bias=bias)
    else:
        pn = estimate_noise_psd(mag)
    # 3) 谱减 + 4) 维纳平滑
    py = mag ** 2
    ps = np.maximum(py - alpha * pn, beta * py)
    gain = np.clip(ps / (ps + pn + 1e-12), 0.0, 1.0)
    out = istft(mag * gain * np.exp(1j * phase), n_fft=n_fft, hop=hop, length=len(y))
    if fs_orig > 16000:                      # 升回原采样率，保持时长与采样率一致
        from math import gcd

        from scipy import signal as _sps
        g = gcd(fs_orig, 16000)
        out = _sps.resample_poly(out.astype(np.float64), fs_orig // g, 16000 // g).astype(np.float32)
        if len(out) > n_orig:
            out = out[:n_orig]
        elif len(out) < n_orig:
            out = np.pad(out, (0, n_orig - len(out)))
    return out


def mixing_snr(noisy: np.ndarray, clean: np.ndarray) -> float:
    """按功率计算 SNR（dB）。"""
    noisy = np.asarray(noisy, dtype=np.float64).reshape(-1)
    clean = np.asarray(clean, dtype=np.float64).reshape(-1)
    n = min(len(noisy), len(clean))
    noise = noisy[:n] - clean[:n]
    pc, pn = float(np.sum(clean[:n] ** 2)), float(np.sum(noise ** 2))
    if pn <= 1e-20:
        return float("inf")
    return 10.0 * np.log10(pc / pn)


# 来版新增：窄带干扰检测与自适应陷波（合并后另行修正）。
DETECT_NFFT = 16384

def detect_spectrum(y: np.ndarray, fs: int = 16000, n_detect: int = 16384) -> np.ndarray:
    """为窄带检测计算的细分辨率谱（单帧，幅度）。

    为什么单独算：干扰检测要报出准确频率，但常规 STFT 的 512 点窗只有 31.25 Hz
    分辨率——实测把 50 Hz 工频报成 62 Hz（最近的 bin）。这里用长窗单帧谱
    （16384 点 ≈ 0.98 Hz 分辨率），并补零到相同长度，代价很低。
    单帧对稳态干扰足够（持续存在的分量在这一段里平均得很稳）。
    """
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    n = int(min(len(y), max(2048, n_detect)))
    seg = y[:n]
    if len(seg) < 256:
        return np.zeros(1)
    if len(seg) < n_detect:
        seg = np.pad(seg, (0, n_detect - len(seg)))
    win = np.hanning(len(seg))
    return np.abs(np.fft.rfft(seg * win, n=n_detect))


def detect_narrowband(mag: np.ndarray, fs: int, n_fft: int = N_FFT,
                      prominence_db: float = 12.0, harmonic_db: float = 9.0,
                      persist_db: float = 12.0, persist_on: bool = True,
                      harmonic_tol_db: float = 2.0, min_harmonics: int = 1,
                      stability_db: float = 3.0, gap_frac: float = 0.35,
                      band: tuple[float, float] = (45.0, 3600.0)) -> dict:
    """检测窄带干扰，返回 {enabled, center, halfwidth, harmonics, reason, ...}。

    从来版合并的谱域突出度、持续性和谐波候选检测。
    本函数仅判断传入的谱；实际波形定位与旁瓣排除由 analyze_narrowband 完成。

        尺度 1（定位，细分辨率单帧谱 16384 点 ≈ 0.98 Hz）：
            找出"高出本底 ≥ prominence_db"的峰，报出准确频率。
            常规 512 点 STFT 只有 31.25 Hz 分辨率，会把 50 Hz 工频报成 62 Hz。

        尺度 2（验持续性，粗分辨率多帧 STFT 512/128）：
            窄带干扰的特征是**跨整段信号持续存在**；语音的谱峰只在说话时出现。
            对每个候选频率，比较它在"最安静 35% 帧"与"最响 35% 帧"的平均功率：
                干扰 -> 比值接近 1（安静时也在）
                语音谱峰 -> 安静时几乎消失，比值可达 20 dB 以上

        尺度 3（谐波列）：基频峰必须 ≥ prominence_db，其 2×/3×/4× 谐波只要
            ≥ harmonic_db 即算命中（工频的形态就是"强基频 + 较弱谐波"）。

    注意：稳定性判据**只在多帧输入时可用**。若传入单帧细谱，persist 检查需要
    调用方通过 persist_on=False 关闭，此时仅靠突出度判断（会更容易误判）。
    """
    mag = np.asarray(mag, dtype=np.float64)
    if mag.ndim != 2 or mag.shape[1] < 4:
        return {"enabled": False, "applied": False, "reason": "输入谱形状不支持检测"}
    power = mag ** 2
    freqs = np.arange(mag.shape[0]) * fs / n_fft
    lo, hi = band
    in_band = (freqs >= lo) & (freqs <= hi)
    if not in_band.any():
        return {"enabled": False, "applied": False, "reason": "频段范围内无频点"}

    from scipy import ndimage
    # 本底必须在**频率轴**上估计，不能在时间轴上：
    # 持续干扰在整段都强，取时间中位数会把干扰自己当成本底（实测 1750 Hz 单频的
    # 突出度因此算成 −0.0 dB，直接漏检）。做法：先取"代表谱"（时间轴 70 分位，
    # 用分位而非最大值以避开突发冲击），再用较宽滑窗取局部 25 分位当作邻域本底。
    repr_spec = np.percentile(power, 70.0, axis=1)
    w = max(15, int(round(400.0 * mag.shape[0] / fs)))      # 约 ±200 Hz 邻域
    if w % 2 == 0:
        w += 1
    local_floor = ndimage.percentile_filter(repr_spec, 25.0, size=w, mode="nearest")
    excess_avg = 10.0 * np.log10(np.maximum(repr_spec, 1e-12)
                                 / np.maximum(local_floor, 1e-12))
    excess_avg[~in_band] = -np.inf
    cand = [i for i in range(1, len(excess_avg) - 1)
            if np.isfinite(excess_avg[i]) and excess_avg[i] >= prominence_db
            and excess_avg[i] >= excess_avg[i - 1] and excess_avg[i] >= excess_avg[i + 1]]
    if not cand:
        best = float(np.nanmax(np.where(np.isfinite(excess_avg), excess_avg, -np.inf)))
        return {"enabled": False, "applied": False, "prominence_db": best,
                "reason": f"未检出窄带干扰（最大突出度 {best:.1f} dB "
                          f"< 基频阈值 {prominence_db:.0f} dB）"}
    cand.sort(key=lambda i: -excess_avg[i])
    cand = cand[:8]

    # ---- 尺度 2：逐候选验证"干扰在语音停顿处依然存在" ----
    frame_e = power.sum(axis=0)
    ks = max(1, int(len(frame_e) * gap_frac))
    order = np.argsort(frame_e)
    quiet, loud = power[:, order[:ks]], power[:, order[-ks:]]

    def _persist(i: int) -> float:
        """候选频点及其邻域的 安静帧/响亮帧 功率比（dB）。越小越像持续干扰。"""
        a, b = max(0, i - 1), min(power.shape[0], i + 2)
        q = float(np.mean(quiet[a:b]))
        l = float(np.mean(loud[a:b]))
        return float(10.0 * np.log10(max(l, 1e-20) / max(q, 1e-20)))

    picked = []
    for i in cand:
        p = _persist(i) if persist_on else 0.0
        if p <= persist_db:
            picked.append((i, p))
            break                                     # 取最强且持续的候选（已按突出度排序）
    if not picked:
        i = cand[0]
        p = _persist(i) if persist_on else float("nan")
        return {"enabled": False, "applied": False, "center": float(freqs[i]),
                "prominence_db": float(excess_avg[i]),
                "persistence_db": p if np.isfinite(p) else None,
                "reason": f"{freqs[i]:.0f} Hz 处有突出峰（{excess_avg[i]:.1f} dB）但在语音"
                          f"停顿处消失（安静/响亮比 {p:.1f} dB > {persist_db:.0f} dB）"
                          f"→ 判为语音谱峰（非持续干扰）"}
    i, p = picked[0]                                  # 已按突出度排序
    f0 = float(freqs[i])

    # ---- 尺度 3：谐波列（每条谐波都要通过**持续性**检验）----
    frame_e = power.sum(axis=0)
    ks = max(1, int(len(frame_e) * gap_frac))
    order = np.argsort(frame_e)
    quiet, loud = power[:, order[:ks]], power[:, order[-ks:]]

    def _persist_at(i: int, half_bins: int = 3) -> float:
        """安静帧/响亮帧 功率比（dB）。越小越像"持续存在"的干扰。

        在 **±half_bins 的频带**上取平均而不是只看单个 bin：窄带干扰的裙边同样持续，
        用宽带平均能显著提高判据稳定性（实测真谐波 0.9~11.9 dB，噪声起伏 31~42 dB）。
        """
        a, b = max(0, i - half_bins), min(power.shape[0], i + half_bins + 1)
        return float(10.0 * np.log10(max(float(np.mean(loud[a:b])), 1e-20)
                                     / max(float(np.mean(quiet[a:b])), 1e-20)))

    def _ex_at(f: float) -> float:
        if f >= freqs[-1]:
            return -np.inf
        b = int(np.argmin(np.abs(freqs - f)))
        return float(np.max(excess_avg[max(0, b - 1): b + 2]))

    hits = []
    for m in (2, 3, 4):
        f = f0 * m
        if f > hi:
            break
        b = int(np.argmin(np.abs(freqs - f)))
        if _ex_at(f) >= harmonic_db and _persist_at(b) <= persist_db:
            hits.append(f)
    # 待陷波频点只看**基频的整数倍**：曾把候选峰周围的裙边也当成谐波
    # （纯 50 Hz 单频时误检出 98/200 Hz），结果白陷两刀、损伤语音。
    # 持续性判据（实测：真谐波 0.9~11.9 dB，噪声起伏 31~42 dB）把假峰彻底排除。
    kind = f"谐波列（命中 {len(hits)} 个谐波）" if hits else "稳定单频"
    reason = (f"检出 {kind}：基频 {f0:.0f} Hz，突出 {excess_avg[i]:.1f} dB，"
              f"安静/响亮比 {p:.1f} dB（越小越像持续干扰）"
              + (f"，谐波 {', '.join(f'{h:.0f}' for h in hits[:3])} Hz" if hits else ""))
    return {"enabled": True, "applied": False, "center": f0,
            "halfwidth": max(30.0, f0 * 0.03), "prominence_db": float(excess_avg[i]),
            "persistence_db": p, "harmonics": hits, "reason": reason}


def design_notch(fs: int, center: float, bandwidth: float = 40.0) -> dict:
    """二阶零极点陷波器（biquad）：零点在单位圆上对准干扰频率，极点在半径 r 处。

    为什么不用 design_bandstop（buttord + butter）：
        低频窄带（如 50 Hz 工频，归一化频率仅 0.006）上 Butterworth 带阻会设计出
        **不稳定**的滤波器——实测 50 Hz 时最大极点模长 1.348，filtfilt 直接发散；
        而稳定性校验又把这个频率静默丢弃，结果只有谐波被陷、基频漏掉。
    零极点法**结构上保证稳定**（r < 1 恒成立），且带宽控制直观：
        r = exp(-π·BW/fs)
    """
    f0 = min(max(float(center), 1.0), fs / 2.0 - 50.0)
    bw = max(float(bandwidth), 5.0)
    r = float(np.exp(-np.pi * bw / fs))
    r = min(max(r, 0.5), 0.99995)
    th = 2.0 * np.pi * f0 / fs
    a1 = -2.0 * r * np.cos(th)
    a2 = r * r
    b1 = -2.0 * np.cos(th)
    b2 = 1.0
    b = np.array([1.0, b1, b2])
    a = np.array([1.0, a1, a2])
    # 归一化：让远离陷波点的频段增益为 1（DC/奈奎斯特处增益补偿）
    g = float(abs((1.0 + b1 + b2) / (1.0 + a1 + a2)))
    b = b / max(g, 1e-9)
    return {"b": b, "a": a, "center": f0, "bandwidth": bw, "r": r,
            "poles": np.roots(a), "stable": bool(np.max(np.abs(np.roots(a))) < 1.0)}


def _validate_notch_audio(y, fs):
    y = np.asarray(y, dtype=np.float32)
    if y.ndim != 1 or not len(y) or not np.isfinite(y).all():
        raise ValueError("自动陷波需要非空有限的单声道音频")
    if not np.isfinite(fs) or fs != int(fs) or fs < 8000:
        raise ValueError("采样率须为至少 8000 Hz 的整数")
    return y, int(fs)


def analyze_narrowband(y, fs=16000, prominence_db=12.0, max_harmonics=20):
    """来版突出度/持续性判据，改用多帧细谱定位，避免粗谱假谐波和首段静音漏检。

    在 16 kHz 检测；每帧 4096 点，补零至 16384，最多采样 128 个完整帧。
    只返回决策，不修改音频。稳定音乐音符和持续元音仍可能被误判。
    """
    y, fs = _validate_notch_audio(y, fs)
    if not np.isfinite(prominence_db) or prominence_db <= 0 or max_harmonics != int(max_harmonics) or not 1 <= max_harmonics <= 20:
        raise ValueError("突出度须为正数，最多陷波频点须为 1～20 的整数")
    empty = {"enabled": False, "applied": False, "notched": [], "stopbands": [], "filters": []}
    if len(y) / fs < .35 or np.max(np.abs(y)) < 1e-8:
        return {**empty, "reason": "音频过短或静音，跳过自动陷波"}
    from math import gcd
    g = gcd(fs, 16000)
    work = signal.resample_poly(y.astype(np.float64), 16000 // g, fs // g) if fs != 16000 else y.astype(np.float64)
    win = min(4096, len(work))
    count = min(128, 1 + (len(work) - win) // 1024)
    starts = np.linspace(0, len(work) - win, count).astype(int)
    frames = work[starts[:, None] + np.arange(win)] * np.hanning(win)
    mag = np.abs(np.fft.rfft(frames, n=DETECT_NFFT, axis=1)).T
    info = detect_narrowband(mag, 16000, n_fft=DETECT_NFFT, prominence_db=prominence_db)
    if not info["enabled"]:
        return {**empty, **info}
    # 对所有候选重做细谱突出度、持续性和强度验证；不会盲目过滤整数倍频率。
    from scipy import ndimage
    power = mag ** 2
    representative = np.percentile(power, 70, axis=1)
    floor = ndimage.percentile_filter(representative, 25, size=205, mode="nearest")
    excess = 10 * np.log10(np.maximum(representative, 1e-12) / np.maximum(floor, 1e-12))
    order = np.argsort(power.sum(axis=0))
    k = max(1, int(len(order) * .35))
    quiet, loud = power[:, order[:k]], power[:, order[-k:]]
    quiet_power = quiet.mean(axis=1)
    freq = np.fft.rfftfreq(DETECT_NFFT, 1 / 16000)
    peaks, _ = signal.find_peaks(excess, height=prominence_db, distance=12)
    accepted = []
    for index in peaks:
        if not 45 <= freq[index] <= min(3600, fs / 2 - 50):
            continue
        a, b = max(0, index - 3), min(len(freq), index + 4)
        persistence = float(10 * np.log10(max(float(loud[a:b].mean()), 1e-20) / max(float(quiet[a:b].mean()), 1e-20)))
        # 拒绝极小旁瓣；否则纯 50 Hz 泄漏会被当作 100/200 Hz 谐波。
        if persistence > 12 or quiet_power[index] < quiet_power.max() * 1e-4:
            continue
        accepted.append((int(index), persistence))
    accepted.sort(key=lambda pair: -quiet_power[pair[0]])
    # 细谱能精确定位，但可能将低能量语音尾部的残留基频当作稳定线。
    # 沿用来版粗 STFT 的宽邻域持续性检查，确认至少一个主干扰；
    # 已确认主频的真正细谱谐波可保留，避免粗窗把语音混入谐波邻域造成漏检。
    coarse_count = min(512, 1 + (len(work) - 512) // 128)
    coarse_starts = np.linspace(0, len(work) - 512, coarse_count).astype(int)
    coarse_frames = work[coarse_starts[:, None] + np.arange(512)] * np.hanning(512)
    coarse_power = np.abs(np.fft.rfft(coarse_frames, axis=1)).T ** 2
    coarse_order = np.argsort(coarse_power.sum(axis=0))
    ck = max(1, int(len(coarse_order) * .35))
    cq, cl = coarse_power[:, coarse_order[:ck]], coarse_power[:, coarse_order[-ck:]]
    main = []
    for index, persistence in accepted:
        bin_index = round(freq[index] * 512 / 16000)
        a, b = max(0, bin_index - 1), min(len(cq), bin_index + 2)
        ratio = 10 * np.log10(max(float(cl[a:b].mean()), 1e-20) / max(float(cq[a:b].mean()), 1e-20))
        if ratio <= 12:
            main.append(index)
    accepted = [(i, p) for i, p in accepted if i in main or any(
        freq[i] > freq[base] * 1.5 and abs(freq[i] / freq[base] - round(freq[i] / freq[base])) < .05
        for base in main)]
    if not accepted:
        return {**empty, "reason": "候选峰未通过持续性或旁瓣排除检查，跳过带阻"}
    # 在功率谱实际峰值附近做抛物线插值，减少约 1 Hz 栅格量化偏差。
    centers = []
    for index, persistence in accepted:
        nearby = np.arange(max(1, index - 4), min(len(freq) - 1, index + 5))
        peak = int(nearby[np.argmax(quiet_power[nearby])])
        values = np.log(np.maximum(quiet_power[peak - 1:peak + 2], 1e-20))
        denominator = values[0] - 2 * values[1] + values[2]
        offset = np.clip(.5 * (values[0] - values[2]) / denominator, -.5, .5) if abs(denominator) > 1e-12 else 0
        centers.append(float((peak + offset) * 16000 / DETECT_NFFT))
    strongest = centers[0]
    separated = []
    for center in centers:
        # 以修准后的真实功率峰合并旁瓣；使用突出度峰位置会受起止边界偏移影响。
        if not any(abs(center - kept) <= max(12.5, kept * .03) for kept in separated):
            separated.append(center)
    centers = sorted({round(c, 2) for c in separated[:int(max_harmonics)]})
    return {**empty, "enabled": True, "center": strongest, "centers": centers,
            "halfwidth": max(12.5, strongest * .03),
            "harmonics": [c for c in centers if c > strongest * 1.5 and abs(c / strongest - round(c / strongest)) < .05],
            "prominence_db": float(excess[accepted[0][0]]), "persistence_db": accepted[0][1],
            "reason": "检出持续窄带干扰：" + "、".join(f"{c:.1f} Hz" for c in centers)}


def adaptive_notch(y, fs=16000, prominence_db=12.0, n_fft=N_FFT, hop=HOP, max_harmonics=20):
    """分析持续窄带干扰，再在原采样率执行稳定陷波；未检出时原样返回。"""
    y, fs = _validate_notch_audio(y, fs)
    info = analyze_narrowband(y, fs, prominence_db, max_harmonics)
    if not info["enabled"]:
        return y.copy(), info
    out = y.astype(np.float64)
    filters, centers, stops = [], [], []
    for center in info["centers"]:
        bandwidth = max(25, center * .06)
        half = bandwidth / 2
        lo, hi = max(5, center - half), min(fs / 2 - 50, center + half)
        fp1, fp2 = max(2, center - 2.6 * half), min(fs / 2 - 20, center + 2.6 * half)
        if center < 200:
            design = design_notch(fs, center, bandwidth)
            sos = signal.tf2sos(design["b"], design["a"])
            kind = "biquad"
        else:
            design = design_bandstop(fs, fp1, fp2, lo, hi)
            sos, kind = design["sos"], "butterworth_sos"
        filtered = signal.sosfiltfilt(sos, out)
        if not np.isfinite(filtered).all():
            raise ValueError("陷波输出异常，已拒绝应用")
        out = filtered
        centers.append(float(center))
        stops.append([float(lo), float(hi)])
        filters.append({"center": float(center), "bandwidth": float(bandwidth), "kind": kind, "sos": sos.tolist()})
    info.update(applied=bool(filters), notched=centers, stopband=stops[0], stopbands=stops,
                filters=filters, reason=info["reason"] + f"；已陷波 {len(filters)} 个频点")
    return out.astype(np.float32), info


def classic_denoise_with_notch(y, fs=16000, auto=True, notch=None):
    """统一界面/方案对比入口：自动或手动带阻 → 仅谱减，避免第二次固定带阻。"""
    y, fs = _validate_notch_audio(y, fs)
    if auto:
        filtered, info = adaptive_notch(y, fs)
        info["mode"] = "auto"
    else:
        values = tuple(notch or (1500, 2100, 1750, 1850))
        design = design_bandstop(fs, *values)
        filtered = signal.sosfiltfilt(design["sos"], y.astype(np.float64)).astype(np.float32)
        fp1, fp2, fs1, fs2 = map(float, values)
        info = {"mode": "manual", "enabled": True, "applied": True,
                "center": (fs1 + fs2) / 2, "notched": [(fs1 + fs2) / 2],
                "stopbands": [[fs1, fs2]], "frequencies": [fp1, fp2, fs1, fs2],
                "filters": [{"kind": "manual_butterworth_sos", "sos": design["sos"].tolist()}],
                "reason": f"手动带阻 {fs1:g}～{fs2:g} Hz"}
    out = classic_denoise(filtered, fs, bandstop=False)
    if out.shape != y.shape or not np.isfinite(out).all():
        raise ValueError("谱减输出长度或数值异常")
    return out, info


