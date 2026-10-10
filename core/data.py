"""本地合成数据集：噪声库 + 随机 SNR 混合 + 理想比值掩蔽（IRM）标签。

为什么要自己造数据？
    语音增强的监督学习需要 (带噪, 干净) 成对数据。公开数据集
    （VoiceBank-DEMAND / MS-SNSD / MUSAN / DEMAND）动辄数 GB，下载和
    版权都不方便；而"干净语音 + 合成噪声 + 已知 SNR"是学术界最常用的
    配对方法（VoiceBank-DEMAND 本身就是这么造的）。所以这里：

    干净语音  <- 你自己的录音（data.wav / manba.wav / 麦克风新录）
    噪声      <- 本文件用 NumPy 程序化合成（白噪/粉噪/工频/键盘/嘈杂人声/车内/空调）
    混合      <- 随机 SNR(-5~20 dB) 逐样本混合，SNR 精确可控
    标签      <- 理想比值掩蔽 IRM = |S| / (|S| + |N|)，无需人工标注

    于是训练集可以无限扩充，且天然带"干净参考"，可以精确算降噪指标。
"""

from __future__ import annotations

import os

import numpy as np
from scipy import signal

from . import audio_io, dsp

# ----------------------------------------------------------------------------
# 噪声库
# ----------------------------------------------------------------------------
NOISE_TYPES = ("white", "pink", "brown", "hum", "click", "babble", "car", "ac")
FILE_NOISE_KIND = "real_file"          # 来自真实噪声数据集（DEMAND / MUSAN 等）
NOISE_LABELS = {
    "white": "白噪声",
    "pink": "粉红噪声(1/f)",
    "brown": "布朗噪声(1/f²)",
    "hum": "50Hz 工频嗡嗡声",
    "click": "键盘敲击脉冲",
    "babble": "嘈杂人声(babble)",
    "car": "车内噪声",
    "ac": "空调低鸣",
    FILE_NOISE_KIND: "真实噪声数据集",
}


def _rng(seed: int | None = None) -> np.random.Generator:
    return np.random.default_rng(seed)


def make_noise(kind: str, n: int, fs: int = 16000, seed: int | None = None,
               speech_pool: list[np.ndarray] | None = None) -> np.ndarray:
    """合成指定类型的噪声，长度 n 个采样点。"""
    rng = _rng(seed)
    n = int(n)
    if n <= 0:
        return np.zeros(0, dtype=np.float32)

    if kind == "white":
        x = rng.standard_normal(n)

    elif kind == "pink":  # 1/f：白噪声做 -3dB/oct 整形
        w = rng.standard_normal(n + 512)
        X = np.fft.rfft(w)
        f = np.arange(len(X))
        f[0] = 1
        X = X / np.sqrt(f)
        x = np.fft.irfft(X, n=len(w))[:n]

    elif kind == "brown":  # 1/f²：累积和
        x = np.cumsum(rng.standard_normal(n))
        x = signal.detrend(x)

    elif kind == "hum":  # 50 Hz 工频 + 谐波 + 缓慢调制（电源/电机嗡嗡声）
        t = np.arange(n) / fs
        f0 = rng.uniform(49.0, 51.0)
        x = np.zeros(n)
        for k, amp in enumerate((1.0, 0.5, 0.3, 0.15, 0.08), start=1):
            x += amp * np.sin(2 * np.pi * f0 * k * t + rng.uniform(0, 2 * np.pi))
        x *= 1.0 + 0.15 * np.sin(2 * np.pi * rng.uniform(0.5, 3.0) * t)
        x += 0.05 * rng.standard_normal(n)  # 叠加一点底噪，避免过于纯净

    elif kind == "click":  # 键盘敲击：稀疏指数衰减脉冲串
        x = 0.002 * rng.standard_normal(n)
        n_hits = max(1, int(n / fs * rng.uniform(4, 9)))
        for _ in range(n_hits):
            pos = rng.integers(0, n)
            ln = min(n - pos, int(0.02 * fs))
            env = np.exp(-np.arange(ln) / (0.002 * fs))
            x[pos:pos + ln] += rng.uniform(0.4, 1.0) * env * rng.standard_normal(ln)

    elif kind == "babble":  # 多个说话人叠加（用现有语音做素材，变调+错位）
        if not speech_pool:
            return make_noise("pink", n, fs, seed)
        x = np.zeros(n)
        n_spk = int(rng.integers(3, 7))
        for _ in range(n_spk):
            s = speech_pool[int(rng.integers(0, len(speech_pool)))]
            s = np.asarray(s, dtype=np.float64).reshape(-1)
            if s.size < 256:
                continue
            rate = rng.uniform(0.85, 1.2)                     # 模拟音高/语速差异
            idx = np.clip((np.arange(int(s.size / rate)) * rate).astype(int), 0, s.size - 1)
            v = signal.resample(s[idx], int(s.size / rate))
            start = int(rng.integers(0, max(1, n)))
            reps = int(np.ceil((n - start) / max(1, v.size))) + 1
            tile = np.tile(v, reps)[:n - start]
            x[start:start + tile.size] += tile
        x = signal.lfilter([1.0], [1.0, -0.6], x)             # 轻微低通，更像隔墙人声

    elif kind == "car":  # 车内：低频轰鸣 + 宽带路噪
        w = make_noise("pink", n, fs, seed)
        b, a = signal.butter(2, [60 / (fs / 2), 1200 / (fs / 2)], btype="band")
        x = signal.lfilter(b, a, w)
        t = np.arange(n) / fs
        x += 0.6 * np.sin(2 * np.pi * rng.uniform(25, 45) * t)  # 发动机低频
        x *= 1.0 + 0.2 * np.sin(2 * np.pi * 0.7 * t)

    elif kind == "ac":  # 空调：低通噪声 + 稳定窄带成分
        w = make_noise("brown", n, fs, seed)
        b, a = signal.butter(3, 900 / (fs / 2), btype="low")
        x = signal.lfilter(b, a, w)
        t = np.arange(n) / fs
        x += 0.3 * np.sin(2 * np.pi * rng.uniform(150, 260) * t)

    else:
        raise ValueError(f"未知噪声类型：{kind}（可选 {NOISE_TYPES}）")

    x = np.asarray(x, dtype=np.float64).reshape(-1)[:n]
    x = x - x.mean()
    return (x / (np.max(np.abs(x)) + 1e-12)).astype(np.float32)


def load_noise_pool(noise_dir: str, fs: int = 16000, max_files: int = 400,
                    max_sec_per_file: float = 10.0) -> list[np.ndarray]:
    """读取真实噪声文件（DEMAND / MUSAN / MS-SNSD / ESC-50 等），统一为单声道 float32。

    每个文件最多取 max_sec_per_file 秒，避免个别超长文件主导训练分布。
    找不到目录或没有可用文件时返回空列表（训练脚本会自动退回程序化合成噪声）。
    """
    import glob as _glob

    if not noise_dir or not os.path.isdir(noise_dir):
        return []
    exts = ("*.wav", "*.flac", "*.mp3", "*.ogg", "*.m4a")
    paths: list[str] = []
    for e in exts:
        paths.extend(_glob.glob(os.path.join(noise_dir, "**", e), recursive=True))
    paths = sorted(paths)[:max_files]
    pool: list[np.ndarray] = []
    for p in paths:
        try:
            x, sr = audio_io.load_audio(p)
        except Exception:
            continue
        x = audio_io.to_mono(x)
        if x.size < fs // 4:                      # 太短的片段丢弃
            continue
        if sr != fs:
            from scipy import signal as _sps
            x = _sps.resample(x, int(round(x.size * fs / sr))).astype(np.float32)
        x = x[:int(max_sec_per_file * fs)]
        peak = float(np.max(np.abs(x)))
        if peak > 1e-6:
            x = (x / peak * 0.9).astype(np.float32)
            pool.append(x)
    return pool


def file_noise(pool: list[np.ndarray], n: int, rng: np.random.Generator) -> np.ndarray:
    """从真实噪声池里随机取一段（循环拼接）作为长度为 n 的噪声。"""
    if not pool:
        return np.zeros(n, dtype=np.float32)
    x = pool[int(rng.integers(0, len(pool)))]
    if x.size >= n:
        start = int(rng.integers(0, x.size - n + 1))
        return x[start:start + n].astype(np.float32)
    reps = int(np.ceil(n / x.size))
    y = np.tile(x, reps)[:n]
    if y.size < n:
        y = np.pad(y, (0, n - y.size))
    return y.astype(np.float32)


# ----------------------------------------------------------------------------
# 干净语音池
# ----------------------------------------------------------------------------
def load_clean_pool(paths: list[str], fs: int = 16000, min_sec: float = 1.0) -> list[np.ndarray]:
    """读取干净语音，统一为单声道 float32 / 指定采样率，切成 1~4 秒片段。"""
    pool: list[np.ndarray] = []
    for p in paths:
        try:
            y, sr = audio_io.load_audio(p)
        except Exception as exc:  # pragma: no cover
            print(f"[warn] 跳过 {p}：{exc}")
            continue
        y = audio_io.to_mono(y)
        if y.size == 0:
            continue
        if sr != fs:
            n_new = int(round(y.size * fs / sr))
            y = signal.resample(y, n_new).astype(np.float32)
        peak = float(np.max(np.abs(y)))
        if peak > 0:
            y = (y / peak * 0.95).astype(np.float32)
        seg = int(min_sec * fs)
        for s in range(0, max(1, y.size - seg + 1), 4 * fs):
            piece = y[s:s + 4 * fs]
            if piece.size >= seg:
                pool.append(piece.astype(np.float32))
    if not pool:
        raise RuntimeError("没有可用的干净语音，请先录音或指定 --clean 音频文件")
    return pool


# ----------------------------------------------------------------------------
# 混合
# ----------------------------------------------------------------------------
def mix_at_snr(clean: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """把噪声按指定 SNR 叠加到干净语音上。"""
    clean = np.asarray(clean, dtype=np.float64).reshape(-1)
    noise = np.asarray(noise, dtype=np.float64).reshape(-1)
    n = len(clean)
    if len(noise) < n:
        reps = int(np.ceil(n / max(1, len(noise))))
        noise = np.tile(noise, reps)
    noise = noise[:n]
    pc = float(np.sum(clean ** 2)) / max(n, 1)
    pn = float(np.sum(noise ** 2)) / max(n, 1)
    if pn <= 1e-20 or pc <= 1e-20:
        return clean.astype(np.float32)
    scale = np.sqrt(pc / (pn * (10.0 ** (snr_db / 10.0))))
    return (clean + scale * noise).astype(np.float32)


def random_mixture(clean: np.ndarray, fs: int = 16000, snr_range: tuple[float, float] = (-5.0, 20.0),
                   kinds: tuple[str, ...] = NOISE_TYPES, seed: int | None = None,
                   speech_pool: list[np.ndarray] | None = None,
                   noise_pool: list[np.ndarray] | None = None,
                   file_noise_ratio: float = 0.5) -> dict:
    """随机生成一条 (带噪, 干净) 样本。

    noise_pool 非空时，会以 file_noise_ratio 的概率使用**真实噪声数据集**
    （DEMAND / MUSAN 等），其余概率使用程序化合成噪声。
    """
    rng = _rng(seed)
    use_file = bool(noise_pool) and rng.random() < file_noise_ratio
    if use_file:
        kind = "real_file"
        noise = file_noise(noise_pool, len(clean), rng)
    else:
        kind = kinds[int(rng.integers(0, len(kinds)))]
        noise = make_noise(kind, len(clean), fs, seed=int(rng.integers(0, 2 ** 31 - 1)),
                           speech_pool=speech_pool)
    snr = float(rng.uniform(*snr_range))
    noisy = mix_at_snr(clean, noise, snr)
    return {"clean": np.asarray(clean, dtype=np.float32), "noisy": noisy,
            "noise": (noisy - np.asarray(clean, dtype=np.float32)), "snr_db": snr, "kind": kind}


# ----------------------------------------------------------------------------
# 训练集（懒生成：每个 epoch 现造，等价于无限数据增强）
# ----------------------------------------------------------------------------
def stack_context(mag: np.ndarray, context: int = 2) -> np.ndarray:
    """把每帧的幅度谱与其前后 context 帧拼接，得到 (n_bins*(2*context+1), n_frames)。

    为什么需要：单帧幅度谱里"语音+噪声"与"较强噪声"长得几乎一样，
    网络无法判断该保留多少；加入时间邻域后，它就能利用语音的连续性与
    噪声的平稳性做出判断——这正是掩蔽类语音增强效果好坏的关键。
    """
    mag = np.asarray(mag, dtype=np.float32)
    n_bins, n_frames = mag.shape
    if context <= 0:
        return mag
    padded = np.pad(mag, ((0, 0), (context, context)), mode="edge")
    feats = np.empty((n_bins * (2 * context + 1), n_frames), dtype=np.float32)
    for k in range(2 * context + 1):
        feats[k * n_bins:(k + 1) * n_bins, :] = padded[:, k:k + n_frames]
    return feats


def build_features(mag: np.ndarray, context: int = 2, use_noise_floor: bool = True,
                   use_snr: bool = True, noise_psd: np.ndarray | None = None,
                   noise_db_prefix: np.ndarray | None = None) -> np.ndarray:
    """构造网络输入特征，返回 (n_feat_bins*(2*context+1), n_frames)。

    三类特征（均取对数刻度，dB 域）：
        1) 带噪对数幅度谱      20log10|Y|            —— 观测
        2) 噪声底估计          10log10(N_est)        —— 关键！只看 |Y| 无法判断局部信噪比，
                                                        补上噪声底后网络才知道"这点是噪声还是语音"
        3) 局部信噪比          20log10|Y| − 10log10(N_est)（钳位到 [−20, 40] dB）

    实测：只用特征 1 时网络输出几乎是常数（逐频点相关仅 0.54，掩蔽标准差 0.18）；
    加入特征 2、3 后网络才真正学到"逐频点取舍"。

    noise_db_prefix：直接用外部给定的噪声底（dB，形状 (n_bins, n_frames)），
    例如预训练 GTCRN 的"输入 − 输出"残差。GTCRN 在真实噪声上训练过，它的残差
    比自研的平滑最小值统计更准，可明显提升对未见噪声类型的泛化。
    """
    mag = np.asarray(mag, dtype=np.float32)
    streams = [mlp_log_mag(mag)]
    if use_noise_floor or use_snr:
        if noise_db_prefix is not None:
            nb = np.asarray(noise_db_prefix, dtype=np.float32)
            if nb.shape[1] != mag.shape[1] or nb.shape[0] != mag.shape[0]:
                raise ValueError("外部噪声特征与输入谱尺寸不一致")
            noise_db_full = nb
        else:
            pn = dsp.minstat_noise_psd(mag) if noise_psd is None else np.asarray(noise_psd)
            noise_db = 10.0 * np.log10(np.maximum(pn, 1e-12))
            noise_db_full = np.repeat(noise_db, mag.shape[1], axis=1)
        if use_noise_floor:
            streams.append(noise_db_full.astype(np.float32))
        if use_snr:
            streams.append(np.clip(streams[0] - noise_db_full, -20.0, 40.0).astype(np.float32))
    feat = np.concatenate(streams, axis=0)
    return stack_context(feat, context)


def mlp_log_mag(mag: np.ndarray) -> np.ndarray:
    """幅度谱 -> 对数幅度（dB），与 core.model.log_mag_features 一致。"""
    return (20.0 * np.log10(np.maximum(np.asarray(mag, dtype=np.float32), 1e-10))).astype(np.float32)


FEATURE_DIM_PER_FRAME = {"with_noise": 3, "mag_only": 1}


def feature_dim(n_bins: int, context: int, n_streams: int = 3) -> int:
    """特征维度 = 频点数 × 特征流数 × (2*context+1)。"""
    return int(n_bins * n_streams * (2 * context + 1))


class MixtureDataset:
    """按需生成训练批次的迭代器。

    每个 epoch 都重新随机（噪声类型、SNR、噪声实现），因此不存在过拟合固定训练集的
    问题；对教学/答辩也便于解释"数据量可以按需扩增"。

    context 控制输入的时间上下文帧数（默认前后各 2 帧，输入维度 257*5=1285）。
    """

    def __init__(self, pool: list[np.ndarray], fs: int = 16000,
                 snr_range: tuple[float, float] = (-5.0, 20.0),
                 kinds: tuple[str, ...] = NOISE_TYPES, seed: int = 0,
                 n_fft: int = dsp.N_FFT, hop: int = dsp.HOP, context: int = 2,
                 use_noise_floor: bool = True, use_snr: bool = True,
                 noise_pool: list[np.ndarray] | None = None, file_noise_ratio: float = 0.5,
                 noise_db_provider=None):
        self.pool = pool
        self.fs = fs
        self.snr_range = snr_range
        self.kinds = kinds
        self.rng = _rng(seed)
        self.n_fft = n_fft
        self.hop = hop
        self.context = context
        self.use_noise_floor = use_noise_floor
        self.use_snr = use_snr
        self.noise_pool = noise_pool or []
        self.file_noise_ratio = file_noise_ratio if self.noise_pool else 0.0
        # 可选的噪声底来源：给定 callable(noisy, clean) -> (n_bins, n_frames) dB 噪声底。
        # 典型用法是接预训练 GTCRN 的"输入−输出"残差（比最小值统计更准，泛化更好）。
        self.noise_db_provider = noise_db_provider

    @property
    def n_streams(self) -> int:
        return 1 + int(self.use_noise_floor) + int(self.use_snr)

    @property
    def input_dim(self) -> int:
        return feature_dim(self.n_fft // 2 + 1, self.context, self.n_streams)

    def sample_pair(self) -> tuple[np.ndarray, np.ndarray, dict]:
        """返回 (带噪幅度谱, IRM 掩蔽, 元信息)，形状均为 (n_bins, n_frames)。"""
        clean = self.pool[int(self.rng.integers(0, len(self.pool)))]
        mix = random_mixture(clean, self.fs, self.snr_range, self.kinds,
                             seed=int(self.rng.integers(0, 2 ** 31 - 1)), speech_pool=self.pool,
                             noise_pool=self.noise_pool, file_noise_ratio=self.file_noise_ratio)
        sc = dsp.stft(mix["clean"], n_fft=self.n_fft, hop=self.hop)
        sn = dsp.stft(mix["noisy"], n_fft=self.n_fft, hop=self.hop)
        mag_c, mag_n = np.abs(sc), np.abs(sn)
        noise_spec = dsp.stft(mix["noise"], n_fft=self.n_fft, hop=self.hop)
        mask = _irm(mag_c, np.abs(noise_spec))
        return mag_n, mask, mix

    def batch(self, batch_size: int = 32, min_frames: int = 4) -> tuple[np.ndarray, np.ndarray, list[dict]]:
        """拼一个 mini-batch：把若干句的帧堆叠起来（帧级训练，句间互不影响）。

        返回的 X 已经过时间上下文拼接，形状 (n_frames, n_bins*(2*context+1))。
        """
        X, Y, meta = [], [], []
        n_sent = 0
        while n_sent < batch_size:
            mag_n, mask, mix = self.sample_pair()
            if mag_n.shape[1] < min_frames:
                continue
            feats = build_features(mag_n, self.context, self.use_noise_floor, self.use_snr,
                                   noise_db_prefix=self._noise_db(mix))
            X.append(feats.T.astype(np.float32))
            Y.append(mask.T.astype(np.float32))
            meta.append(mix)
            n_sent += 1
        return np.concatenate(X, axis=0), np.concatenate(Y, axis=0), meta

    def _noise_db(self, mix: dict):
        """按需生成噪声底特征（未配置 provider 时返回 None，走内置最小值统计）。"""
        if self.noise_db_provider is None:
            return None
        return self.noise_db_provider(mix["noisy"], mix["clean"])

    def fixed_batch(self, batch_size: int = 8, seed: int = 20240) -> tuple[np.ndarray, np.ndarray]:
        """生成一批固定不变的验证数据（含上下文），保证各 epoch 的验证损失可比。

        真实噪声池通常只有几十条片段，若验证集随机抽样，各轮的噪声构成会大幅波动
        （本项目实测验证损失在 0.35~0.76 之间乱跳），因此这里**逐条确定性地交替使用
        合成噪声与真实噪声**，让验证集每轮完全一致。
        """
        backup = self.rng
        self.rng = _rng(seed)
        backup_ratio = self.file_noise_ratio
        backup_kinds = self.kinds
        try:
            X, Y = [], []
            for i in range(batch_size):
                self.kinds = (backup_kinds[i % len(backup_kinds)],)
                self.file_noise_ratio = 0.0 if (i % 2 == 0 or not self.noise_pool) else 1.0
                mag_n, mask, mix = self.sample_pair()
                feats = build_features(mag_n, self.context, self.use_noise_floor, self.use_snr,
                                       noise_db_prefix=self._noise_db(mix))
                X.append(feats.T.astype(np.float32))
                Y.append(mask.T.astype(np.float32))
            return np.concatenate(X, axis=0), np.concatenate(Y, axis=0)
        finally:
            self.rng = backup
            self.file_noise_ratio = backup_ratio
            self.kinds = backup_kinds


def _irm(clean_mag: np.ndarray, noise_mag: np.ndarray) -> np.ndarray:
    """幅度比例标签 |S|/(|S|+|N|)，N 必须来自实际缩放后的噪声。"""
    s = np.asarray(clean_mag, dtype=np.float32)
    n = np.asarray(noise_mag, dtype=np.float32)
    return np.clip(s / (s + n + 1e-8), 0.0, 1.0).astype(np.float32)


def synthesize_noise_bank(out_dir: str, fs: int = 16000, seconds: float = 4.0,
                          kinds: tuple[str, ...] = NOISE_TYPES,
                          speech_pool: list[np.ndarray] | None = None) -> list[str]:
    """把噪声库导出成 wav 文件，方便在 GUI 里试听/给音频加指定噪声。"""
    import os
    paths = []
    os.makedirs(out_dir, exist_ok=True)
    for i, kind in enumerate(kinds):
        x = make_noise(kind, int(seconds * fs), fs, seed=1000 + i, speech_pool=speech_pool)
        p = os.path.join(out_dir, f"noise_{kind}.wav")
        audio_io.save_audio(p, x * 0.5, fs)
        paths.append(p)
    return paths
