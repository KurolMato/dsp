"""神经网络降噪推理：STFT -> 神经网络预测掩蔽 -> ISTFT。

对应原 MATLAB 的 pushbutton5（语音增强/降噪），把其中的"带阻滤波 + 谱减法"
换成数据驱动的机器学习方法：

    MATLAB 版： yx = filter(b, a, y)          —— 固定带阻滤波器，只压 1750~1850 Hz
               |S|² = max(|Y|² - α|N|², β|Y|²) —— 谱减法，需要人工设阈值
    Python 版： mask = MLP(log|Y|)            —— 从数据里学到的时频掩蔽，逐频点自适应

推理流程与训练完全分离：训练脚本负责学参数，本模块只做前向计算。
"""

from __future__ import annotations

import os

import numpy as np

from . import data as data_mod, dsp, model as mlp

# 掩蔽后处理：频率方向轻微平滑，抑制"音乐噪声"（谱减法常见副作用）
FREQ_SMOOTH_KERNEL = np.array([0.25, 0.5, 0.25], dtype=np.float32)

SR = 16000      # 本项目统一采样率


class Denoiser:
    """加载训练好的模型并执行降噪。模型文件不存在时 is_ready=False。"""

    def __init__(self, model_path: str | None = None, n_fft: int = dsp.N_FFT, hop: int = dsp.HOP):
        self.model_path = model_path or mlp.default_model_path()
        self.n_fft = n_fft
        self.hop = hop
        self.params: dict | None = None
        self.norm: dict | None = None
        self.meta: dict = {}
        self.load_error: str | None = None
        self.reload()

    # -- 状态 -----------------------------------------------------------------
    @property
    def is_ready(self) -> bool:
        return self.params is not None

    def reload(self) -> bool:
        try:
            self.params, self.norm, self.meta = mlp.load_model(self.model_path)
            self.load_error = None
            return True
        except Exception as exc:
            self.params, self.norm, self.meta = None, None, {}
            self.load_error = str(exc)
            return False

    def describe(self) -> str:
        if not self.is_ready:
            return f"模型未加载（{self.load_error}）"
        dims = np.asarray(self.params["dims"]).ravel().tolist()
        n_par = sum(int(np.size(self.params[f"W{i + 1}"])) + int(np.size(self.params[f"b{i + 1}"]))
                    for i in range(int(self.params["n_layers"])))
        ep = self.meta.get("epochs", "?")
        loss = self.meta.get("val_loss", float("nan"))
        ctx = int(self.meta.get("context", 0))
        feat = "带噪谱+噪声底+局部SNR" if self._use_noise_feats else "仅带噪谱"
        return (f"结构 {dims}（输入 {feat} × 时间上下文 ±{ctx} 帧），参数 {n_par:,}，"
                f"训练轮数 {ep}，验证损失 {float(loss):.4f}"
                + ("；旧版标签模型，建议重新训练" if self.meta.get("label_version") != "irm_actual_noise_v2" else ""))

    @property
    def _use_noise_feats(self) -> bool:
        """旧版模型只在元数据里记录 context，此时退回"仅带噪谱"特征。"""
        return bool(self.meta.get("use_noise_floor", False))

    @property
    def _noise_prefix(self) -> str:
        """噪声底估计来源：'ms' = 平滑最小值统计（自研），'gtcrn' = 预训练模型残差。"""
        return str(self.meta.get("noise_prefix", "ms"))

    def features(self, mag: np.ndarray, gtcrn_noise_db: np.ndarray | None = None) -> np.ndarray:
        """按模型训练时的特征配置提取输入特征（推理/训练必须完全一致）。

        gtcrn_noise_db：(n_bins, n_frames) 的噪声底（dB）。仅当模型以 GTCRN 残差
        作为噪声特征时才需要。残差是模型估计的噪声，包含增强误差，
        并非真实噪声；实际效果需要在独立素材上评估。
        """
        ctx = int(self.meta.get("context", 0))
        if self._noise_prefix == "gtcrn" and self._use_noise_feats and gtcrn_noise_db is None:
            raise ValueError("此模型必须使用 GTCRN 残差特征，不能退回其他噪声估计")
        return data_mod.build_features(mag, context=ctx,
                                       use_noise_floor=self._use_noise_feats,
                                       use_snr=bool(self.meta.get("use_snr", self._use_noise_feats)),
                                       noise_db_prefix=gtcrn_noise_db if self._noise_prefix == "gtcrn" else None)

    # -- 推理 -----------------------------------------------------------------
    def predict_mask(self, spec: np.ndarray, smooth: bool = True,
                     gtcrn_noise_db: np.ndarray | None = None) -> np.ndarray:
        """输入复数谱 (n_bins, n_frames)，输出掩蔽 (n_bins, n_frames)。"""
        if not self.is_ready:
            raise RuntimeError("模型未加载")
        feats = self.features(np.abs(spec), gtcrn_noise_db)   # (n_feat_bins, n_frames)
        X = feats.T                                           # (n_frames, n_feat_bins)
        Xn = mlp.normalize_features(X, self.norm["mean"], self.norm["std"])
        mask = mlp.predict(self.params, Xn).T                 # (n_bins, n_frames)
        if smooth and mask.shape[0] >= 3:
            mask = np.apply_along_axis(
                lambda m: np.convolve(m, FREQ_SMOOTH_KERNEL, mode="same"), 0, mask)
        return np.clip(mask, 0.0, 1.0).astype(np.float32)

    def enhance(self, y: np.ndarray, fs: int = SR, smooth: bool = True,
                gtcrn=None) -> np.ndarray:
        """按模型元数据重采样，返回原采样率、原长度的波形。"""
        y = np.asarray(y, dtype=np.float32).reshape(-1)
        if not len(y) or not np.isfinite(y).all() or int(fs) <= 0:
            raise ValueError("音频为空、包含非有限值或采样率无效")
        model_fs = int(self.meta.get("fs", SR))
        work = _resample_to(y, model_fs, int(fs))
        spec = dsp.stft(work, n_fft=self.n_fft, hop=self.hop)
        noise_db = None
        if self._noise_prefix == "gtcrn" and self._use_noise_feats:
            if gtcrn is None:
                from .gtcrn import GTCRNDenoiser
                gtcrn = GTCRNDenoiser()
            noise_db = self.gtcrn_noise_db(spec, gtcrn, work, model_fs)
        mask = self.predict_mask(spec, smooth=smooth, gtcrn_noise_db=noise_db)
        out = dsp.istft(spec * mask, n_fft=self.n_fft, hop=self.hop, length=len(work))
        return _resample_to(out, int(fs), model_fs, length=len(y))

    def gtcrn_noise_db(self, spec: np.ndarray, gtcrn, y: np.ndarray,
                       fs: int = SR) -> np.ndarray:
        """在同一采样率上计算输入−输出残差，避免多余的 48 kHz 往返。"""
        if not getattr(gtcrn, "is_ready", False):
            raise RuntimeError("此模型训练使用 GTCRN 特征，需要可用的 GTCRN 模型")
        resid = np.asarray(y, dtype=np.float32) - gtcrn.enhance(y, fs)
        res_spec = dsp.stft(resid, n_fft=self.n_fft, hop=self.hop)
        if res_spec.shape != spec.shape:
            raise ValueError("GTCRN 残差特征与输入帧数不一致")
        return (20.0 * np.log10(np.maximum(np.abs(res_spec), 1e-10))).astype(np.float32)

    def enhance_cascaded(self, y: np.ndarray, gtcrn, fs: int = SR,
                         smooth: bool = True) -> np.ndarray:
        """真正的串联：GTCRN 输出作为 MLP 输入；特征遵循模型训练配置。"""
        if not self.is_ready or not getattr(gtcrn, "is_ready", False):
            raise RuntimeError("串联降噪需要同时具备 GTCRN 与自训练模型")
        first = gtcrn.enhance(y, fs)
        return self.enhance(first, fs=fs, smooth=smooth, gtcrn=gtcrn)


def _resample_to(x: np.ndarray, fs_out: int, fs_in: int = SR,
                 length: int | None = None) -> np.ndarray:
    """有理数重采样（比 FFT 法更快、相位更准，且上/下行严格对称）。"""
    from math import gcd

    from scipy import signal as sps

    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if fs_out == fs_in:
        out = x
    else:
        g = gcd(int(fs_out), int(fs_in))
        out = sps.resample_poly(x.astype(np.float64), fs_out // g, fs_in // g).astype(np.float32)
    if length is not None:
        if len(out) > length:
            out = out[:length]
        elif len(out) < length:
            out = np.pad(out, (0, length - len(out)))
    return out


def enhance_with_report(y: np.ndarray, fs: int, model_path: str | None = None,
                        smooth: bool = True) -> tuple[np.ndarray, dict]:
    """便捷接口：返回 (降噪后波形, 说明信息)。"""
    d = Denoiser(model_path)
    if not d.is_ready:
        raise RuntimeError(f"神经网络模型不可用：{d.load_error}")
    out = d.enhance(y, fs=fs, smooth=smooth)
    return out, {"model": os.path.basename(d.model_path), "info": d.describe()}
