"""纯 NumPy 实现的降噪神经网络（多层感知机 + 手写反向传播 + Adam）。

为什么不用 PyTorch / TensorFlow？
    本机 Python 版本为 3.14，PyTorch 官方尚未发布 cp314 轮子（见
    https://github.com/pytorch/pytorch/issues/156856），因此这里用 NumPy
    从零实现前向、反向与优化器：依赖为零、可解释、可现场训练。

网络结构（谱域掩蔽，Speech Enhancement by Masking）：
    输入  : 一帧带噪对数幅度谱（257 维，即 0~8 kHz，512 点 FFT）
    隐藏层: tanh(128) -> tanh(64)
    输出  : sigmoid(257) —— 理想比值掩蔽 IRM ∈ [0,1]
    去噪  : S_hat = M ⊙ Y，再做 ISTFT 还原时域波形

即：网络学的是"每个频点该保留多少能量"，这就是最小的机器学习语音增强模型。
"""

from __future__ import annotations

import os

import numpy as np

MODEL_FILE = "denoise_mlp.npz"

# 与 core.dsp 保持一致的 STFT 参数
N_FFT = 512
HOP = 128
N_BINS = N_FFT // 2 + 1          # 257
EPS = 1e-8


# ----------------------------------------------------------------------------
# 参数初始化
# ----------------------------------------------------------------------------
def init_params(dims: list[int], seed: int = 0) -> dict:
    """Xavier/He 风格初始化：W ~ N(0, sqrt(2/fan_in))，b = 0。"""
    rng = np.random.default_rng(seed)
    params: dict[str, np.ndarray] = {}
    for i in range(len(dims) - 1):
        fan_in, fan_out = dims[i], dims[i + 1]
        params[f"W{i + 1}"] = (rng.standard_normal((fan_in, fan_out)) * np.sqrt(2.0 / fan_in)).astype(np.float32)
        params[f"b{i + 1}"] = np.zeros(fan_out, dtype=np.float32)
    params["n_layers"] = np.array(len(dims) - 1)
    params["dims"] = np.array(dims)
    return params


def param_summary(params: dict) -> str:
    dims = np.asarray(params["dims"]).ravel().tolist()
    n = sum(int(np.size(params[f"W{i + 1}"])) + int(np.size(params[f"b{i + 1}"]))
            for i in range(int(params["n_layers"])))
    return f"结构 {dims}，可训练参数 {n:,} 个"


# ----------------------------------------------------------------------------
# 前向 / 反向
# ----------------------------------------------------------------------------
def forward(params: dict, X: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    """前向传播，返回 (输出, 各层缓存)。最后一层用 sigmoid，其余用 tanh。"""
    L = int(params["n_layers"])
    A = np.asarray(X, dtype=np.float32)
    caches: list[np.ndarray] = [A]
    for i in range(1, L + 1):
        Z = A @ params[f"W{i}"] + params[f"b{i}"]
        A = 1.0 / (1.0 + np.exp(-Z)) if i == L else np.tanh(Z)
        caches.append(A)
    return A, caches


def backward(params: dict, caches: list[np.ndarray], Y: np.ndarray,
             sample_weight: np.ndarray | None = None) -> tuple[float, dict[str, np.ndarray]]:
    """反向传播。损失 = 加权二元交叉熵（掩蔽值回归到 [0,1] 的标准做法）。

    若 batch 内同时有带噪输入与干净幅度谱，可直接用 IRM 作监督，
    不需要额外传入权重；sample_weight 用于对语音帧加权（可选）。
    """
    L = int(params["n_layers"])
    A_out = caches[-1]
    Y = np.asarray(Y, dtype=np.float32)
    A_out = np.clip(A_out, EPS, 1.0 - EPS)

    if sample_weight is None:
        w = np.ones_like(Y, dtype=np.float32)
    else:
        w = np.asarray(sample_weight, dtype=np.float32)
    wsum = max(float(w.sum()), EPS)

    loss = -float((w * (Y * np.log(A_out) + (1.0 - Y) * np.log(1.0 - A_out))).sum() / wsum)

    grads: dict[str, np.ndarray] = {}
    # sigmoid + BCE 的导数合并后非常简洁：dZ = w * (A - Y) / sum(w)
    dZ = (w * (A_out - Y)) / wsum
    for i in range(L, 0, -1):
        A_prev = caches[i - 1]
        grads[f"W{i}"] = A_prev.T @ dZ
        grads[f"b{i}"] = dZ.sum(axis=0)
        if i > 1:
            dA = dZ @ params[f"W{i}"].T
            dZ = dA * (1.0 - A_prev ** 2)          # tanh'
    return loss, grads


def predict(params: dict, X: np.ndarray) -> np.ndarray:
    """推理：返回掩蔽值 (batch, n_bins)。"""
    out, _ = forward(params, X)
    return out


# ----------------------------------------------------------------------------
# Adam 优化器
# ----------------------------------------------------------------------------
class Adam:
    def __init__(self, params: dict, lr: float = 2e-3, beta1: float = 0.9,
                 beta2: float = 0.999, eps: float = 1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, beta1, beta2, eps
        self.t = 0
        self.m = {k: np.zeros_like(v, dtype=np.float32) for k, v in params.items() if k.startswith("W") or k.startswith("b")}
        self.v = {k: np.zeros_like(v, dtype=np.float32) for k, v in params.items() if k.startswith("W") or k.startswith("b")}

    def step(self, params: dict, grads: dict[str, np.ndarray]) -> None:
        self.t += 1
        bc1 = 1.0 - self.b1 ** self.t
        bc2 = 1.0 - self.b2 ** self.t
        for k, g in grads.items():
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * (g * g)
            mhat = self.m[k] / bc1
            vhat = self.v[k] / bc2
            params[k] -= (self.lr * mhat / (np.sqrt(vhat) + self.eps)).astype(np.float32)


# ----------------------------------------------------------------------------
# 特征构造与归一化
# ----------------------------------------------------------------------------
def log_mag_features(mag: np.ndarray) -> np.ndarray:
    """幅度谱 -> 对数幅度特征（20*log10，钳位到 -80 dB 以上）。"""
    mag = np.maximum(np.asarray(mag, dtype=np.float32), 1e-10)
    return (20.0 * np.log10(mag)).astype(np.float32)


def normalize_features(X: np.ndarray, mean: float, std: float) -> np.ndarray:
    return ((X - mean) / max(float(std), EPS)).astype(np.float32)


def ideal_ratio_mask(clean_mag: np.ndarray, noise_mag: np.ndarray) -> np.ndarray:
    """幅度比标签 |S|/(|S|+|N|)，使用实际噪声谱。"""
    s = np.asarray(clean_mag, dtype=np.float32)
    n = np.asarray(noise_mag, dtype=np.float32)
    return np.clip(s / (s + n + EPS), 0.0, 1.0).astype(np.float32)


# ----------------------------------------------------------------------------
# 存取
# ----------------------------------------------------------------------------
def save_model(path: str, params: dict, norm: dict, meta: dict | None = None) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    payload = {k: v for k, v in params.items()}
    payload["norm_mean"] = np.array([norm["mean"]], dtype=np.float32)
    payload["norm_std"] = np.array([norm["std"]], dtype=np.float32)
    if meta:
        for k, v in meta.items():
            payload[f"meta_{k}"] = np.array([v])
    import tempfile
    fd, temp = tempfile.mkstemp(suffix=".npz", dir=os.path.dirname(os.path.abspath(path)))
    os.close(fd)
    try:
        np.savez_compressed(temp, **payload)
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.remove(temp)


def load_model(path: str) -> tuple[dict, dict, dict]:
    """读取模型，返回 (params, norm, meta)。"""
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    z = np.load(path, allow_pickle=False)
    params: dict[str, np.ndarray] = {}
    meta: dict = {}
    for k in z.files:
        if k.startswith("meta_"):
            meta[k[5:]] = z[k].ravel()[0]
        elif k not in ("norm_mean", "norm_std"):
            params[k] = z[k]
    norm = {"mean": float(z["norm_mean"].ravel()[0]), "std": float(z["norm_std"].ravel()[0])}
    z.close()
    dims = np.asarray(params.get("dims", [])).reshape(-1)
    layers = int(params.get("n_layers", -1))
    if len(dims) < 2 or layers != len(dims) - 1 or not np.all(dims > 0):
        raise ValueError("模型结构元数据无效")
    for i in range(layers):
        w, b = params.get(f"W{i + 1}"), params.get(f"b{i + 1}")
        if w is None or b is None or w.shape != (dims[i], dims[i + 1]) or b.size != dims[i + 1]:
            raise ValueError("模型权重维度与结构不一致")
        if not np.isfinite(w).all() or not np.isfinite(b).all():
            raise ValueError("模型权重包含非有限值")
    if not np.isfinite(norm["mean"]) or not np.isfinite(norm["std"]) or norm["std"] <= 0:
        raise ValueError("模型归一化参数无效")
    return params, norm, meta


def default_model_path() -> str:
    import json
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    active = root / "models" / "active_model.json"
    if active.exists():
        try:
            candidate = Path(json.loads(active.read_text(encoding="utf-8"))["path"])
            if not candidate.is_absolute():
                candidate = root / candidate
            if candidate.is_file():
                return str(candidate)
        except (ValueError, KeyError, OSError):
            pass
    return str(root / "models" / MODEL_FILE)


def set_active_model_path(path: str) -> None:
    import json
    from pathlib import Path
    # 先加载验证，避免将损坏文件设为当前模型。
    load_model(path)
    root = Path(__file__).resolve().parents[1] / "models"
    root.mkdir(parents=True, exist_ok=True)
    temp = root / "active_model.json.tmp"
    candidate = Path(path).resolve()
    try:
        saved_path = str(candidate.relative_to(root.parent))
    except ValueError:
        saved_path = str(candidate)
    temp.write_text(json.dumps({"path": saved_path}, ensure_ascii=False), encoding="utf-8")
    os.replace(temp, root / "active_model.json")
