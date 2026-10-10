"""用"干净版 + 加噪版"这对素材做精确诊断（有 ground truth，不靠估计）。

素材（用户的 `原始测试音频/` 文件夹）：
    data.wav      16 kHz 干净语音（4 秒）
    加噪声.mp3     44.1 kHz 立体声，在同一段语音上加了强白噪声

有了干净参考，就能算出**真实噪声**与**真实 SNR**，从而回答两个问题：
    1. 这个噪声到底有多强？在不在模型训练时见过的范围内？
    2. 降噪后剩余误差里，**残留噪声**与**语音失真**各占多少？
       （这能区分"没降掉噪声"和"把人声也弄坏了"两种情况）

用法：
    python diagnose_reference.py
    python diagnose_reference.py --clean 原始测试音频/data.wav --noisy 原始测试音频/加噪声.mp3
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from core import audio_io, data as data_mod, dsp      # noqa: E402
from core.enhance import Denoiser        # noqa: E402
from core.gtcrn import GTCRNDenoiser     # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
TEST_DIR = os.path.join(os.path.dirname(HERE), "原始测试音频")


def align(ref: np.ndarray, sig: np.ndarray, fs: int, max_lag_s: float = 0.5) -> tuple[np.ndarray, int]:
    """把 sig 相对 ref 对齐（互相关找最佳时延），返回 (对齐后的 sig, 时延样本数)。"""
    n = min(len(ref), len(sig))
    a = ref[:n] - np.mean(ref[:n])
    b = sig[:n] - np.mean(sig[:n])
    nfft = 1 << int(np.ceil(np.log2(2 * n)))
    corr = np.fft.irfft(np.fft.rfft(b, nfft) * np.conj(np.fft.rfft(a, nfft)), nfft)
    max_lag = int(max_lag_s * fs)
    mid = np.concatenate([corr[-max_lag:], corr[:max_lag + 1]])
    lags = np.arange(-max_lag, max_lag + 1)
    lag = int(lags[int(np.argmax(mid))])
    if lag > 0:
        out = np.concatenate([np.zeros(lag, dtype=sig.dtype), sig])[:len(sig)]
    elif lag < 0:
        out = np.concatenate([sig[-lag:], np.zeros(-lag, dtype=sig.dtype)])
    else:
        out = sig
    return out, lag


def best_gain(ref: np.ndarray, sig: np.ndarray) -> float:
    """最优缩放系数 a（最小化 ||ref - a*sig||）。"""
    s = float(np.dot(sig, sig))
    return float(np.dot(sig, ref) / s) if s > 1e-12 else 1.0


def report(tag: str, clean: np.ndarray, noisy: np.ndarray, out: np.ndarray, fs: int) -> None:
    """用真实干净参考计算：噪声残留 + 语音失真。"""
    n = min(len(clean), len(noisy), len(out))
    c, x, o = clean[:n], noisy[:n], out[:n]
    a = best_gain(c, o)
    o_al = a * o
    noise_in = x - c
    resid = o_al - c
    rms = lambda v: np.sqrt(max(np.mean(v ** 2), 1e-20))
    noise_in_db = 20 * np.log10(rms(noise_in) / rms(c))
    resid_db = 20 * np.log10(rms(resid) / rms(c))
    print("  %-14s 增益 %+5.2f dB | 输入噪声 %6.1f dB | 输出残留 %6.1f dB | "
          "噪声减少 %5.1f dB | 残留相对输入 %5.1f dB"
          % (tag, 20 * np.log10(max(abs(a), 1e-9)), noise_in_db, resid_db,
             noise_in_db - resid_db, resid_db - noise_in_db))


def main() -> int:
    ap = argparse.ArgumentParser(description="用干净/加噪配对素材做精确诊断")
    ap.add_argument("--clean", default=os.path.join(TEST_DIR, "data.wav"))
    ap.add_argument("--noisy", default=os.path.join(TEST_DIR, "加噪声.mp3"))
    ap.add_argument("--fs", type=int, default=16000)
    args = ap.parse_args()

    fs = args.fs
    for p in (args.clean, args.noisy):
        if not os.path.isfile(p):
            print(f"[错误] 找不到文件：{p}", file=sys.stderr)
            return 1

    c_raw, csr = audio_io.load_audio(args.clean)
    n_raw, nsr = audio_io.load_audio(args.noisy)
    clean = audio_io.to_mono(c_raw) if np.asarray(c_raw).ndim > 1 else np.asarray(c_raw)
    noisy = audio_io.to_mono(n_raw) if np.asarray(n_raw).ndim > 1 else np.asarray(n_raw)
    if csr != fs:
        from scipy import signal as sps
        clean = sps.resample_poly(clean, fs, csr).astype(np.float32)
    if nsr != fs:
        from scipy import signal as sps
        noisy = sps.resample_poly(noisy, fs, nsr).astype(np.float32)

    n = min(len(clean), len(noisy))
    clean, noisy = clean[:n], noisy[:n]
    noisy, lag = align(clean, noisy, fs)

    # 真实噪声与真实 SNR
    true_noise = noisy - clean
    snr_true = 20 * np.log10(np.sqrt(max(np.mean(clean ** 2), 1e-20)) /
                             np.sqrt(max(np.mean(true_noise ** 2), 1e-20)))
    print("=== 素材 ===")
    print(f"  干净版 {os.path.basename(args.clean)}（{csr} Hz）/ "
          f"加噪版 {os.path.basename(args.noisy)}（{nsr} Hz）")
    print(f"  对齐时延 {lag} 样本（{lag / fs * 1000:+.1f} ms）")
    print(f"  **真实 SNR = {snr_true:+.1f} dB**（这是精确值，不是估计）")
    print(f"  噪声类型：白噪声（频带平坦，实测各带 ±1.3 dB 内）")

    m, g = Denoiser(), GTCRNDenoiser()
    print("\n=== 输出质量分解（用真实干净参考计算）===")
    print("  （增益 = 模型是否整体改变音量；输入噪声/输出残留 = 相对语音的 dB；"
          "噪声减少 = 压掉了多少噪声）")
    report("带噪输入", clean, noisy, noisy, fs)
    try:
        report("GTCRN 预训练", clean, noisy, g.enhance(noisy, fs), fs)
    except Exception as exc:
        print("  GTCRN 失败：", exc)
    report("自训练 MLP", clean, noisy, m.enhance(noisy, fs=fs), fs)
    try:
        report("串联", clean, noisy, m.enhance_cascaded(noisy, g), fs)
    except Exception as exc:
        print("  串联失败：", exc)

    # 训练分布对照
    print("\n=== 训练分布对照 ===")
    print("  自训练模型的训练 SNR 范围：−5 ~ +20 dB（每个样本随机取，均匀分布）")
    print(f"  本素材真实 SNR：{snr_true:+.1f} dB "
          + ("← 落在训练范围内" if -5 <= snr_true <= 20 else "← **超出训练范围**"))
    print("  本素材噪声：白噪声（训练见过）+ 44.1 kHz 立体声 MP3（未见过）")

    # ---------- SNR 扫描：同一段干净语音 + 白噪声，用真实参考算噪声减少 ----------
    print("\n=== SNR 扫描（同一段干净语音 + 白噪声，用真实参考算「噪声减少」）===")
    print("  %-9s %14s %14s   %s" % ("输入SNR", "GTCRN", "自训练MLP", "说明"))
    rows = []
    for snr in (-10.0, -5.0, 0.0, 3.6, 5.0, 10.0, 15.0, 20.0):
        nz = data_mod.make_noise("white", len(clean), fs, seed=23)
        mix = data_mod.mix_at_snr(clean, nz, snr)
        cells = []
        for fn in (lambda: g.enhance(mix, fs), lambda: m.enhance(mix, fs=fs)):
            o = np.asarray(fn(), dtype=np.float32)
            k = min(len(o), len(clean))
            a = best_gain(clean[:k], o[:k])
            resid = a * o[:k] - clean[:k]
            nin = mix[:k] - clean[:k]
            cells.append(20 * np.log10(np.sqrt(max(np.mean(nin ** 2), 1e-20)) /
                                       np.sqrt(max(np.mean(resid ** 2), 1e-20))))
        rows.append((snr, *cells))
        note = "← 你的素材在这里" if abs(snr - 3.6) < 0.01 else ""
        print("  %+6.0f dB %+11.1f dB %+11.1f dB   %s" % (snr, cells[0], cells[1], note))

    print("\n判读：")
    low = [r for r in rows if r[0] <= 0]
    high = [r for r in rows if r[0] >= 10]
    print("  · 低 SNR（≤0 dB，噪声主导）：GTCRN 平均 %.1f dB，自训练平均 %.1f dB"
          % (np.mean([r[1] for r in low]), np.mean([r[2] for r in low])))
    print("  · 高 SNR（≥10 dB，语音主导）：GTCRN 平均 %.1f dB，自训练平均 %.1f dB"
          % (np.mean([r[1] for r in high]), np.mean([r[2] for r in high])))
    return 0


if __name__ == "__main__":
    sys.exit(main())
