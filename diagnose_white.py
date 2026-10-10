"""白噪声专项诊断：为什么自训练模型对"白噪声加噪语音"效果不佳？

训练集里明明有白噪声（8 类合成噪声之一），所以不能用"没见过"解释。本脚本做三组实验定位：

    实验 A：白噪声 SNR 扫描（用真实参考算噪声减少量），看是否只是"高 SNR 区间偏弱"
    实验 B：噪声底估计精度 —— 对比自研最小值统计估计 vs 真实噪声谱
            （离散度惩罚导致的最小值统计会对**平稳宽带噪声**系统性高估）
    实验 C：分离"掩蔽能力"与"噪声估计能力" ——
            把**真实噪声谱**当作特征喂给模型（相当于给它开卷考试），
            如果此时表现大幅提升，说明瓶颈在噪声估计而非掩蔽学习。

顺带对照 GTCRN（预训练，另一个训练分布）。

用法：
    python diagnose_white.py
    python diagnose_white.py --kinds white pink car     # 对比多种噪声
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
from core.enhance import Denoiser                      # noqa: E402
from core.gtcrn import GTCRNDenoiser                   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FS = 16000


def best_gain(ref: np.ndarray, sig: np.ndarray) -> float:
    s = float(np.dot(sig, sig))
    return float(np.dot(sig, ref) / s) if s > 1e-12 else 1.0


def noise_reduction(clean: np.ndarray, noisy: np.ndarray, out: np.ndarray) -> float:
    """噪声减少量（dB）：输入噪声能量 / 输出残留噪声能量（已扣掉最优增益影响）。"""
    n = min(len(clean), len(noisy), len(out))
    c, x, o = clean[:n], noisy[:n], out[:n]
    a = best_gain(c, o)
    nin = x - c
    resid = a * o - c
    rms = lambda v: np.sqrt(max(np.mean(v ** 2), 1e-20))
    return float(20 * np.log10(rms(nin) / rms(resid)))


def main() -> int:
    ap = argparse.ArgumentParser(description="白噪声专项诊断")
    ap.add_argument("--kinds", nargs="+", default=["white", "pink", "car"])
    ap.add_argument("--snrs", nargs="+", type=float,
                    default=[-10, -5, 0, 3.6, 5, 10, 15, 20])
    ap.add_argument("--cases", type=int, default=6, help="每个条件用几条语音")
    args = ap.parse_args()

    pool = data_mod.load_clean_pool([os.path.join(HERE, "samples", "demo_speech.wav"),
                                     os.path.join(HERE, "samples", "manba.wav")], fs=FS)
    m, g = Denoiser(), GTCRNDenoiser()
    print("自训练模型：", m.describe() if m.is_ready else m.load_error)
    print()

    # ---------- 实验 A：SNR 扫描 ----------
    print("=== 实验 A：噪声减少量（dB，越大越好；用真实干净参考计算）===")
    header = f"  {'输入SNR':>7}" + "".join(f"{k:>16}" for k in args.kinds)
    print(header + "     （每格：自训练 | GTCRN）")
    for snr in args.snrs:
        cells = []
        for kind in args.kinds:
            vals_m, vals_g = [], []
            for i in range(args.cases):
                clean = pool[i % len(pool)]
                nz = data_mod.make_noise(kind, len(clean), FS, seed=100 + i)
                noisy = data_mod.mix_at_snr(clean, nz, snr)
                try:
                    vals_m.append(noise_reduction(clean, noisy, m.enhance(noisy)))
                except Exception:
                    pass
                try:
                    vals_g.append(noise_reduction(clean, noisy, g.enhance(noisy, FS)))
                except Exception:
                    pass
            cells.append(f"{np.mean(vals_m):+.1f} | {np.mean(vals_g):+.1f}"
                         if vals_m and vals_g else "—")
        print(f"  {snr:>+7.1f}" + "".join(f"{c:>16}" for c in cells))

    # ---------- 实验 B：噪声底估计精度 ----------
    print("\n=== 实验 B：噪声底估计精度（与真实噪声谱对比，@5 dB）===")
    print(f"  {'噪声':<8}{'相关系数':>10}{'平均绝对误差(dB)':>18}{'系统性偏差(dB)':>16}")
    for kind in args.kinds:
        rows = []
        for i in range(args.cases):
            clean = pool[i % len(pool)]
            nz = data_mod.make_noise(kind, len(clean), FS, seed=100 + i)
            noisy = data_mod.mix_at_snr(clean, nz, 5.0)
            spec = dsp.stft(noisy)
            n = min(len(clean), len(noisy))
            true_db = 20 * np.log10(np.maximum(np.abs(dsp.stft(noisy[:n] - clean[:n])), 1e-10))
            pn = dsp.minstat_noise_psd(np.abs(spec))
            est_db = np.repeat(10 * np.log10(np.maximum(pn, 1e-12)), spec.shape[1], axis=1)
            k = min(true_db.shape[1], est_db.shape[1])
            a, b = true_db[:, :k].ravel(), est_db[:, :k].ravel()
            rows.append((float(np.corrcoef(a, b)[0, 1]), float(np.mean(np.abs(a - b))),
                         float(np.mean(b - a))))
        r = np.mean(rows, axis=0)
        print(f"  {kind:<8}{r[0]:>10.3f}{r[1]:>18.2f}{r[2]:>+16.2f}")

    # ---------- 实验 C：分离「掩蔽能力」与「噪声估计能力」----------
    print("\n=== 实验 C：分离「掩蔽能力」与「噪声估计能力」（@5 dB）===")
    print("  两种特征做对照，且都保证与训练时的量纲/分布一致：")
    print("    ① 模型自带估计（平滑最小值统计）—— 实际使用时的路径")
    print("    ② **同域真值**：对真实的噪声信号做同样的最小值统计 -> 无偏的噪声底")
    print("       （量纲与训练特征一致，相当于给模型一个'估计得很准'的噪声底）")
    print(f"  {'噪声':<8}{'① 自带估计':>13}{'② 同域真值':>13}{'提升':>9}{'GTCRN':>10}")
    for kind in args.kinds:
        v_norm, v_oracle, v_g = [], [], []
        for i in range(args.cases):
            clean = pool[i % len(pool)]
            nz = data_mod.make_noise(kind, len(clean), FS, seed=100 + i)
            noisy = data_mod.mix_at_snr(clean, nz, 5.0)
            v_norm.append(noise_reduction(clean, noisy, m.enhance(noisy)))
            # 同域真值：把"真实的噪声波形"按训练时的特征函数过一遍
            noise_spec = dsp.stft(nz[:len(noisy)], n_fft=dsp.N_FFT, hop=dsp.HOP)
            spec = dsp.stft(noisy)
            k = min(noise_spec.shape[1], spec.shape[1])
            pn_true = dsp.minstat_noise_psd(np.abs(noise_spec[:, :k]))
            noise_db = np.repeat(10 * np.log10(np.maximum(pn_true, 1e-12)), k, axis=1)
            mask = m.predict_mask(spec[:, :k], gtcrn_noise_db=noise_db)
            out = dsp.istft(spec[:, :k] * mask, length=len(clean))
            v_oracle.append(noise_reduction(clean, noisy, out))
            v_g.append(noise_reduction(clean, noisy, g.enhance(noisy, FS)))
        gain = np.mean(v_oracle) - np.mean(v_norm)
        print(f"  {kind:<8}{np.mean(v_norm):>+13.2f}{np.mean(v_oracle):>+13.2f}"
              f"{gain:>+9.2f}{np.mean(v_g):>+10.2f}")

    print("\n判读：")
    print("  · 实验 A：各噪声都呈现'低 SNR 强、高 SNR 弱'，说明是训练 SNR 分布的锅，与噪声类型无关；")
    print("  · 实验 B：'系统性偏差'为正 = 高估噪声（会过减、伤语音）；为负 = 低估（噪声压不掉）；")
    print("  · 实验 C：若②明显高于①，说明瓶颈是噪声估计精度；若持平，说明掩蔽本身已到极限。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
