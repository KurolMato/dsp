"""对比评估：神经网络降噪 vs 经典谱减法 vs 掩蔽类理想参考（非严格上限）。

为什么需要这个脚本？
    只用 SNR 提升不能完整评价降噪。理想幅度配带噪相位及真实噪声 IRM
    只是参考方案，不构成任意模型都不能超过的严格上限。这里给出多个指标：

        ΔSNR   —— 相对于干净参考的信噪比提升（越大越好）
        LSD    —— 对数谱失真（越小越好，反映语音失真）
        SSNR   —— 分段信噪比（越大越好）
        噪声抑制 —— 静音段能量下降多少 dB（听感上的"噪声变小了"）

用法：
    python eval_compare.py                        # 用自带 wav + 8 种噪声 × 3 档 SNR
    python eval_compare.py --clean ..\\data.wav --snrs 0 5 10
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from core import audio_io, data as data_mod, dsp          # noqa: E402
from core.corpus import evaluation_pool
from core.enhance import Denoiser                          # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.join(HERE, "results")


# ----------------------------------------------------------------------------
# 指标
# ----------------------------------------------------------------------------
def log_spectral_distance(ref: np.ndarray, test: np.ndarray) -> float:
    """对数谱失真（dB，越小越好）。"""
    n = min(len(ref), len(test))
    sr = np.abs(dsp.stft(ref[:n])) ** 2
    st = np.abs(dsp.stft(test[:n])) ** 2
    m = min(sr.shape[1], st.shape[1])
    sr, st = np.maximum(sr[:, :m], 1e-12), np.maximum(st[:, :m], 1e-12)
    num = sr.sum(axis=1, keepdims=True)
    den = st.sum(axis=1, keepdims=True)
    ratio = np.maximum(sr * den / np.maximum(st * num, 1e-12), 1e-12)
    return float(np.mean(np.sqrt(np.mean((10 * np.log10(ratio)) ** 2, axis=0))))


def segmental_snr(ref: np.ndarray, test: np.ndarray, fs: int = 16000,
                  frame: float = 0.02, hop: float = 0.01) -> float:
    """分段信噪比（dB，越大越好）。"""
    fl, hs = int(frame * fs), int(hop * fs)
    n = min(len(ref), len(test))
    idx = np.arange(fl)[None, :] + hs * np.arange(max(1, (n - fl) // hs + 1))[:, None]
    r, t = ref[:n][idx], test[:n][idx]
    pr, pe = (r ** 2).sum(axis=1), ((r - t) ** 2).sum(axis=1)
    ok = pr > 1e-6
    if not ok.any():
        return float("nan")
    return float(np.mean(np.clip(10 * np.log10(np.maximum(pr[ok], 1e-12) / np.maximum(pe[ok], 1e-12)), -10, 35)))


def noise_suppression(clean: np.ndarray, noisy: np.ndarray, out: np.ndarray, fs: int = 16000) -> float:
    """静音段（噪声-only）能量下降 dB：衡量听感上的噪声降低。"""
    fl, hs = int(0.02 * fs), int(0.01 * fs)
    n = min(len(clean), len(noisy), len(out))
    idx = np.arange(fl)[None, :] + hs * np.arange(max(1, (n - fl) // hs + 1))[:, None]
    e_c = (clean[:n][idx] ** 2).sum(axis=1)
    sil = e_c <= max(np.percentile(e_c, 20), 1e-9)      # 能量最低的 20% 帧当作静音段
    if not sil.any():
        return float("nan")
    e_n = (noisy[:n][idx][sil] ** 2).sum(axis=1).mean()
    e_o = (out[:n][idx][sil] ** 2).sum(axis=1).mean()
    return float(10 * np.log10(max(e_n, 1e-20) / max(e_o, 1e-20)))


def oracle_ceilings(clean: np.ndarray, noisy: np.ndarray) -> dict:
    """两个使用已知干净语音的理想参考：干净幅度+带噪相位（幅度参考）、理想掩蔽 IRM。"""
    S = np.abs(dsp.stft(clean))
    Y = np.abs(dsp.stft(noisy))
    ph = np.angle(dsp.stft(noisy))
    ir = np.clip(S / (S + np.abs(dsp.stft(noisy - clean)) + 1e-8), 0, 1)
    return {
        "phase_only": dsp.mixing_snr(dsp.istft(S * np.exp(1j * ph), length=len(clean)), clean),
        "ideal_mask": dsp.mixing_snr(dsp.istft(Y * ir * np.exp(1j * ph), length=len(clean)), clean),
    }


# ----------------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="对比评估：神经网络 vs 经典谱减法 vs 理想参考（非严格上限）")
    ap.add_argument("--clean", nargs="+", default=None)
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--fs", type=int, default=16000)
    ap.add_argument("--snrs", nargs="+", type=float, default=[0.0, 5.0, 10.0])
    ap.add_argument("--kinds", nargs="+", default=["white", "pink", "hum", "babble", "car", "ac", "click", "brown"])
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--model", type=str, default=None)
    ap.add_argument("--out-json", type=str, default=os.path.join(RESULT_DIR, "compare_report.json"))
    args = ap.parse_args()

    os.makedirs(RESULT_DIR, exist_ok=True)
    fs = args.fs
    denoiser = Denoiser(args.model)
    if not denoiser.is_ready:
        print(f"[警告] 神经网络模型不可用（{denoiser.load_error}），只评估经典方法。\n")
    else:
        print(f"神经网络：{denoiser.describe()}\n")

    pool = evaluation_pool(args.clean, args.model, args.manifest, fs=fs)
    clean = pool[0]
    print(f"测试语音：{len(pool)} 段，本次使用第一段 {len(clean) / fs:.2f} s\n")

    rows = []
    print(f"{'噪声':<12}{'SNR入':>6}{'带噪':>8}{'神经':>8}{'经典':>8}{'幅度参考':>9}{'IRM':>8}"
          f"{'Δ神经':>8}{'Δ经典':>8}{'LSD带噪':>9}{'LSD神经':>9}{'噪声抑制':>9}")
    for i, kind in enumerate(args.kinds):
        for snr_in in args.snrs:
            noise = data_mod.make_noise(kind, len(clean), fs, seed=args.seed * 100 + i)
            noisy = data_mod.mix_at_snr(clean, noise, snr_in)
            row = {"kind": kind, "snr_in": snr_in, "snr_noisy": dsp.mixing_snr(noisy, clean)}
            row.update({f"oracle_{k}": v for k, v in oracle_ceilings(clean, noisy).items()})

            t0 = time.time()
            row["snr_classic"] = dsp.mixing_snr(dsp.classic_denoise(noisy, fs), clean)
            row["t_classic"] = time.time() - t0

            if denoiser.is_ready:
                t0 = time.time()
                out_ml = denoiser.enhance(noisy, fs=fs)
                row["t_nn"] = time.time() - t0
                row["snr_nn"] = dsp.mixing_snr(out_ml, clean)
                row["lsd_nn"] = log_spectral_distance(clean, out_ml)
                row["ssnr_nn"] = segmental_snr(clean, out_ml, fs)
                row["noise_supp_nn"] = noise_suppression(clean, noisy, out_ml, fs)
            out_cl = dsp.classic_denoise(noisy, fs)
            row["lsd_noisy"] = log_spectral_distance(clean, noisy)
            row["lsd_classic"] = log_spectral_distance(clean, out_cl)
            row["ssnr_noisy"] = segmental_snr(clean, noisy, fs)
            row["ssnr_classic"] = segmental_snr(clean, out_cl, fs)
            row["noise_supp_classic"] = noise_suppression(clean, noisy, out_cl, fs)
            row["d_snr_nn"] = row.get("snr_nn", float("nan")) - row["snr_noisy"]
            row["d_snr_classic"] = row["snr_classic"] - row["snr_noisy"]
            rows.append(row)

            print(f"{data_mod.NOISE_LABELS.get(kind, kind):<12}{snr_in:>6.0f}{row['snr_noisy']:>8.2f}"
                  f"{row.get('snr_nn', float('nan')):>8.2f}{row['snr_classic']:>8.2f}"
                  f"{row['oracle_phase_only']:>9.2f}{row['oracle_ideal_mask']:>8.2f}"
                  f"{row['d_snr_nn']:>8.2f}{row['d_snr_classic']:>8.2f}"
                  f"{row['lsd_noisy']:>9.2f}{row.get('lsd_nn', float('nan')):>9.2f}"
                  f"{row.get('noise_supp_nn', float('nan')):>9.1f}")

    def mean(key: str) -> float:
        vals = [r[key] for r in rows if key in r and np.isfinite(r[key])]
        return float(np.mean(vals)) if vals else float("nan")

    summary = {
        "n_cases": len(rows),
        "snr_in_mean": mean("snr_in"),
        "d_snr_nn": mean("d_snr_nn"),
        "d_snr_classic": mean("d_snr_classic"),
        # 两个上限换算成"相对输入的提升"，才可与 ΔSNR 直接比较
        "ceiling_phase_db": mean("oracle_phase_only") - mean("snr_noisy"),
        "ceiling_irm_db": mean("oracle_ideal_mask") - mean("snr_noisy"),
        "lsd_noisy": mean("lsd_noisy"),
        "lsd_nn": mean("lsd_nn"),
        "lsd_classic": mean("lsd_classic"),
        "ssnr_noisy": mean("ssnr_noisy"),
        "ssnr_nn": mean("ssnr_nn"),
        "ssnr_classic": mean("ssnr_classic"),
        "noise_supp_nn_db": mean("noise_supp_nn"),
        "noise_supp_classic_db": mean("noise_supp_classic"),
        "t_nn_sec": mean("t_nn"),
        "t_classic_sec": mean("t_classic"),
        "model": denoiser.describe() if denoiser.is_ready else None,
    }

    print("\n===== 平均结果（%d 个用例）=====" % len(rows))
    print(f"  输入平均 SNR            {summary['snr_in_mean']:.2f} dB")
    print(f"  理想参考（非严格上限）（干净幅度+带噪相位） {summary['ceiling_phase_db']:+.2f} dB")
    print(f"  理想参考（非严格上限）（理想掩蔽 IRM）     {summary['ceiling_irm_db']:+.2f} dB")
    print(f"  神经网络降噪 ΔSNR        {summary['d_snr_nn']:+.2f} dB   （耗时 {summary['t_nn_sec'] * 1000:.0f} ms/次）")
    print(f"  经典谱减法 ΔSNR          {summary['d_snr_classic']:+.2f} dB   （耗时 {summary['t_classic_sec'] * 1000:.0f} ms/次）")
    print(f"  对数谱失真 LSD：带噪 {summary['lsd_noisy']:.2f} -> 神经 {summary['lsd_nn']:.2f} / 经典 {summary['lsd_classic']:.2f} dB")
    print(f"  分段信噪比 SSNR：带噪 {summary['ssnr_noisy']:.2f} -> 神经 {summary['ssnr_nn']:.2f} / 经典 {summary['ssnr_classic']:.2f} dB")
    print(f"  静音段噪声抑制：神经 {summary['noise_supp_nn_db']:.1f} dB / 经典 {summary['noise_supp_classic_db']:.1f} dB")

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "rows": rows}, f, ensure_ascii=False, indent=2)
    print(f"\n报告已保存：{args.out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
