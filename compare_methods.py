"""四种降噪方法在同一批带噪样本上的公平对比。

为什么要单独写一个脚本：在 GUI 里对"本来就干净的录音"做降噪是没有意义的
（实测底噪只有 −74 dB，任何方法都会过度处理）。要公平比较就必须用**已知干净
参考、已知 SNR** 的合成带噪样本，这样才能算 ΔSNR、对数谱失真等有参考指标。

四种方法：
    1. GTCRN（预训练，ICASSP 2024，DNS3 上训练，onnxruntime 推理）
    2. 自训练 MLP（本项目用你的录音 + 本地合成噪声训练）
    3. 经典谱减法（平滑最小值统计噪声估计 + 维纳平滑）
    4. 带阻滤波（原 MATLAB 的做法，仅压 1750~1850 Hz）
另外给出两个理想参考（非严格上限）作为标尺。

用法：
    python compare_methods.py                       # 默认 8 类噪声 × 3 档 SNR
    python compare_methods.py --kinds white hum --snrs 5
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
from core.gtcrn import GTCRNDenoiser                       # noqa: E402
from eval_compare import log_spectral_distance, noise_suppression, segmental_snr  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.join(HERE, "results")


def main() -> int:
    ap = argparse.ArgumentParser(description="四种降噪方法的公平对比")
    ap.add_argument("--clean", nargs="+", default=None)
    ap.add_argument("--manifest", default=None, help="训练保存的 split_manifest.json")
    ap.add_argument("--fs", type=int, default=16000)
    ap.add_argument("--kinds", nargs="+",
                    default=["white", "pink", "hum", "babble", "car", "ac", "click", "brown"])
    ap.add_argument("--snrs", nargs="+", type=float, default=[0.0, 5.0, 10.0])
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=os.path.join(RESULT_DIR, "compare_methods.json"))
    args = ap.parse_args()

    fs = args.fs
    pool = evaluation_pool(args.clean, manifest_path=args.manifest, fs=fs)
    rng = np.random.default_rng(args.seed)
    val = [pool[i] for i in rng.permutation(len(pool))[:3]]

    gtcrn = GTCRNDenoiser()
    mlp = Denoiser()
    print("参与对比的方法：")
    print(f"  1. {gtcrn.describe() if gtcrn.is_ready else 'GTCRN 不可用：' + str(gtcrn.load_error)}")
    print(f"  2. {mlp.describe() if mlp.is_ready else '自训练 MLP 不可用'}")
    print("  3. 经典谱减法（平滑最小值统计 + 维纳平滑）")
    print("  4. 带阻滤波（原 MATLAB 做法）\n")

    rows = []
    header = f"{'噪声':<11}{'SNR入':>6}" + "".join(f"{n:>16}" for n in
                                                 ("GTCRN", "自训练MLP", "经典谱减", "带阻滤波")) + f"{'参考(相位)':>12}{'参考(IRM)':>11}"
    print(header)
    for i, kind in enumerate(args.kinds):
        for snr_in in args.snrs:
            clean = val[i % len(val)]
            noise = data_mod.make_noise(kind, len(clean), fs, seed=args.seed * 100 + i)
            noisy = data_mod.mix_at_snr(clean, noise, snr_in)

            # 理想参考（非严格上限）
            S = np.abs(dsp.stft(clean))
            Y = np.abs(dsp.stft(noisy))
            ph = np.angle(dsp.stft(noisy))
            ceil_phase = dsp.mixing_snr(dsp.istft(S * np.exp(1j * ph), length=len(clean)), clean) - snr_in
            irm = np.clip(S / (S + np.abs(dsp.stft(noisy - clean)) + 1e-8), 0, 1)
            ceil_irm = dsp.mixing_snr(dsp.istft(Y * irm * np.exp(1j * ph), length=len(clean)), clean) - snr_in

            outs: dict[str, np.ndarray] = {}
            t0 = time.time()
            if gtcrn.is_ready:
                outs["GTCRN"] = gtcrn.enhance(noisy, fs)
            t_gtcrn = time.time() - t0
            if mlp.is_ready:
                outs["自训练MLP"] = mlp.enhance(noisy, fs=fs)
            outs["经典谱减"] = dsp.classic_denoise(noisy, fs)
            bs = dsp.design_bandstop(fs)
            from scipy import signal as sps
            outs["带阻滤波"] = sps.filtfilt(bs["b"], bs["a"], noisy.astype(np.float64)).astype(np.float32)

            row = {"kind": kind, "snr_in": snr_in, "ceiling_phase_db": ceil_phase,
                   "ceiling_irm_db": ceil_irm, "t_gtcrn": t_gtcrn}
            cells = []
            for name, out in outs.items():
                d = dsp.mixing_snr(out, clean) - snr_in
                row[name] = {
                    "d_snr": float(d),
                    "lsd_noisy": log_spectral_distance(clean, noisy),
                    "lsd": log_spectral_distance(clean, out),
                    "ssnr": segmental_snr(clean, out, fs),
                    "ssnr_noisy": segmental_snr(clean, noisy, fs),
                    "noise_supp": noise_suppression(clean, noisy, out, fs),
                }
                cells.append(f"{d:+9.2f}/{row[name]['lsd']:5.1f}")
            rows.append(row)
            print(f"{data_mod.NOISE_LABELS.get(kind, kind):<11}{snr_in:>6.0f}"
                  + "".join(f"{c:>16}" for c in cells)
                  + f"{ceil_phase:>12.2f}{ceil_irm:>11.2f}")

    def mean_of(method: str, key: str) -> float:
        vals = [r[method][key] for r in rows if method in r]
        return float(np.mean(vals)) if vals else float("nan")

    print("\n（每格 = ΔSNR dB / 对数谱失真 LSD dB，ΔSNR 越大越好，LSD 越小越保真）")
    print("\n===== 平均值汇总 =====")
    print(f"{'方法':<14}{'ΔSNR':>9}{'LSD带噪':>9}{'LSD降噪':>9}{'SSNR改善':>10}{'噪声抑制':>10}")
    for m in ("GTCRN", "自训练MLP", "经典谱减", "带阻滤波"):
        if not any(m in r for r in rows):
            continue
        d = mean_of(m, "d_snr")
        l0 = mean_of(m, "lsd_noisy")
        l1 = mean_of(m, "lsd")
        s = mean_of(m, "ssnr") - mean_of(m, "ssnr_noisy")
        ns = mean_of(m, "noise_supp")
        print(f"{m:<14}{d:>+9.2f}{l0:>9.2f}{l1:>9.2f}{s:>+10.2f}{ns:>9.1f} dB")
    print(f"{'理想参考（非严格上限）':<14}{np.mean([r['ceiling_phase_db'] for r in rows]):>+9.2f}"
          f"{'':>9}{'':>9}{'':>10}{'':>10}  (幅度参考)")
    print(f"{'理想掩蔽IRM':<14}{np.mean([r['ceiling_irm_db'] for r in rows]):>+9.2f}")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print(f"\n明细已保存：{args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
