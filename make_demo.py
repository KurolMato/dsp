"""导出可试听的对比样本：原始干净语音 / 带噪语音 / 神经网络降噪 / 经典降噪。

每种噪声导出一组，放在 results/audio_demo/ 下，用播放器 A/B 对比即可听出差别。
同时打印每种组合的指标，方便写报告。

用法：
    python make_demo.py                 # 默认 6 种噪声 × 3 档 SNR
    python make_demo.py --kinds white hum --snrs 5
"""

from __future__ import annotations

import argparse
import json
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
from eval_compare import log_spectral_distance, noise_suppression, segmental_snr  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "results", "audio_demo")


def main() -> int:
    ap = argparse.ArgumentParser(description="导出降噪前后对比音频（可试听）")
    ap.add_argument("--clean", default=audio_io.default_sample_path())
    ap.add_argument("--fs", type=int, default=16000)
    ap.add_argument("--kinds", nargs="+", default=["white", "hum", "babble", "car", "pink", "ac"])
    ap.add_argument("--snrs", nargs="+", type=float, default=[0.0, 5.0, 10.0])
    ap.add_argument("--model", default=None)
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    fs = args.fs
    y, sr = audio_io.load_audio(os.path.abspath(args.clean))
    clean = audio_io.to_mono(y)
    if sr != fs:
        from scipy import signal as sps
        clean = sps.resample(clean, int(round(len(clean) * fs / sr))).astype(np.float32)

    den = Denoiser(args.model)
    print(f"语音：{os.path.basename(args.clean)}，{len(clean) / fs:.2f} s")
    print(f"神经网络：{den.describe() if den.is_ready else '未加载（只导出经典降噪结果）'}\n")

    table = []
    for i, kind in enumerate(args.kinds):
        for snr_in in args.snrs:
            noise = data_mod.make_noise(kind, len(clean), fs, seed=100 + i)
            noisy = data_mod.mix_at_snr(clean, noise, snr_in)
            out_cl = dsp.classic_denoise(noisy, fs)
            out_nn = den.enhance(noisy, fs=fs) if den.is_ready else None
            tag = f"{kind}_{int(snr_in):+03d}dB"
            audio_io.save_audio(os.path.join(OUT_DIR, f"{tag}_1_clean.wav"), clean, fs)
            audio_io.save_audio(os.path.join(OUT_DIR, f"{tag}_2_noisy.wav"), noisy, fs)
            if out_nn is not None:
                audio_io.save_audio(os.path.join(OUT_DIR, f"{tag}_3_neural.wav"), out_nn, fs)
            audio_io.save_audio(os.path.join(OUT_DIR, f"{tag}_4_classic.wav"), out_cl, fs)

            row = {
                "kind": kind, "snr_in": snr_in,
                "snr_noisy": dsp.mixing_snr(noisy, clean),
                "snr_nn": dsp.mixing_snr(out_nn, clean) if out_nn is not None else None,
                "snr_classic": dsp.mixing_snr(out_cl, clean),
                "lsd_noisy": log_spectral_distance(clean, noisy),
                "lsd_nn": log_spectral_distance(clean, out_nn) if out_nn is not None else None,
                "lsd_classic": log_spectral_distance(clean, out_cl),
                "ssnr_nn": segmental_snr(clean, out_nn, fs) if out_nn is not None else None,
                "ssnr_classic": segmental_snr(clean, out_cl, fs),
                "noise_supp_nn": noise_suppression(clean, noisy, out_nn, fs) if out_nn is not None else None,
                "noise_supp_classic": noise_suppression(clean, noisy, out_cl, fs),
            }
            table.append(row)
            print(f"{data_mod.NOISE_LABELS.get(kind, kind):<12} {snr_in:>4.0f} dB | "
                  f"神经 ΔSNR {row['snr_nn'] - row['snr_noisy']:+5.2f} dB（LSD {row['lsd_nn']:.2f}，"
                  f"噪声抑制 {row['noise_supp_nn']:.1f} dB）| "
                  f"经典 ΔSNR {row['snr_classic'] - row['snr_noisy']:+5.2f} dB（LSD {row['lsd_classic']:.2f}，"
                  f"噪声抑制 {row['noise_supp_classic']:.1f} dB）" if out_nn is not None else "")

    with open(os.path.join(OUT_DIR, "demo_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(table, f, ensure_ascii=False, indent=2)
    print(f"\n音频已导出到 {OUT_DIR}")
    print("  文件命名：<噪声>_<SNR>_1_clean / 2_noisy / 3_neural / 4_classic.wav，可直接 A/B 试听")
    return 0


if __name__ == "__main__":
    sys.exit(main())
