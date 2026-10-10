"""模型版本对比基准：在同一批固定用例上比较经典方法与各版神经网络。

固定基准的意义：不同模型必须在**同一批带噪样本**上比较，否则噪声类型/SNR 每次抽样
不同，数字无法对比（本项目调试时正因此被误导过一次）。

用法：
    python bench_models.py
    python bench_models.py --model models/denoise_mlp.npz --model models_v2/denoise_mlp.npz
"""

from __future__ import annotations

import argparse
import glob
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

from core import data as dm, dsp                              # noqa: E402
from core.corpus import evaluation_pool
from core.enhance import Denoiser                             # noqa: E402
from eval_compare import log_spectral_distance, noise_suppression, segmental_snr  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
KINDS = ("white", "pink", "hum", "babble", "car", "ac")
SNRS = (0.0, 5.0, 10.0)


def build_cases(fs: int = 16000, manifest=None):
    pool = evaluation_pool(manifest_path=manifest, fs=fs)
    rng = np.random.default_rng(0)
    val = [pool[i] for i in rng.permutation(len(pool))[:3]]
    cases = []
    for i, kind in enumerate(KINDS):
        for s_in in SNRS:
            c = val[i % len(val)]
            nz = dm.make_noise(kind, len(c), fs, seed=11 + i)
            cases.append((kind, s_in, c, dm.mix_at_snr(c, nz, s_in)))
    return cases


def evaluate(name: str, enhance, cases, ret_rows=False, fs=16000):
    dsnr, dlsd, dssnr, dns = [], [], [], []
    per_kind: dict[str, list[float]] = {}
    elapsed = 0.0
    for kind, s_in, c, noisy in cases:
        t0 = time.time()
        out = enhance(noisy)
        elapsed += time.time() - t0
        d = dsp.mixing_snr(out, c) - s_in
        dsnr.append(d)
        dlsd.append(log_spectral_distance(c, noisy) - log_spectral_distance(c, out))
        dssnr.append(segmental_snr(c, out, fs) - segmental_snr(c, noisy, fs))
        dns.append(noise_suppression(c, noisy, out, fs))
        per_kind.setdefault(kind, []).append(d)
    ms = 1000 * elapsed / len(cases)
    print(f"{name:<24} ΔSNR {np.mean(dsnr):+6.2f} dB | LSD 改善 {np.mean(dlsd):+5.2f} dB | "
          f"SSNR 改善 {np.mean(dssnr):+5.2f} dB | 噪声抑制 {np.mean(dns):5.1f} dB | {ms:5.0f} ms/次")
    print("      " + "  ".join(f"{k} {np.mean(v):+.1f}" for k, v in per_kind.items()))
    return float(np.mean(dsnr))


def main() -> int:
    ap = argparse.ArgumentParser(description="同一基准下对比各版降噪模型")
    ap.add_argument("--model", action="append", default=None, help="模型文件（可多次指定）")
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--fs", type=int, default=16000)
    args = ap.parse_args()

    models = args.model or sorted(glob.glob(os.path.join(HERE, "models*", "denoise_mlp.npz")))
    cases = build_cases(args.fs, args.manifest)
    print(f"=== 固定基准：{len(KINDS)} 类噪声 × {len(SNRS)} 档 SNR = {len(cases)} 个用例"
          f"（语料隔离状态见上方提示）===\n")

    evaluate("经典谱减法（基线）", lambda x: dsp.classic_denoise(x, args.fs), cases, fs=args.fs)
    for path in models:
        d = Denoiser(path)
        if not d.is_ready:
            print(f"{os.path.relpath(path, HERE):<24} 无法加载：{d.load_error}")
            continue
        print(f"  [{os.path.relpath(path, HERE)}] {d.describe()}")
        evaluate("  神经网络降噪", lambda x, dd=d: dd.enhance(x, fs=args.fs), cases, fs=args.fs)

    ceil_p, ceil_i = [], []
    for kind, s_in, c, noisy in cases:
        S = np.abs(dsp.stft(c))
        Y = np.abs(dsp.stft(noisy))
        ph = np.angle(dsp.stft(noisy))
        irm = np.clip(S / (S + np.abs(dsp.stft(noisy - c)) + 1e-8), 0, 1)
        ceil_p.append(dsp.mixing_snr(dsp.istft(S * np.exp(1j * ph), length=len(c)), c) - s_in)
        ceil_i.append(dsp.mixing_snr(dsp.istft(Y * irm * np.exp(1j * ph), length=len(c)), c) - s_in)
    print(f"\n理想参考（非严格上限）：干净幅度+带噪相位 {np.mean(ceil_p):+.2f} dB | 理想掩蔽 IRM {np.mean(ceil_i):+.2f} dB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
