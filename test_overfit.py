"""过拟合检验：把噪声类型"留出"再训练，检验模型是真会降噪还是只是记住了训练分布。

为什么需要这个脚本
------------------
自训练模型在我们的评测里表现很好（真实 DEMAND 噪声上 ΔSNR +7.83 dB），
但它的训练噪声与评测噪声来自**同一批场景**——这不足以排除"过拟合"的嫌疑。
本脚本用最直接的办法检验：**训练时故意排除若干噪声类型**，再用这些"从未见过"的
噪声类型评测，对比"见过 vs 没见过"的差距。

判读标准
--------
- 留出模型在未见噪声上**仍显著为正** → 学到的是可泛化的降噪规律；
- 若留出模型在未见噪声上接近 0 甚至为负 → 说明此前的好成绩主要来自分布匹配（过拟合）。

用法
----
    # 1) 先训一个"留出 car/ac/click"的模型（约 3 分钟）
    python train_model.py --clean ..\\data.wav ..\\manba.wav \\
        --holdout-kinds car ac click --epochs 12 --steps-per-epoch 30 \\
        --batch-size 6 --context 6 --hidden 256,128 --out-dir models_loo

    # 2) 跑本脚本对比
    python test_overfit.py
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

from core import audio_io, data as dm, dsp          # noqa: E402
from core.corpus import validation_pool
from core.enhance import Denoiser                    # noqa: E402
from eval_compare import log_spectral_distance       # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SEEN = ("white", "pink", "hum", "babble", "brown")
UNSEEN = ("car", "ac", "click")


def evaluate(model: Denoiser, kinds, speech, fs: int = 16000, snrs=(0.0, 5.0, 10.0)):
    ds, ls = [], []
    for i, k in enumerate(kinds):
        for snr in snrs:
            c = speech[i % len(speech)]
            nz = dm.make_noise(k, len(c), fs, seed=41 + i)
            noisy = dm.mix_at_snr(c, nz, snr)
            out = model.enhance(noisy, fs=fs)
            n = min(len(out), len(c))
            ds.append(dsp.mixing_snr(out[:n], c[:n]) - snr)
            ls.append(log_spectral_distance(c[:n], out[:n]) - log_spectral_distance(c[:n], noisy[:n]))
    return float(np.mean(ds)), float(np.mean(ls))


def main() -> int:
    ap = argparse.ArgumentParser(description="留出噪声类型的过拟合检验")
    ap.add_argument("--full", default=os.path.join(HERE, "models", "denoise_mlp.npz"),
                    help="在全部噪声类型上训练的模型")
    ap.add_argument("--loo", default=os.path.join(HERE, "models_loo", "denoise_mlp.npz"),
                    help="排除 car/ac/click 训练的模型")
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--fs", type=int, default=16000)
    args = ap.parse_args()

    manifest_path = args.manifest or os.path.join(os.path.dirname(os.path.abspath(args.full)), "split_manifest.json")
    if not os.path.isfile(manifest_path):
        print("缺少固定验证清单，请用修正后的训练脚本重新训练两种模型，并提供 --manifest。", file=sys.stderr)
        return 2
    val, manifest = validation_pool(manifest_path)
    if manifest["fs"] != args.fs:
        print("采样率与验证清单不一致", file=sys.stderr)
        return 2
    full = Denoiser(args.full)
    loo = Denoiser(args.loo)
    if any(model.meta.get("split_id") != manifest["split_id"] for model in (full, loo)):
        print("两种模型必须使用相同的数据划分清单，旧模型无法证明验证集隔离。", file=sys.stderr)
        return 2

    print("完整模型（见过 8 类噪声）:", full.describe() if full.is_ready else full.load_error)
    print("留出模型（未见过 car/ac/click）:", loo.describe() if loo.is_ready else loo.load_error)
    if not (full.is_ready and loo.is_ready):
        print("\n缺少模型。请先按文件头注释训练 models_loo（约 3 分钟）。", file=sys.stderr)
        return 1

    print(f"\n{'测试噪声':<38}{'完整模型':>20}{'留出模型':>20}")
    rows = []
    for label, kinds in (("未见噪声 car/ac/click（关键）", UNSEEN),
                         ("已见噪声 white/pink/hum/babble/brown", SEEN)):
        a = evaluate(full, kinds, val, args.fs)
        b = evaluate(loo, kinds, val, args.fs)
        rows.append((label, a, b))
        print(f"{label:<38}" + f"ΔSNR {a[0]:+5.2f} LSD {a[1]:+5.2f}".rjust(20)
              + f"ΔSNR {b[0]:+5.2f} LSD {b[1]:+5.2f}".rjust(20))

    gap = rows[1][2][0] - rows[0][2][0]
    print(f"\n判读：留出模型在**未见噪声**上 ΔSNR {rows[0][2][0]:+.2f} dB（对照：已见噪声 "
          f"{rows[1][2][0]:+.2f} dB，相差 {gap:+.2f} dB）")
    if rows[0][2][0] > 1.0:
        print("  → 未见噪声上仍有明显增益，说明学到的是可泛化的降噪规律，不是死记训练分布。")
    elif rows[0][2][0] > -0.5:
        print("  → 未见噪声上增益有限，说明效果相当程度上依赖「训练分布匹配」。")
    else:
        print("  → 未见噪声上反而变差，存在明显过拟合，需要扩充噪声多样性。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
