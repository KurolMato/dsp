"""A/B 试听工具：判断"降噪到底有没有效果"，用耳朵而不是看数字。

为什么需要它
------------
1. 示例音频本身几乎是干净的（底噪约 −74 dB），对它降噪听不出区别很正常；
2. 数字对比（ΔSNR 等）需要干净参考，实际使用中往往没有；
3. 逐段试听时如果不做**响度对齐**，"整体变小"会掩盖真实差别。

这个工具把三件事都处理掉：
    · 现场把指定强度/类型的噪声混进你的干净录音（可造出很吵的测试样本）；
    · 所有版本**峰值响度对齐**，避免"变小了"骗过耳朵；
    · 顺序播放 [干净 → 带噪 → 各降噪方案]，同一段落反复对比。

用法
----
    python ab_listen.py                          # 交互式：输入序号选噪声/强度/方案
    python ab_listen.py --kind babble --snr -5   # 直接指定，做成"很吵"的对照
    python ab_listen.py --export                 # 把对比音频导出到 results/ab_demo/
"""

from __future__ import annotations

import argparse
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

from core import audio_io, data as data_mod, dsp      # noqa: E402
from core.enhance import Denoiser                      # noqa: E402
from core.gtcrn import GTCRNDenoiser                   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DEMO_DIR = os.path.join(HERE, "results", "ab_demo")


def peak_normalize(x: np.ndarray, peak: float = 0.9) -> np.ndarray:
    """峰值归一化：让所有对比版本响度可比，避免"变小了"造成听感误判。"""
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    m = float(np.max(np.abs(x))) if x.size else 0.0
    return (x * (peak / m)).astype(np.float32) if m > 1e-9 else x


def build_variants(clean: np.ndarray, fs: int, kind: str, snr: float,
                   gtcrn=None, mlp=None, normalize: bool = True) -> list[tuple[str, np.ndarray]]:
    """生成 [干净 / 带噪 / 各降噪方案] 的对比列表（可统一响度对齐）。"""
    noise = data_mod.make_noise(kind, len(clean), fs, seed=7)
    noisy = data_mod.mix_at_snr(clean, noise, snr)
    variants: list[tuple[str, np.ndarray]] = [("① 干净原始", clean), (f"② 带噪输入（{snr:+.0f} dB）", noisy)]
    if gtcrn is not None and gtcrn.is_ready:
        variants.append(("③ GTCRN 预训练降噪", gtcrn.enhance(noisy, fs)))
    if mlp is not None and mlp.is_ready:
        variants.append(("④ 自训练 MLP 降噪", mlp.enhance(noisy, fs=fs)))
        if gtcrn is not None and gtcrn.is_ready:
            variants.append(("⑤ 串联 GTCRN→自训练", mlp.enhance_cascaded(noisy, gtcrn, fs=fs)))
    variants.append(("⑥ 经典谱减法", dsp.classic_denoise(noisy, fs)))
    # 长度统一，避免任何长度差异影响听感对比
    n = len(clean)
    out = []
    for name, sig in variants:
        sig = np.asarray(sig, dtype=np.float32).reshape(-1)[:n]
        if len(sig) < n:
            sig = np.pad(sig, (0, n - len(sig)))
        out.append((name, peak_normalize(sig) if normalize else sig))
    return out


def report_levels(variants: list[tuple[str, np.ndarray]], fs: int) -> None:
    """打印每个版本的响度与噪声底，便于理解"听起来为什么不同"。"""
    print(f"\n{'版本':<24}{'峰值':>8}{'RMS(dB)':>10}{'静音段底噪(dB)':>16}")
    for name, sig in variants:
        e = np.sort(np.array([np.sum(f ** 2) for f in np.array_split(sig, max(1, len(sig) // (fs // 50)))]))
        floor = np.sqrt(max(e[:max(1, len(e) // 5)].mean(), 1e-20))
        rms = np.sqrt(max((sig ** 2).mean(), 1e-20))
        print(f"{name:<24}{np.abs(sig).max():>8.3f}{20 * np.log10(rms):>10.1f}"
              f"{20 * np.log10(floor):>16.1f}")


def play_sequence(variants: list[tuple[str, np.ndarray]], fs: int, gap: float = 0.45) -> None:
    for name, sig in variants:
        print(f"    ▶ {name}")
        audio_io.play(sig, fs, blocking=True)
        time.sleep(gap)


def interactive(fs: int, clean: np.ndarray, gtcrn, mlp) -> None:
    kinds = list(data_mod.NOISE_TYPES)
    print("\n可选噪声类型：")
    for i, k in enumerate(kinds, 1):
        print(f"   {i:2d}. {data_mod.NOISE_LABELS[k]}")
    print("   建议：想明显听出降噪效果，选『嘈杂人声 babble』或『车内噪声 car』，强度用 -5 dB")
    while True:
        try:
            ks = input("\n输入噪声序号（回车默认 babble）：").strip()
            kind = kinds[0] if not ks else kinds[int(ks) - 1]
            ss = input("输入 SNR（dB，越小越吵；回车默认 -5）：").strip()
            snr = -5.0 if not ss else float(ss)
            break
        except (ValueError, IndexError):
            print("  输入无效，请重新输入。")
    variants = build_variants(clean, fs, kind, snr, gtcrn, mlp)
    report_levels(variants, fs)
    print("\n将依次播放各版本（已做响度对齐，可直接对比）：")
    while True:
        play_sequence(variants, fs)
        if input("\n按回车再听一遍，输入 q 退出：").strip().lower() == "q":
            break


def main() -> int:
    ap = argparse.ArgumentParser(description="降噪效果 A/B 试听工具")
    ap.add_argument("--clean", default=audio_io.default_sample_path(), help="干净语音文件")
    ap.add_argument("--kind", default=None, help="噪声类型（默认交互选择）")
    ap.add_argument("--snr", type=float, default=None, help="输入 SNR（dB，默认交互选择）")
    ap.add_argument("--fs", type=int, default=16000)
    ap.add_argument("--seconds", type=float, default=4.0, help="试听片段长度（默认 4 秒）")
    ap.add_argument("--export", action="store_true", help="导出对比音频到 results/ab_demo/ 而不播放")
    ap.add_argument("--no-normalize", action="store_true",
                    help="不做响度对齐（用于体会不对齐时整体变小带来的听感误判）")
    args = ap.parse_args()

    y, sr = audio_io.load_audio(os.path.abspath(args.clean))
    clean = audio_io.to_mono(y)[: int(args.seconds * sr)]
    fs = int(sr)
    print(f"干净语音：{os.path.basename(args.clean)}，{len(clean) / fs:.2f} 秒，"
          f"峰值 {np.abs(clean).max():.3f}")
    if np.abs(clean).max() < 0.05:
        print("  [注意] 这段录音电平很低，建议换一段更响亮的语音，否则噪声会盖住人声。")

    gtcrn = GTCRNDenoiser()
    mlp = Denoiser()
    print(f"可用降噪方案：GTCRN={gtcrn.is_ready}，自训练 MLP={mlp.is_ready}")

    if args.export:
        os.makedirs(DEMO_DIR, exist_ok=True)
        for kind in ("babble", "car", "white"):
            for snr in (0.0, -5.0):
                variants = build_variants(clean, fs, kind, snr, gtcrn, mlp)
                for name, sig in variants:
                    tag = f"{kind}_{snr:+.0f}dB_{name[2:].split('（')[0].strip()}"
                    tag = tag.replace(" ", "").replace("→", "-")
                    audio_io.save_audio(os.path.join(DEMO_DIR, f"{tag}.wav"), sig, fs)
        print(f"\n已导出到 {DEMO_DIR}")
        print("  命名规则：<噪声>_<SNR>_<版本>.wav，按 ①→②→③… 顺序播放即可对比")
        print("  建议先用 babble_-05dB_* 这一组（嘈杂人声最难，差别最明显）")
        return 0

    if args.kind and args.snr is not None:
        variants = build_variants(clean, fs, args.kind, args.snr, gtcrn, mlp,
                                  normalize=not args.no_normalize)
        report_levels(variants, fs)
        print("\n依次播放：")
        play_sequence(variants, fs)
    else:
        interactive(fs, clean, gtcrn, mlp)
    return 0


if __name__ == "__main__":
    sys.exit(main())
