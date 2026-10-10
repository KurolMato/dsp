"""噪声库试听：把 noise_bank 里的噪声逐个放出来听，判断"这到底是不是噪声"。

为什么需要它：DEMAND 的真实录音是为"加噪训练"准备的，原始电平很低
（实测 RMS 只有 0.002~0.07），直接播放会又轻又闷，容易让人以为"文件坏了/不是噪声"。
本脚本会**自动增益到合适响度**再播放，并打印每个文件的真实电平和所用增益。

用法：
    python listen_noise.py                    # 依次试听 8 个合成噪声 + 10 个场景各 1 通道
    python listen_noise.py --real-only        # 只试听真实噪声（DEMAND）
    python listen_noise.py --synth-only       # 只试听合成噪声
    python listen_noise.py --channels 3       # 每个场景试听 3 个通道（听通道差异）
    python listen_noise.py --export out_dir   # 不播放，导出增益后的 wav 到目录
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

from core import audio_io, data as data_mod      # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
BANK = os.path.join(HERE, "models", "noise_bank")
REAL = os.path.join(BANK, "real")


def gain_to(x: np.ndarray, target_rms_db: float = -20.0, peak_limit: float = 0.95):
    """把音频增益到目标 RMS（避免削顶），返回 (增益后的音频, 增益 dB)。"""
    x = np.asarray(x, dtype=np.float32)
    rms = float(np.sqrt(max(np.mean(x ** 2), 1e-20)))
    want = 10 ** (target_rms_db / 20.0)
    g = want / rms
    peak = float(np.max(np.abs(x))) * g
    if peak > peak_limit:                      # 削顶保护
        g = peak_limit / max(float(np.max(np.abs(x))), 1e-20)
    return (x * g).astype(np.float32), 20 * np.log10(max(g, 1e-12))


def describe(path: str, name: str, export_dir: str | None, play: bool) -> None:
    y_raw, sr = audio_io.load_audio(path)
    y = audio_io.to_mono(y_raw)
    n_ch = 1 if np.asarray(y_raw).ndim == 1 else np.asarray(y_raw).shape[1]
    y_g, g_db = gain_to(y)
    # 简单特征，帮助判断"听起来是什么"
    seg = y_g[: min(len(y_g), sr * 5)]
    S = np.abs(np.fft.rfft(seg * np.hanning(len(seg)))) ** 2
    fr = np.fft.rfftfreq(len(seg), 1 / sr)
    tot = max(S.sum(), 1e-30)
    low = S[fr < 300].sum() / tot
    high = S[fr > 3000].sum() / tot
    zcr = float(np.mean(np.abs(np.diff(np.sign(seg)))) / 2)
    print(f"  {name:<26} {sr} Hz {n_ch}ch {len(y) / sr:5.2f}s  "
          f"原始RMS {np.sqrt(np.mean(y ** 2)):.5f}  增益 {g_db:+5.1f} dB  "
          f"低频{low * 100:4.0f}% 高频{high * 100:4.0f}% 过零{zcr:.3f}")
    if export_dir:
        os.makedirs(export_dir, exist_ok=True)
        # 序号 + 净化后的名字，避免中文括号/上标把文件名搞坏
        safe = "".join(c for c in name.split("（")[0] if c.isalnum() or c in "_-")
        safe = safe or "noise"
        out = os.path.join(export_dir, f"{safe}.wav")
        audio_io.save_audio(out, y_g, sr)
    if play:
        audio_io.play(y_g, sr, blocking=True)
        time.sleep(0.3)


def main() -> int:
    ap = argparse.ArgumentParser(description="噪声库试听（自动增益到合适响度）")
    ap.add_argument("--real-only", action="store_true")
    ap.add_argument("--synth-only", action="store_true")
    ap.add_argument("--channels", type=int, default=1, help="每个真实场景试听几个通道")
    ap.add_argument("--export", default=None, help="导出到目录而不播放")
    ap.add_argument("--seconds", type=float, default=5.0, help="每个文件试听时长")
    ap.add_argument("--target-rms-db", type=float, default=-20.0, help="目标响度")
    args = ap.parse_args()
    play = args.export is None

    print("=" * 78)
    print("  噪声库试听：所有音频都会自动增益到同一响度后播放")
    print("  （原始 DEMAND 录音电平极低，直接播会又轻又闷，这是正常的）")
    print("=" * 78)

    if not args.real_only:
        print("\n=== A) 程序化合成噪声（本程序自己生成的，共 8 类）===")
        for fn in sorted(f for f in os.listdir(BANK)
                         if f.startswith("noise_") and f.endswith(".wav")):
            kind = fn[len("noise_"):-4]
            label = data_mod.NOISE_LABELS.get(kind, kind)
            p = os.path.join(BANK, fn)
            y, sr = audio_io.load_audio(p)
            y = audio_io.to_mono(y)[: int(args.seconds * sr)]
            tmp = os.path.join(BANK, "_tmp_listen.wav")
            audio_io.save_audio(tmp, y, sr)
            describe(tmp, f"{kind}（{label}）", args.export, play)
            os.remove(tmp)

    if not args.synth_only and os.path.isdir(REAL):
        scenes = sorted(d for d in os.listdir(REAL) if os.path.isdir(os.path.join(REAL, d)))
        print(f"\n=== B) 真实环境录音（DEMAND 数据集，{len(scenes)} 个场景）===")
        print("  每个场景取前 N 个通道试听——同一场景不同通道是**同一段声音的不同麦克风**，")
        print("  听起来会很像（这是多通道阵列录音的正常现象），但文件内容不同。")
        for sc in scenes:
            sd = os.path.join(REAL, sc)
            for i, fn in enumerate(sorted(os.listdir(sd))[: max(1, args.channels)]):
                p = os.path.join(sd, fn)
                y, sr = audio_io.load_audio(p)
                y = audio_io.to_mono(y)[: int(args.seconds * sr)]
                tmp = os.path.join(BANK, "_tmp_real.wav")
                audio_io.save_audio(tmp, y, sr)
                describe(tmp, f"{sc}/{fn}" if args.channels > 1 else sc, args.export, play)
                os.remove(tmp)

    if args.export:
        print(f"\n已导出到 {os.path.abspath(args.export)}（已增益到统一响度，可直接试听对比）")
    else:
        print("\n试听完毕。若想导出成文件慢慢听：python listen_noise.py --export results/noise_demo")
    return 0


if __name__ == "__main__":
    sys.exit(main())
