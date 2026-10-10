"""音频体检：检查素材里的直流偏置与次声成分（<20 Hz），并可选做高通预处理。

为什么需要
----------
实测发现 DEMAND 真实噪声录音里有大量**次声成分**（<20 Hz 占总能量 10%~98%）：
    DEMAND_DWASHING  <20 Hz 占 98.0%   300 Hz~3 kHz 仅占 0.2%
    DEMAND_OOFFICE   <20 Hz 占 80.1%   300 Hz~3 kHz 仅占 0.3%
这些成分人耳听不见、也不影响语音，但会**扭曲"按总能量计的 SNR"**：
    含次声时，"5 dB SNR" 意味着人声频段的实际干扰远小于标称值（任务偏易）。
    滤掉次声后，同样的 5 dB SNR 才真正对应人声频段的强干扰（任务变难）。

实测影响（真实 DEMAND 噪声 @5 dB，同一段干净语音）：
    噪声未滤波：自训练 +8.05 dB | GTCRN +0.02 dB
    高通 60 Hz：自训练 +4.06 dB | GTCRN +3.80 dB
    → 结论：**不能把含次声的数字与滤净后的数字混着比较**，报告里必须注明是否做过高通。

用法
----
    python check_audio_health.py                          # 体检噪声库与语料
    python check_audio_health.py --file 某音频.wav         # 体检单个文件
    python check_audio_health.py --fix-dir 输出目录        # 把噪声库高通滤波后另存一份
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

from core import audio_io      # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REAL = os.path.join(HERE, "models", "noise_bank", "real")


def band_shares(y: np.ndarray, sr: int, secs: float = 20.0) -> dict:
    """各频带能量占比（%）：<20 / 20-100 / 100-300 / 300-3000 / >3000 Hz。"""
    seg = np.asarray(y, dtype=np.float64)[: int(sr * secs)]
    if len(seg) < 1024:
        return {}
    S = np.abs(np.fft.rfft(seg * np.hanning(len(seg)))) ** 2
    fr = np.fft.rfftfreq(len(seg), 1 / sr)
    tot = max(S.sum(), 1e-30)
    edges = [(0, 20, "<20Hz"), (20, 100, "20-100"), (100, 300, "100-300"),
             (300, 3000, "300-3k"), (3000, 8000, ">3k")]
    return {name: float(S[(fr >= lo) & (fr < hi)].sum() / tot * 100) for lo, hi, name in edges}


def report(path: str, name: str = "") -> dict:
    y_raw, sr = audio_io.load_audio(path)
    y = audio_io.to_mono(y_raw)
    b = band_shares(y, sr)
    dc = float(np.mean(y)) if len(y) else 0.0
    print(f"  {name or os.path.basename(path):<26} DC {dc:+.2e} | "
          + " ".join(f"{k} {v:5.1f}%" for k, v in b.items()))
    return b


def main() -> int:
    ap = argparse.ArgumentParser(description="音频体检：直流与次声成分")
    ap.add_argument("--file", default=None, help="只体检单个文件")
    ap.add_argument("--fix-dir", default=None, help="把噪声库高通滤波后另存到该目录")
    ap.add_argument("--hp", type=float, default=60.0, help="高通截止频率（Hz），默认 60")
    args = ap.parse_args()

    if args.file:
        print("=== 单文件体检 ===")
        report(args.file)
        return 0

    print("=== 1) 噪声库（真实 DEMAND 录音）体检 ===")
    if os.path.isdir(REAL):
        shares = []
        for sc in sorted(d for d in os.listdir(REAL) if os.path.isdir(os.path.join(REAL, d))):
            b = report(os.path.join(REAL, sc, "ch01.wav"), sc)
            if b:
                shares.append(b)
        if shares:
            sub = np.mean([s.get("<20Hz", 0) for s in shares])
            speech = np.mean([s.get("300-3k", 0) for s in shares])
            print(f"\n  平均：<20 Hz 占 {sub:.1f}%，语音关键带 300 Hz~3 kHz 占 {speech:.1f}%")
            if sub > 10:
                print("  [注意] 次声占比偏高：按总能量计 SNR 会让『人声频段实际干扰』小于标称值，")
                print("         与不含次声的数据集（如合成噪声）**不可直接比较**。")
    else:
        print("  （没有噪声库）")

    print("\n=== 2) 训练语料体检（看有没有直流漂移）===")
    for d, pats in ((HERE, ("data.wav",)), (os.path.join(HERE, "samples"), None)):
        if not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            if fn.lower().endswith(".wav") and (pats is None or fn in pats):
                report(os.path.join(d, fn), os.path.relpath(os.path.join(d, fn), HERE))

    if args.fix_dir:
        from scipy import signal as sps
        print(f"\n=== 3) 高通滤波（{args.hp:.0f} Hz）另存到 {args.fix_dir} ===")
        b, a = sps.butter(4, args.hp / (8000.0), btype="high")     # 按 16 kHz 设计
        n = 0
        for sc in sorted(os.listdir(REAL)) if os.path.isdir(REAL) else []:
            sd = os.path.join(REAL, sc)
            if not os.path.isdir(sd):
                continue
            out_dir = os.path.join(args.fix_dir, sc)
            os.makedirs(out_dir, exist_ok=True)
            for fn in sorted(os.listdir(sd)):
                y, sr = audio_io.load_audio(os.path.join(sd, fn))
                y = audio_io.to_mono(y)
                if sr != 16000:
                    y = sps.resample_poly(y.astype(np.float64), 16000, sr)
                yf = sps.filtfilt(b, a, np.asarray(y, dtype=np.float64)).astype(np.float32)
                audio_io.save_audio(os.path.join(out_dir, fn), yf, 16000)
                n += 1
        print(f"  已输出 {n} 个文件（该目录可直接用 --noise-dir 训练）")
        print("  提示：用它训练得到的模型指标，与用原始含次声噪声训练的模型**不可直接比较**。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
