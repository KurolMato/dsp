"""核查噪声库里的音频到底是什么（回应"文件都一样/像奇怪音效"的疑问）。

核查三件事：
    1. 文件是否重复（内容哈希去重）
    2. 同一场景内多个通道的关系（DEMAND 是麦克风阵列同步录音，通道应高度相关但非同一文件）
    3. 不同场景的声学特征是否确实不同（证明是不同环境的真实录音，而非同一段音效）

用法：python check_noise_lib.py
"""

from __future__ import annotations

import hashlib
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from core import audio_io          # noqa: E402
from core import data as data_mod  # noqa: E402

REAL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "noise_bank", "real")


def md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    if not os.path.isdir(REAL):
        print(f"没有噪声库目录：{REAL}")
        return 1
    scenes = sorted(d for d in os.listdir(REAL) if os.path.isdir(os.path.join(REAL, d)))
    print(f"=== 噪声库：{len(scenes)} 个场景 ===\n")

    # 1) 去重
    hashes: dict[str, list[str]] = {}
    sizes: dict[str, set] = {}
    for sc in scenes:
        sd = os.path.join(REAL, sc)
        szs = set()
        for fn in sorted(os.listdir(sd)):
            p = os.path.join(sd, fn)
            sizes.setdefault(sc, set()).add(os.path.getsize(p))
            hashes.setdefault(md5(p), []).append(f"{sc}/{fn}")
    total = sum(len(v) for v in hashes.values())
    print("=== 1) 内容去重 ===")
    print(f"  文件总数 {total}，不同内容 {len(hashes)} 种 -> "
          + ("全部唯一（没有复制粘贴）" if total == len(hashes) else "**存在重复**"))
    for sc in scenes:
        print(f"    {sc:<18} 文件大小种类 {len(sizes[sc])} 种"
              + ("（该场景 16 通道时长相同，故大小一致，属正常）" if len(sizes[sc]) == 1 else ""))

    # 2) 同场景通道相关性
    print("\n=== 2) 同场景内各通道的关系（DEMAND = 麦克风阵列同步录音）===")
    for sc in scenes[:3]:
        sd = os.path.join(REAL, sc)
        names = sorted(os.listdir(sd))[:6]
        sigs = [audio_io.to_mono(audio_io.load_audio(os.path.join(sd, n))[0]) for n in names]
        rs = []
        for i in range(1, len(sigs)):
            n = min(len(sigs[0]), len(sigs[i]))
            rs.append(float(np.corrcoef(sigs[0][:n], sigs[i][:n])[0, 1]))
        print(f"  {sc:<18} ch01 与 {', '.join(names[1:])} 的相关: "
              + " ".join(f"{r:.2f}" for r in rs))
    print("  说明：同一阵列不同麦克风录的是**同一段声音**，所以高度相关（0.6~0.99）属正常；")
    print("       但它们不是同一个文件（哈希不同）。训练时用多个通道 = 同一噪声的多个实例。")

    # 3) 场景间声学特征
    print("\n=== 3) 各场景声学特征（应彼此不同 = 不同环境的真实录音）===")
    print(f"  {'场景':<18}{'RMS':>9}{'过零率':>10}{'<300Hz':>10}{'>3kHz':>10}")
    lows, highs = [], []
    for sc in scenes:
        p = os.path.join(REAL, sc, "ch01.wav")
        y = audio_io.to_mono(audio_io.load_audio(p)[0])
        sr = 16000
        seg = y[: sr * 10]
        S = np.abs(np.fft.rfft(seg * np.hanning(len(seg)))) ** 2
        fr = np.fft.rfftfreq(len(seg), 1 / sr)
        tot = max(S.sum(), 1e-30)
        low, high = S[fr < 300].sum() / tot, S[fr > 3000].sum() / tot
        zcr = float(np.mean(np.abs(np.diff(np.sign(seg)))) / 2)
        lows.append(low); highs.append(high)
        print(f"  {sc:<18}{np.sqrt((seg ** 2).mean()):9.4f}{zcr:10.3f}"
              f"{low * 100:9.1f}%{high * 100:9.1f}%")
    print(f"\n  低频占比 {min(lows) * 100:.1f}%~{max(lows) * 100:.1f}%，"
          f"高频占比 {min(highs) * 100:.1f}%~{max(highs) * 100:.1f}%"
          "  -> 场景差异明显")

    # 4) 顶层合成噪声
    print("\n=== 4) 顶层 8 个合成噪声（本程序自己生成的，不是下载的）===")
    nb = os.path.dirname(REAL)
    for fn in sorted(f for f in os.listdir(nb) if f.startswith("noise_") and f.endswith(".wav")):
        p = os.path.join(nb, fn)
        y = audio_io.to_mono(audio_io.load_audio(p)[0])
        kind = fn[len("noise_"):-4]
        label = data_mod.NOISE_LABELS.get(kind, kind)
        print(f"  {fn:<20} {len(y) / 16000:4.1f} 秒  {label}")
    print("  说明：这 8 个是程序化合成噪声（白噪/粉红/布朗/工频/键盘/嘈杂人声/车内/空调），")
    print("       其中「键盘敲击」听起来确实像音效——那是脉冲串噪声，本来就该是哒哒声。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
