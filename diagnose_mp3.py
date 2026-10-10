"""对照实验：MP3 有损编码会不会让自训练模型的降噪失效？

背景：用户提供的强噪声素材是 **44.1 kHz 立体声 MP3**，而模型只用 **16 kHz 无损 WAV** 训练过。
MP3 会带来两类训练时没见过的差异：
    1. **有损压缩痕迹**：频谱上的整形噪声、高频截断、预回声；
    2. **重采样链**：44.1 kHz → 16 kHz 的多相重采样（训练数据都是原生 16 kHz）。

本脚本对**同一条带噪信号**做三种处理，用同一个模型对比，从而把"格式/采样率"因素单独隔离：
    A. 原生 16 kHz 无损（训练分布内）
    B. 16 kHz → 编码为 44.1 kHz 立体声 MP3 → 解码回 16 kHz（模拟用户的文件）
    C. 仅重采样 16k→44.1k→16k，不经过 MP3（隔离"重采样"因素）

判读：若 A 明显好于 B，说明"格式/编码"是失效主因；若 B≈C，说明 MP3 编码影响不大、
      主要是重采样或 SNR 过低所致。

用法：
    python diagnose_mp3.py
"""

from __future__ import annotations

import os
import sys
import tempfile

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


def to_mp3_roundtrip(y: np.ndarray, fs: int, bitrate: str = "128k") -> np.ndarray:
    """把音频编码成 MP3 再解码回来（依赖 ffmpeg；不可用时返回 None）。"""
    import shutil
    import subprocess

    ff = shutil.which("ffmpeg")
    if not ff:
        return None
    with tempfile.TemporaryDirectory() as td:
        wav = os.path.join(td, "a.wav")
        mp3 = os.path.join(td, "a.mp3")
        audio_io.save_audio(wav, y, fs)
        try:
            subprocess.run([ff, "-y", "-loglevel", "error", "-i", wav,
                            "-b:a", bitrate, mp3], check=True)
            back, sr = audio_io.load_audio(mp3)
        except Exception:
            return None
    back = audio_io.to_mono(back)
    if sr != fs:
        from scipy import signal as sps
        back = sps.resample_poly(back, fs, sr)
    n = min(len(back), len(y))
    return np.asarray(back[:n], dtype=np.float32)


def resample_roundtrip(y: np.ndarray, fs: int, mid: int = 44100) -> np.ndarray:
    """只做 16k -> 44.1k -> 16k 的重采样往返（不经过编码）。"""
    from scipy import signal as sps
    up = sps.resample_poly(y.astype(np.float64), mid, fs)
    down = sps.resample_poly(up, fs, mid)
    return down[: len(y)].astype(np.float32)


def main() -> int:
    fs = 16000
    clean = data_mod.load_clean_pool([os.path.join(HERE, "samples", "demo_speech.wav")], fs=fs)[0]
    m, g = Denoiser(), GTCRNDenoiser()

    print("=== 对照实验：格式/采样率对自训练模型的影响 ===")
    print("（同一段带噪信号，三种处理方式，模型完全相同）\n")
    print("  %-6s %-34s %12s %12s" % ("SNR", "处理方式", "自训练MLP", "GTCRN"))

    for snr in (-5.0, 0.0, 5.0):
        nz = data_mod.make_noise("white", len(clean), fs, seed=11)
        noisy = data_mod.mix_at_snr(clean, nz, snr)

        variants = [("A 原生 16 kHz 无损（训练分布内）", noisy),
                    ("C 仅重采样 16k→44.1k→16k", resample_roundtrip(noisy, fs))]
        mp3v = to_mp3_roundtrip(noisy, fs)
        if mp3v is not None:
            variants.insert(1, ("B MP3 往返（44.1k 立体声 128k）", mp3v))

        for tag, sig in variants:
            row = []
            for fn in (lambda s=sig: m.enhance(s, fs=fs), lambda s=sig: g.enhance(s, fs)):
                o = np.asarray(fn(), dtype=np.float32)
                n = min(len(o), len(clean))
                row.append(dsp.mixing_snr(o[:n], clean[:n]) - snr)
            print("  %+5.0f  %-34s %+9.2f dB %+9.2f dB" % (snr, tag, row[0], row[1]))
        print()

    if to_mp3_roundtrip(clean, fs) is None:
        print("[注意] 没找到 ffmpeg，已跳过 MP3 对照（只比较了重采样）。")
        print("       安装 ffmpeg 后可看到 MP3 编码的单独影响。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
