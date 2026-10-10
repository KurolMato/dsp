"""诊断：为什么同一段强噪声素材上，预训练 GTCRN 明显好于自训练模型。

针对用户提供的 `原始测试音频/加噪声.mp3`（44.1 kHz 立体声、纯白噪声、很吵）做三件事：
    1. 鉴定噪声类型与强度（频带平坦度 + 静音段/语音段比值 + 估计 SNR）；
    2. 两个模型在该素材上的无参考指标对比；
    3. 在**同一批干净语音**上加白噪声做 SNR 扫描，看差距是否随 SNR 变化。

用法：
    python diagnose_noisy.py                       # 默认分析 原始测试音频/加噪声.mp3
    python diagnose_noisy.py --file 别的音频.wav
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

from core import audio_io, data as data_mod, dsp, metrics      # noqa: E402
from core.enhance import Denoiser                              # noqa: E402
from core.gtcrn import GTCRNDenoiser                           # noqa: E402
from eval_compare import log_spectral_distance                 # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_FILE = os.path.join(os.path.dirname(HERE), "原始测试音频", "加噪声.mp3")


def quiet_ratio_db(x: np.ndarray, fs: int) -> float:
    fr = np.array_split(x, max(1, len(x) // (fs // 50)))
    e = np.sort(np.array([np.sum(f ** 2) for f in fr]))
    k = max(1, len(e) // 5)
    return float(10 * np.log10(max(e[-k:].mean(), 1e-20) / max(e[:k].mean(), 1e-20)))


def noise_shape(x: np.ndarray, fs: int) -> list[tuple[str, float]]:
    """用能量最低的 10% 帧近似噪声谱，给出各频带相对能量（鉴定白/粉/布朗噪声）。"""
    fl = fs // 50
    fr = np.array_split(x, max(1, len(x) // fl))
    order = np.argsort([np.sum(f ** 2) for f in fr])
    segs = [x[i * fl: i * fl + fl] for i in order[: max(1, len(order) // 10)]]
    nz = np.concatenate(segs)
    ns = np.abs(np.fft.rfft(nz * np.hanning(len(nz)))) ** 2
    freqs = np.fft.rfftfreq(len(nz), 1 / fs)
    out, base = [], None
    for lo, hi in ((100, 300), (300, 800), (800, 2000), (2000, 4000), (4000, 7000)):
        m = (freqs >= lo) & (freqs < hi)
        if not m.any():
            continue
        v = 10 * np.log10(max(ns[m].mean(), 1e-30))
        base = v if base is None else base
        out.append((f"{lo}-{hi} Hz", v - base))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="诊断强噪声素材上两个模型的差异")
    ap.add_argument("--file", default=DEFAULT_FILE)
    ap.add_argument("--fs", type=int, default=16000, help="SNR 扫描用的采样率")
    ap.add_argument("--clean", default=None, help="SNR 扫描用的干净语音（默认用 samples/demo_speech.wav）")
    args = ap.parse_args()

    g, m = GTCRNDenoiser(), Denoiser()
    print("模型：GTCRN=%s | 自训练=%s" % (g.is_ready, m.is_ready))

    # ---------- 1) 素材鉴定 ----------
    if os.path.isfile(args.file):
        y_raw, sr0 = audio_io.load_audio(args.file)
        y = audio_io.to_mono(y_raw) if np.asarray(y_raw).ndim > 1 else np.asarray(y_raw)
        n_ch = 1 if np.asarray(y_raw).ndim == 1 else np.asarray(y_raw).shape[1]
        print(f"\n=== 素材：{os.path.basename(args.file)} ===")
        print(f"  {sr0} Hz, {n_ch} 声道, {len(y) / sr0:.2f} 秒, 峰值 {np.abs(y).max():.4f}, "
              f"RMS {np.sqrt((y ** 2).mean()):.5f}")
        print(f"  静音段/语音段 = {quiet_ratio_db(y, sr0):+.1f} dB（越接近 0 越吵）")
        print("  噪声频带形状（以 100-300 Hz 为参考）：")
        for name, v in noise_shape(y, sr0):
            print(f"    {name:<14} {v:+6.1f} dB")
        print("    参考：纯白噪≈平坦；粉红噪≈每倍频程 −3 dB；布朗噪≈−6 dB")

        # ---------- 2) 该素材上的无参考对比 ----------
        print("\n=== 该素材上的表现（无参考指标）===")
        print("  %-14s %16s %14s %16s" % ("方法", "静音段底噪下降", "语音段保留", "自参考谱失真"))
        for tag, fn in (("GTCRN 预训练", lambda: g.enhance(y, sr0)),
                        ("自训练 MLP", lambda: m.enhance(y, sr0)),
                        ("串联", lambda: m.enhance_cascaded(y, g, fs=sr0))):
            try:
                out = np.asarray(fn(), dtype=np.float32)
                r = metrics.assess(y, out, sr0)
                print("  %-14s %13.1f dB %11.1f dB %13.1f dB"
                      % (tag, r["noise_floor_drop_db"], r["speech_keep_db"], r["lsd_self_db"]))
            except Exception as exc:
                print(f"  {tag:<14} 失败：{exc}")
    else:
        print(f"\n[跳过] 没找到素材文件 {args.file}")

    # ---------- 3) 同一干净语音上的 SNR 扫描 ----------
    fs = args.fs
    clean_path = args.clean or os.path.join(HERE, "samples", "demo_speech.wav")
    pool = data_mod.load_clean_pool([clean_path], fs=fs)
    clean = pool[0]
    print(f"\n=== SNR 扫描（干净语音 {os.path.basename(clean_path)} + 白噪声）===")
    print("  %-9s %13s %13s %13s" % ("输入SNR", "GTCRN", "自训练MLP", "串联"))
    for snr in (-5.0, 0.0, 5.0, 10.0, 15.0):
        nz = data_mod.make_noise("white", len(clean), fs, seed=11)
        noisy = data_mod.mix_at_snr(clean, nz, snr)
        cells = []
        for fn in (lambda: g.enhance(noisy, fs), lambda: m.enhance(noisy, fs=fs),
                   lambda: m.enhance_cascaded(noisy, g, fs=fs)):
            o = np.asarray(fn(), dtype=np.float32)
            n = min(len(o), len(clean))
            cells.append(dsp.mixing_snr(o[:n], clean[:n]) - snr)
        print("  %+6.0f dB %+10.2f dB %+10.2f dB %+10.2f dB" % (snr, *cells))
    print("\n说明：ΔSNR 越大越好；若差距随输入 SNR 变小而扩大，说明是训练 SNR 覆盖问题。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
