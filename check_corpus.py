"""查看训练会用到哪些语料与噪声（不训练，秒出结果）。

用途：点「训练降噪模型」之前先确认语料是否齐全——尤其是刚录的语音有没有被算进去。

    python check_corpus.py
"""

from __future__ import annotations

import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from core import audio_io, data as data_mod      # noqa: E402
from core.corpus import default_clean_paths

HERE = os.path.dirname(os.path.abspath(__file__))
RECORD_DIR = os.path.join(HERE, "recordings")
NOISE_DIR = os.path.join(HERE, "models", "noise_bank")


def find_training_files() -> list[str]:
    """与 app.py 中 DSPApp._find_training_files() 保持一致。"""
    files = default_clean_paths()
    for ext in ("*.flac", "*.mp3", "*.m4a", "*.ogg"):
        files.extend(sorted(glob.glob(os.path.join(RECORD_DIR, ext))))
    return list(dict.fromkeys(files))


def quiet_to_peak_db(y, sr: int) -> float:
    """静音段能量 / 语音段能量（dB）。

    这是判断"这段录音能不能当干净参考"的关键指标：越小越干净。
    实测参考值：干净的 MATLAB 原始语音约 −72 dB；用手机/耳麦在室内录的通常在 −25~−50 dB
    （自带环境噪声）。把底噪大的录音当"干净参考"去训练，等于教模型"把噪声当语音保留"。
    """
    import numpy as _np
    y = _np.asarray(y).reshape(-1)
    fl = max(1, sr // 50)
    n = max(1, len(y) // fl)
    e = _np.sort(_np.array([_np.sum(f ** 2) for f in _np.array_split(y, n)]))
    k = max(1, len(e) // 5)
    return float(10 * _np.log10(max(e[:k].mean(), 1e-20) / max(e[-k:].mean(), 1e-20)))


def voiced_ratio(y, sr: int) -> float:
    """有效样本占比（%）：接近 0 说明录到的是纯数字静音（麦克风被静音/禁用）。"""
    import numpy as _np
    y = _np.asarray(y).reshape(-1)
    if y.size == 0:
        return 0.0
    return float(_np.mean(_np.abs(y) > 1e-6) * 100.0)


def main() -> int:
    print("=== 干净语音（训练输入）===")
    files = find_training_files()
    total = 0.0
    clean_ish, noisy_ish, silent = [], [], []
    print(f"  {'时长':>8}  {'峰值':>6}  {'底噪/语音':>10}  {'有效样本':>8}  文件")
    for p in files:
        try:
            y, sr = audio_io.load_audio(p)
            dur = (y.shape[0] / float(sr))
            total += dur
            ym = audio_io.to_mono(y)
            yq = quiet_to_peak_db(ym, sr)
            vr = voiced_ratio(ym, sr)
            flags = []
            if yq > -40:
                noisy_ish.append((p, yq)); flags.append("底噪偏大")
            if vr < 30:
                silent.append((p, vr)); flags.append("近乎静音")
            if not flags:
                clean_ish.append(p)
            print(f"  {dur:7.2f}s  {float(abs(y).max()):6.3f}  {yq:9.1f} dB  {vr:7.1f}%  "
                  f"{os.path.relpath(p, HERE)}" + (f"  ← {'、'.join(flags)}" if flags else ""))
        except Exception as exc:
            print(f"  读取失败  {os.path.relpath(p, HERE)}：{exc}")
    print(f"  合计 {len(files)} 个文件 / {total:.1f} 秒")
    if silent:
        print(f"\n  [注意] {len(silent)} 个文件近乎静音（有效样本 < 30%），当训练输入无意义：")
        for p, v in silent:
            print(f"           {os.path.relpath(p, HERE)}  仅 {v:.1f}% 有效样本")
        print("         通常是麦克风被静音/禁用时误录；建议删除后重新录制。")
    if noisy_ish:
        print(f"\n  [注意] {len(noisy_ish)} 个文件底噪偏大（静音段/语音段 > −40 dB）：")
        for p, v in noisy_ish:
            print(f"           {os.path.relpath(p, HERE)}  {v:.1f} dB")
        print("         这类录音当『干净参考』会教模型保留噪声，建议在安静环境重录，"
              "或只把它们当作『带噪素材』使用")
    if total < 60:
        print(f"\n  [提示] 少于 60 秒，模型容易欠拟合；界面「① 录音」多录几段会自动归档累积")

    print("\n=== 噪声（训练时与语音混合）===")
    real = os.path.join(NOISE_DIR, "real")
    real_files = glob.glob(os.path.join(real, "**", "*.wav"), recursive=True) if os.path.isdir(real) else []
    if real_files:
        by_dir: dict[str, int] = {}
        for p in real_files:
            by_dir[os.path.relpath(os.path.dirname(p), real)] = by_dir.get(os.path.relpath(os.path.dirname(p), real), 0) + 1
        print(f"  真实噪声：{len(real_files)} 个文件，来自 {len(by_dir)} 组")
        for k, v in sorted(by_dir.items()):
            print(f"      {k:<16} {v} 个")
        print(f"  训练时约 30% 概率使用真实噪声，其余用合成噪声")
    else:
        print("  真实噪声：无（可用 python download_datasets.py 下载 DEMAND，或用界面「导入真实噪声…」）")
    print(f"  合成噪声：{len(data_mod.NOISE_TYPES)} 类 —— "
          + "、".join(data_mod.NOISE_LABELS[k] for k in data_mod.NOISE_TYPES))
    synthetics = glob.glob(os.path.join(NOISE_DIR, "noise_*.wav"))
    if synthetics:
        print(f"  （已导出的合成噪声 wav：{len(synthetics)} 个，可直接试听）")

    print("\n=== 训练时实际会执行 ===")
    print("  python train_model.py --clean " + " ".join(
        f'"{p}"' for p in files[:3]) + (" ..." if len(files) > 3 else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
