"""训练语料批量导入工具：把散落各处的音频整理进项目，直接可用于重训。

**图形界面里也有对应功能**（左侧「训练数据」面板的「批量导入文件夹…」），本脚本
是同一套逻辑的命令行入口，适合批量/脚本化场景。

为什么需要它
------------
界面上的「导入干净语音…」一次只能选若干文件，几百个素材（下载的语音库、手机批量
录音、别人给的素材包）就不够用。而外部素材常见三个问题：采样率五花八门、有重复文件、
片段太短或电平过低。本工具一次处理掉。

它会做什么
----------
    · 从文件 / 整个文件夹（可递归）/ zip 压缩包 收集音频
    · **跳过重复**：用"时长 + 频谱形状"近邻匹配（跨格式/采样率也认得出同一素材）
    · 统一重采样到 16 kHz、转单声道、峰值归一化（可选）
    · 只保留长度在 [--min-sec, --max-sec] 的片段
    · 复制（默认，保留原件）或移动（--move）到 recordings/
    · 导入干净语音时**自动质检**，底噪偏大的会给出警告

用法示例
--------
    python corpus_tool.py import --from D:\\语音素材 --recursive --dry-run   # 预演
    python corpus_tool.py import --from D:\\语音素材 --recursive             # 导入
    python corpus_tool.py import --from 语音素材.zip                         # 压缩包
    python corpus_tool.py import --from D:\\DEMAND --recursive --as-noise    # 导噪声
    python corpus_tool.py audit                                             # 体检语料
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
import tempfile
import time
import zipfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from core import audio_io      # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RECORD_DIR = os.path.join(HERE, "recordings")                    # 干净语音语料
NOISE_DIR = os.path.join(HERE, "models", "noise_bank", "real")    # 真实噪声
AUDIO_EXTS = (".wav", ".flac", ".mp3", ".m4a", ".ogg", ".aiff", ".aif")
BAND_EDGES = np.array([0, 150, 300, 600, 1200, 2400, 4000, 6000, 8001])


# ----------------------------------------------------------------------------
# 基础工具
# ----------------------------------------------------------------------------
def md5_of(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def collect_from_dir(root: str, recursive: bool) -> list[str]:
    out: list[str] = []
    if recursive:
        for dirpath, _dirs, files in os.walk(root):
            out += [os.path.join(dirpath, f) for f in files
                    if os.path.splitext(f)[1].lower() in AUDIO_EXTS]
    else:
        out += [os.path.join(root, f) for f in os.listdir(root)
                if os.path.splitext(f)[1].lower() in AUDIO_EXTS]
    return sorted(out)


def collect_from_zip(path: str, workdir: str) -> list[str]:
    out: list[str] = []
    with zipfile.ZipFile(path) as z:
        for name in z.namelist():
            if os.path.splitext(name)[1].lower() in AUDIO_EXTS:
                dst = os.path.join(workdir, os.path.basename(name))
                with z.open(name) as src, open(dst, "wb") as f:
                    shutil.copyfileobj(src, f)
                out.append(dst)
    return sorted(out)


def resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return np.asarray(x, dtype=np.float32)
    from scipy import signal as sps
    return sps.resample_poly(np.asarray(x, dtype=np.float64), sr_out, sr_in).astype(np.float32)


def peak_normalize(x: np.ndarray, peak: float = 0.95) -> np.ndarray:
    m = float(np.max(np.abs(x))) if x.size else 0.0
    return (x / m * peak).astype(np.float32) if m > 1e-9 else x


# ----------------------------------------------------------------------------
# 质量指标
# ----------------------------------------------------------------------------
def quiet_ratio_db(y: np.ndarray, sr: int) -> float:
    """静音段能量 / 语音段能量（dB）：越小越干净，> −40 dB 视为底噪偏大。"""
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    fl = max(1, sr // 50)
    fr = np.array_split(y, max(1, len(y) // fl))
    e = np.sort(np.array([np.sum(f ** 2) for f in fr]))
    k = max(1, len(e) // 5)
    return float(10 * np.log10(max(e[:k].mean(), 1e-20) / max(e[-k:].mean(), 1e-20)))


def voiced_ratio(y: np.ndarray) -> float:
    """有效样本占比（%）：接近 0 = 纯数字静音（麦克风被静音时误录）。"""
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    return float(np.mean(np.abs(y) > 1e-6) * 100.0) if y.size else 0.0


def prepare_clean_list(paths: list[str], max_quiet_db: float = -40.0,
                       min_voiced_pct: float = 30.0, verbose: bool = True) -> list[str]:
    """训练前筛掉不适合当"干净参考"的素材，返回可用文件列表。

    两条硬规则（都有实测依据）：
        1. **近乎静音**（有效样本 < 30%）—— 麦克风被静音/禁用时误录，当训练输入无意义；
        2. **底噪偏大**（静音段/语音段 > −40 dB）—— 当"干净参考"会教模型保留噪声。
    实测例子：`manba.wav` 底噪 −28.1 dB；有一条录音仅 3.4% 有效样本。
    """
    usable, dropped = [], []
    for p in paths:
        try:
            y_raw, sr = audio_io.load_audio(p)
            y = audio_io.to_mono(y_raw)
            vr = voiced_ratio(y)
            q = quiet_ratio_db(y, sr)
            if vr < min_voiced_pct:
                dropped.append((p, f"近乎静音（有效样本 {vr:.1f}%）"))
            elif q > max_quiet_db:
                dropped.append((p, f"底噪偏大（{q:.1f} dB）"))
            else:
                usable.append(p)
        except Exception as exc:
            dropped.append((p, f"读取失败（{exc}）"))
    if verbose:
        for p, why in dropped:
            print(f"      [跳过] {os.path.basename(p)}：{why}")
        if dropped:
            print(f"      （共跳过 {len(dropped)} 个素材；"
                  "用 python check_corpus.py 可查看全部素材的体检结果）")
    return usable


# ----------------------------------------------------------------------------
# 去重：时长 + 频谱形状近邻匹配
# ----------------------------------------------------------------------------
def audio_feature(y: np.ndarray, sr: int = 16000) -> tuple[int, np.ndarray]:
    """音频指纹：返回 (样本数@16kHz, 8 维频带形状)。

    形状 = 各频带能量 dB 减去全局均值（**增益无关**），再按帧平均。
    用于判断"这条素材是否已在库里"。
    """
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    if sr != 16000:
        y = resample(y, sr, 16000)
        sr = 16000
    if y.size < 512:
        return len(y), np.zeros(8, dtype=np.float32)
    win, hop = 512, 256
    n = 1 + (len(y) - win) // hop
    idx = np.arange(win)[None, :] + hop * np.arange(n)[:, None]
    spec = np.abs(np.fft.rfft(y[idx] * np.hanning(win)[None, :], axis=1)) ** 2
    freqs = np.fft.rfftfreq(win, 1 / sr)
    bands = np.array([spec[:, (freqs >= lo) & (freqs < hi)].mean(axis=1)
                      for lo, hi in zip(BAND_EDGES[:-1], BAND_EDGES[1:])])
    db = 10.0 * np.log10(np.maximum(bands, 1e-12))
    shape = (db - db.mean(axis=0, keepdims=True)).mean(axis=1)
    return len(y), shape.astype(np.float32)


def is_duplicate(feat: tuple[int, np.ndarray],
                 existing: list[tuple[int, np.ndarray]],
                 len_tol: int = 320, shape_tol: float = 0.35) -> bool:
    """时长接近（±20 ms）且频带形状相似 -> 视为重复。

    为什么不用哈希：入库会做峰值归一化 + int16 量化，浮点末位抖动会让"分带 dB 值"
    跨过量化台阶，哈希随之完全改变（实测同一素材两次导入得到不同哈希）。
    """
    for n_exist, shape_exist in existing:
        if abs(n_exist - feat[0]) <= len_tol and float(
                np.linalg.norm(shape_exist - feat[1])) < shape_tol:
            return True
    return False


def existing_features(directory: str) -> list[tuple[int, np.ndarray]]:
    feats: list[tuple[int, np.ndarray]] = []
    if not os.path.isdir(directory):
        return feats
    for dirpath, _d, files in os.walk(directory):
        for f in files:
            if os.path.splitext(f)[1].lower() in AUDIO_EXTS:
                try:
                    y_raw, sr = audio_io.load_audio(os.path.join(dirpath, f))
                    feats.append(audio_feature(audio_io.to_mono(y_raw), sr))
                except Exception:
                    pass
    return feats


# ----------------------------------------------------------------------------
# 导入核心（GUI 与命令行共用）
# ----------------------------------------------------------------------------
def import_corpus(source: str, as_noise: bool = False, recursive: bool = False,
                  move: bool = False, normalize: bool = True, skip_duplicates: bool = True,
                  min_sec: float = 0.3, max_sec: float = 30.0, dry_run: bool = False,
                  verbose: bool = False, progress=None) -> dict:
    """把音频批量导入训练语料（或噪声库），返回统计字典。

    progress(done, total, 当前文件名, 统计) 会被逐文件调用（GUI 用来刷新进度）。
    """
    src = os.path.abspath(source)
    base = {"n_ok": 0, "n_dup": 0, "n_skip": 0, "n_fail": 0, "seconds": 0.0,
            "manifest": [], "dropped": [], "quality_warnings": [], "n_candidates": 0,
            "dry_run": dry_run, "preview": [], "error": None,
            "target": NOISE_DIR if as_noise else RECORD_DIR}
    if not os.path.exists(src):
        return {**base, "error": f"找不到输入：{src}"}

    dst_root = base["target"]
    tmpdir = None
    if os.path.isdir(src):
        files = collect_from_dir(src, recursive)
    elif src.lower().endswith(".zip"):
        tmpdir = tempfile.mkdtemp(prefix="corpus_zip_")
        files = collect_from_zip(src, tmpdir)
    elif os.path.splitext(src)[1].lower() in AUDIO_EXTS:
        files = [src]
    else:
        return {**base, "error": f"不支持的输入类型：{src}"}

    base["n_candidates"] = len(files)
    if not files:
        if tmpdir:
            shutil.rmtree(tmpdir, ignore_errors=True)
        return {**base, "error": f"没有找到音频文件（支持 {', '.join(AUDIO_EXTS)}）"}
    if dry_run:
        if tmpdir:
            shutil.rmtree(tmpdir, ignore_errors=True)
        return {**base, "preview": files[:10]}

    os.makedirs(dst_root, exist_ok=True)
    seen = existing_features(dst_root) if skip_duplicates else []
    n_ok = n_dup = n_skip = n_fail = 0
    total_sec = 0.0
    stamp = time.strftime("%Y%m%d_%H%M%S")
    manifest: list[tuple[str, float, int]] = []
    dropped: list[tuple[str, str]] = []

    for i, p in enumerate(files, 1):
        try:
            y_raw, sr = audio_io.load_audio(p)
            y = audio_io.to_mono(y_raw)
            if sr != 16000:
                y = resample(y, sr, 16000)
            sr = 16000
            if normalize:
                y = peak_normalize(y)
            if skip_duplicates:
                feat = audio_feature(y, sr)
                if is_duplicate(feat, seen):
                    n_dup += 1
                    if progress:
                        progress(i, len(files), os.path.basename(p),
                                 {"n_ok": n_ok, "n_dup": n_dup, "n_skip": n_skip, "n_fail": n_fail})
                    continue
            dur = len(y) / sr
            if dur < min_sec or dur > max_sec:
                n_skip += 1
                dropped.append((p, f"时长 {dur:.2f}s 不在 {min_sec}~{max_sec}s 内"))
                continue

            stem = os.path.splitext(os.path.basename(p))[0]
            safe = "".join(c for c in stem if c.isalnum() or c in "_-")[:40] or "clip"
            if as_noise:                       # 噪声按来源目录分组
                group = os.path.basename(os.path.dirname(p)) or "imported"
                safegroup = "".join(c for c in group if c.isalnum() or c in "_-")[:24] or "imported"
                out_dir = os.path.join(dst_root, f"IMPORTED_{safegroup}")
            else:
                out_dir = dst_root
            os.makedirs(out_dir, exist_ok=True)
            out = os.path.join(out_dir, f"bulk_{stamp}_{i:04d}_{safe}.wav")
            audio_io.save_audio(out, y, sr)
            if skip_duplicates:
                seen.append(feat)              # 同一次导入内的新文件也参与后续去重
            if move and os.path.abspath(p).startswith(src) and src != p:
                try:
                    os.remove(p)
                except Exception:
                    pass
            n_ok += 1
            total_sec += dur
            manifest.append((os.path.relpath(out, HERE), dur, sr))
        except Exception as exc:
            n_fail += 1
            if verbose:
                dropped.append((p, f"读取失败（{exc}）"))
        if progress:
            progress(i, len(files), os.path.basename(p),
                     {"n_ok": n_ok, "n_dup": n_dup, "n_skip": n_skip, "n_fail": n_fail})

    if tmpdir:
        shutil.rmtree(tmpdir, ignore_errors=True)

    # 导入干净语音时自动质检：底噪大的文件当"干净参考"会教模型保留噪声
    warnings: list[str] = []
    if not as_noise and manifest:
        bad: list[tuple[str, float]] = []
        for rel, _dur, _sr in manifest:
            try:
                y_raw, sr = audio_io.load_audio(os.path.join(HERE, rel))
                q = quiet_ratio_db(audio_io.to_mono(y_raw), sr)
                if q > -40.0:
                    bad.append((rel, q))
            except Exception:
                pass
        if bad:
            warnings.append(f"有 {len(bad)} 个导入文件底噪偏大（> −40 dB），"
                            "当『干净参考』会让模型学会保留噪声：")
            warnings += [f"    {q:5.1f} dB  {rel}" for rel, q in bad[:8]]
            warnings.append("  建议删除它们，或改导进噪声库（--as-noise）")

    return {"n_ok": n_ok, "n_dup": n_dup, "n_skip": n_skip, "n_fail": n_fail,
            "seconds": total_sec, "manifest": manifest, "dropped": dropped,
            "quality_warnings": warnings, "n_candidates": len(files),
            "dry_run": False, "preview": [], "target": dst_root, "error": None}


# ----------------------------------------------------------------------------
# 命令行入口
# ----------------------------------------------------------------------------
def do_import(args) -> int:
    src = os.path.abspath(args.source)
    kind = "真实噪声" if args.as_noise else "干净语音"
    print(f"=== 导入{kind} ===")
    print(f"  来源：{src}{'（递归）' if args.recursive and os.path.isdir(src) else ''}")

    def _progress(done: int, total: int, name: str, stats: dict) -> None:
        if done % 25 == 0 or done == total:
            print(f"    已处理 {done}/{total}…", end="\r", flush=True)

    res = import_corpus(source=args.source, as_noise=args.as_noise, recursive=args.recursive,
                        move=args.move, normalize=args.normalize,
                        skip_duplicates=args.skip_duplicates, min_sec=args.min_sec,
                        max_sec=args.max_sec, dry_run=args.dry_run, verbose=args.verbose,
                        progress=None if args.dry_run else _progress)
    if res.get("error"):
        print(f"[错误] {res['error']}", file=sys.stderr)
        return 1
    print()
    if res.get("dry_run"):
        print(f"  候选 {res['n_candidates']} 个，前 10 个：")
        for p in res.get("preview", []):
            print("    " + os.path.basename(p))
        print("  去掉 --dry-run 即真正导入。")
        return 0
    print(f"  目标：{os.path.relpath(res['target'], HERE)}｜候选 {res['n_candidates']} 个")
    print(f"  结果：成功 {res['n_ok']} 个（{res['seconds']:.1f} 秒）"
          f"｜重复跳过 {res['n_dup']}｜时长不符 {res['n_skip']}｜读取失败 {res['n_fail']}")
    if res["manifest"] and args.report:
        print("\n  清单（前 10 条）：")
        for rel, dur, _sr in res["manifest"][:10]:
            print(f"    {dur:6.2f}s  {rel}")
    for w in res.get("quality_warnings", []):
        print("  " + w)
    if args.verbose:
        for p, why in res.get("dropped", []):
            print(f"    [跳过] {os.path.basename(p)}：{why}")
    return 0


def do_audit(_args) -> int:
    import subprocess
    print("=== 调用语料体检（check_corpus.py）===")
    return subprocess.call([sys.executable, os.path.join(HERE, "check_corpus.py")])


def main() -> int:
    ap = argparse.ArgumentParser(
        description="训练语料批量导入 / 体检",
        epilog="导入完成后，在界面点「训练降噪模型」，或运行：\n"
               "  python train_model.py --noise-dir models\\noise_bank\\real")
    sub = ap.add_subparsers(dest="cmd", required=True)

    ip = sub.add_parser("import", help="批量导入音频（图形界面也有同样功能）")
    ip.add_argument("--from", dest="source", required=True, help="文件 / 文件夹 / zip")
    ip.add_argument("--recursive", action="store_true", help="递归子目录")
    ip.add_argument("--as-noise", action="store_true", help="导入噪声库而不是干净语音语料")
    ip.add_argument("--move", action="store_true", help="导入后删除源文件（默认只复制）")
    ip.add_argument("--no-normalize", dest="normalize", action="store_false", help="不做峰值归一化")
    ip.add_argument("--skip-duplicates", dest="skip_duplicates", action="store_true",
                    default=True, help="跳过重复内容（默认开启）")
    ip.add_argument("--min-sec", type=float, default=0.3, help="最短片段（默认 0.3 秒）")
    ip.add_argument("--max-sec", type=float, default=30.0, help="最长片段（默认 30 秒）")
    ip.add_argument("--dry-run", action="store_true", help="只预演，不写盘")
    ip.add_argument("--report", action="store_true", help="导入后打印清单")
    ip.add_argument("--verbose", action="store_true", help="打印每个跳过/失败的原因")
    ip.set_defaults(func=do_import)

    au = sub.add_parser("audit", help="体检现有语料（底噪、静音、训练建议）")
    au.set_defaults(func=do_audit)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
