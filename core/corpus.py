"""语料发现、按来源分组划分和可复现的验证集清单。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly
from math import gcd

from . import audio_io

ROOT = Path(__file__).resolve().parents[1]


def default_clean_paths() -> list[str]:
    candidates = list((ROOT / "recordings").glob("*.wav"))
    candidates += [ROOT / "samples" / "demo_speech.wav", ROOT.parent / "data.wav",
                   ROOT.parent / "manba.wav"]
    for folder in ("原matlab项目", "原始测试音频（可以试这里的语音）"):
        candidates += [ROOT.parent / folder / name for name in ("data.wav", "manba.wav")]
    return [str(p.resolve()) for p in candidates if p.is_file()]


def read_clean(path: str, fs: int) -> np.ndarray:
    raw, sr = audio_io.load_audio(path)
    y = audio_io.to_mono(raw)
    if not len(y) or not np.isfinite(y).all():
        raise ValueError(f"无效音频：{path}")
    if sr != fs:
        g = gcd(int(sr), int(fs))
        y = resample_poly(y, fs // g, sr // g).astype(np.float32)
    peak = float(np.max(np.abs(y)))
    return (y / peak * 0.95).astype(np.float32) if peak else y


def _segments(y, source, group, fs, start=0, end=None):
    end = len(y) if end is None else end
    records, pieces = [], []
    for a in range(start, end, 4 * fs):
        b = min(a + 4 * fs, end)
        if b - a < fs:
            continue
        pieces.append(y[a:b].copy())
        records.append({"source": source, "group": group, "start": a, "end": b})
    return pieces, records


def split_clean_corpus(paths, fs=16000, val_ratio=0.25, seed=0):
    """多来源按内容哈希分组；单来源按时间隔离，先划分再切非重叠片段。"""
    if not 0 < val_ratio < 1:
        raise ValueError("验证集比例必须在 0 和 1 之间")
    sources = {}
    for path in paths:
        y = read_clean(path, fs)
        group = hashlib.sha256(y.tobytes()).hexdigest()
        sources.setdefault(group, (str(Path(path).resolve()), y))
    if not sources:
        raise ValueError("没有可用的语音来源")
    groups = list(sources)
    rng = np.random.default_rng(seed)
    train, val, tr_records, va_records = [], [], [], []
    if len(groups) > 1:
        rng.shuffle(groups)
        count = min(len(groups) - 1, max(1, round(len(groups) * val_ratio)))
        val_groups = set(groups[:count])
        for group in groups:
            source, y = sources[group]
            pieces, records = _segments(y, source, group, fs)
            (val if group in val_groups else train).extend(pieces)
            (va_records if group in val_groups else tr_records).extend(records)
        mode = "source_group"
    else:
        group = groups[0]
        source, y = sources[group]
        if len(y) < 2 * fs:
            raise ValueError("单个素材至少需要 2 秒，才能建立互不重叠的训练和验证集")
        boundary = min(len(y) - fs, max(fs, int(len(y) * (1 - val_ratio))))
        train, tr_records = _segments(y, source, group, fs, end=boundary)
        val, va_records = _segments(y, source, group, fs, start=boundary)
        mode = "single_source_temporal"  # 不声称说话人独立泛化。
    if not train or not val:
        raise ValueError("划分后训练集或验证集为空，请补充至少一秒的有效素材")
    manifest = {"version": 2, "fs": fs, "seed": seed, "mode": mode,
                "val_ratio": val_ratio, "train": tr_records, "validation": va_records}
    manifest["split_id"] = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    return train, val, manifest


def validation_pool(manifest_path: str) -> tuple[list[np.ndarray], dict]:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    fs = manifest["fs"]
    cache, pieces = {}, []
    for record in manifest["validation"]:
        source = record["source"]
        if source not in cache:
            y = read_clean(source, fs)
            if hashlib.sha256(y.tobytes()).hexdigest() != record["group"]:
                raise ValueError(f"验证语料已变化：{source}")
            cache[source] = y
        pieces.append(cache[source][record["start"]:record["end"]].copy())
    if not pieces:
        raise ValueError("清单没有验证语料")
    return pieces, manifest


def evaluation_pool(clean_paths=None, model_path=None, manifest_path=None, fs=16000):
    """优先复用模型的验证清单；演示素材明确不声称训练隔离。"""
    from . import data, model
    path = Path(manifest_path) if manifest_path else Path(model_path or model.default_model_path()).with_name("split_manifest.json")
    if not clean_paths and path.is_file():
        pool, manifest = validation_pool(str(path))
        if manifest["fs"] != fs:
            raise ValueError("评估采样率与验证清单不同，请使用训练时的采样率")
        print(f"使用固定验证清单：{path}；模式 {manifest['mode']}；ID {manifest['split_id'][:12]}")
        return pool
    print("[提示] 使用显式素材或演示素材；未校验与训练集隔离，不能据此宣称独立泛化。")
    return data.load_clean_pool(clean_paths or default_clean_paths(), fs=fs)
