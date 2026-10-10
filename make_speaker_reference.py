"""用项目内两段不同录音生成同时说话的参考混合音频及已知原始两路。"""
import hashlib
import json
from pathlib import Path
import numpy as np
import soundfile as sf
from core import audio_io
from core.music import resample_audio
from core.speakers import REFERENCE_DIR, ROOT


def make_reference():
    fs, seconds, offset_seconds = 16000, 6, .3
    length = round(fs * seconds)
    originals = [ROOT / "samples/demo_speech.wav", ROOT / "samples/manba.wav"]
    tracks, sources = [], []
    for index, path in enumerate(originals):
        raw, sr = audio_io.load_audio(str(path))
        mono = audio_io.to_mono(raw)
        mono = resample_audio(mono, sr, fs)[:, 0][:4 * fs]
        rms = float(np.sqrt(np.mean(mono ** 2)))
        if rms < 1e-8:
            raise ValueError(f"参考录音静音：{path}")
        values = np.zeros(length, np.float32)
        start = 0 if index == 0 else round(offset_seconds * fs)
        gain = .07 / rms
        values[start:start + len(mono)] = mono * gain
        tracks.append(values)
        sources.append({"source": path.relative_to(ROOT).as_posix(), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "offset_seconds": start / fs, "source_gain": gain, "used_seconds": len(mono) / fs})
    mix = tracks[0] + tracks[1]
    peak = float(np.max(np.abs(mix)))
    common_gain = .85 / peak if peak > .85 else 1.0
    tracks = [values * common_gain for values in tracks]
    mix = tracks[0] + tracks[1]
    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    files = {"reference_1": "reference_1.wav", "reference_2": "reference_2.wav", "mixture": "overlap_mix.wav"}
    for key, values in zip(files, [*tracks, mix]):
        sf.write(REFERENCE_DIR / files[key], values, fs, subtype="FLOAT")
    meta = {"fs": fs, "seconds": seconds, "common_gain": common_gain, "sources": sources,
            "files": files, "note": "原始两路仅作为测试参考；分离输出不固定对应参考1/2。原录音可能含背景声。"}
    (REFERENCE_DIR / "reference.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(str(REFERENCE_DIR / "overlap_mix.wav"))


if __name__ == "__main__":
    make_reference()
