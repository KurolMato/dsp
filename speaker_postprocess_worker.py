"""分离后的逐轨增强 worker：保留原轨、写独立增强轨和参数清单。"""
import json
from pathlib import Path
import sys
import time
import traceback


def emit(kind, **data):
    print("DSP_EVENT " + json.dumps({"type": kind, **data}, ensure_ascii=False), flush=True)


def run(request):
    import numpy as np
    import soundfile as sf
    from core import audio_io
    from core.speaker_cleanup import cleanup_track, validate_config
    from core.gtcrn import GTCRNDenoiser
    t0 = time.monotonic()
    source = request["source_result"]
    fs, samples = int(source["fs"]), int(source["samples"])
    configs = {name: validate_config(request["configs"][name]) for name in ("speaker_1", "speaker_2")}
    raw_sources = source.get("raw_files", source["files"])
    output = Path(request["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    emit("stage", message="载入原始分离音轨和降噪模型")
    needs_model = any(cfg["denoise_enabled"] and cfg["denoise_strength"] > 0 for cfg in configs.values())
    denoiser = GTCRNDenoiser() if needs_model else None
    if denoiser is not None and not denoiser.is_ready:
        raise RuntimeError(f"GTCRN 模型不可用：{denoiser.load_error}")
    raw_files, enhanced_files = {}, {}
    for name in ("original", "speaker_1", "speaker_2"):
        audio, sr = audio_io.load_audio(raw_sources[name])
        if audio.ndim == 2 and audio.shape[1] == 1:
            audio = audio[:, 0]
        if sr != fs or audio.shape != (samples,) or not np.isfinite(audio).all():
            raise ValueError("原始音轨与结果清单不匹配")
        raw_path = output / ("original.wav" if name == "original" else name + "_raw.wav")
        sf.write(raw_path, audio, fs, subtype="FLOAT")
        raw_files[name] = str(raw_path.resolve())
        if name == "original":
            enhanced_files[name] = raw_files[name]
            continue
        emit("stage", message=f"增强音轨 {name[-1]}：轻度降噪 / 可选噪声门")
        cleaned = cleanup_track(audio, fs, configs[name], denoiser)
        path = output / (name + "_enhanced.wav")
        sf.write(path, cleaned, fs, subtype="FLOAT")
        enhanced_files[name] = str(path.resolve())
        emit("progress", done=int(name[-1]), total=2, message=f"已增强 {name[-1]}/2 条音轨")
    result = {**source, "task_id": request["task_id"], "files": enhanced_files,
              "raw_files": raw_files, "enhanced_files": enhanced_files, "audition_version": "enhanced",
              "postprocess": {"algorithm": "GTCRN blend + soft noise gate", "tracks": configs,
                              "source_task_id": source.get("task_id"), "seconds": time.monotonic() - t0}}
    temp = output / "result.json.tmp"
    temp.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(output / "result.json")
    emit("done", result=result)


if __name__ == "__main__":
    try:
        run(json.loads(Path(sys.argv[1]).read_text(encoding="utf-8")))
    except Exception as exc:
        emit("error", message=str(exc), detail=traceback.format_exc())
        raise SystemExit(1)
