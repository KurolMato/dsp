"""音乐分离 worker：公开 pymss API + JSON 行事件，不依赖桌面 GUI。"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import time
import traceback


def emit(kind, **payload):
    print("DSP_EVENT " + json.dumps({"type": kind, **payload}, ensure_ascii=False), flush=True)


def run(request):
    import numpy as np
    import soundfile as sf
    try:
        import torch
        import av
        from pymss import MSSeparator, download_model, resolve_model
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "音乐分离依赖缺失或无法加载。请关闭程序，在项目目录运行 "
            "setup_music_windows.bat，确认安装成功后用 run_windows.bat 启动。"
            "\n非 Windows 系统请在当前 Python 环境安装 PyTorch 和 requirements-music.txt。"
            f"\n当前 Python：{sys.executable}\n原始错误：{exc}"
        ) from exc
    from core import audio_io
    from core.music import resample_audio

    t0 = time.monotonic()
    model_dir = request["model_dir"]
    model = request["model"]
    emit("stage", stage="download", message="检查 / 下载模型文件")

    def progress(done, total, message=None):
        emit("progress", done=float(done), total=float(total or 0), message=message or "处理中")

    try:
        resolve_model(model, model_dir=model_dir, require_exists=True)
    except FileNotFoundError:
        download_model(model, model_dir=model_dir, source=request.get("source", "modelscope"),
                       progress_callback=progress)
    if request.get("download_only"):
        emit("done", download_only=True, model=model, message="模型已就绪")
        return
    emit("stage", stage="load", message="加载分离模型")
    with MSSeparator.from_model_name(
        model, model_dir=model_dir, download=False, device=request.get("device", "auto"),
        output_format="wav", store_dirs=request["output_dir"], progress_callback=progress,
        inference_params={"batch_size": 1, "normalize": False}) as separator:
        model_fs = int(separator.config.audio.get("sample_rate", 44100))
        original, fs = audio_io.load_audio(request["input"])
        original = resample_audio(original, fs, fs)
        if original.shape[1] > 2:
            raise ValueError("音乐分离支持单声道和立体声，请先将多声道素材转换为立体声")
        work = resample_audio(original, fs, model_fs)
        emit("stage", stage="separate", message="正在分离音轨")
        outputs = separator.separate(work.T, pbar=False)
        tracks = {}
        for name, audio in outputs.items():
            key = str(name).lower().strip()
            if key in ("vocal", "voice"):
                key = "vocals"
            elif key in ("instrument", "accompaniment"):
                key = "instrumental"
            tracks[key] = resample_audio(audio, model_fs, fs, length=len(original))
        if "vocals" not in tracks:
            raise ValueError("所选模型没有输出人声音轨")
        if "instrumental" not in tracks:
            others = [a for name, a in tracks.items() if name != "vocals"]
            tracks["instrumental"] = sum(others, np.zeros_like(original)) if others else original - tracks["vocals"]
        if any(a.shape != original.shape or not np.isfinite(a).all() for a in tracks.values()):
            raise ValueError("分离结果声道、长度或数值异常")
        emit("stage", stage="export", message="保存音轨及结果清单")
        directory = Path(request["output_dir"])
        # 使用浮点 WAV 保持原始增益；试听限幅，混音导出可显式归一化。
        tracks = {"original": original, **tracks}
        files = {}
        for name, audio in tracks.items():
            path = directory / f"{name}.wav"
            sf.write(path, audio, fs, subtype="FLOAT")
            files[name] = str(path.resolve())
        result = {"version": 1, "task_id": request["task_id"], "input": request["input"],
                  "model": model, "device": str(separator.device), "fs": fs,
                  "channels": original.shape[1], "samples": len(original),
                  "seconds": time.monotonic() - t0, "files": files}
        temp = directory / "result.json.tmp"
        temp.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        temp.replace(directory / "result.json")
        emit("done", result=result, manifest=str(directory / "result.json"))


def main():
    try:
        request = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
        run(request)
        return 0
    except Exception as exc:
        emit("error", message=str(exc), detail=traceback.format_exc())
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
