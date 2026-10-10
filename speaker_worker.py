"""独立双人语音分离 worker；模型推理不会改变音乐分离进程或环境。"""
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback


def emit(kind, **payload):
    print("DSP_EVENT " + json.dumps({"type": kind, **payload}, ensure_ascii=False), flush=True)


def download_model(directory, source):
    import requests
    from core.speakers import MODEL_ID, REVISION, model_ready
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if model_ready(directory):
        manifest = json.loads((directory / "model_manifest.json").read_text(encoding="utf-8"))
        if all(hashlib.sha256((directory / name).read_bytes()).hexdigest() == record["sha256"] for name, record in manifest["files"].items()):
            return
        emit("stage", message="缓存校验失败，重新获取模型文件")
    endpoints = {"hf-mirror": "https://hf-mirror.com", "huggingface": "https://huggingface.co"}
    if source not in endpoints:
        raise ValueError("未知下载源")
    manifest = {"model": MODEL_ID, "revision": REVISION, "files": {}}
    names = ("hyperparams.yaml", "encoder.ckpt", "decoder.ckpt", "masknet.ckpt")
    for name in names:
        path, part = directory / name, directory / (name + ".part")
        url = f"{endpoints[source]}/{MODEL_ID}/resolve/{REVISION}/{name}"
        # 文件下载采用有长度校验的续传，取消时 .part 不会被当作完整模型。
        done = False
        expected_hash = None
        for attempt in range(5):
            offset = part.stat().st_size if part.exists() else 0
            headers = {"Range": f"bytes={offset}-"} if offset else {}
            try:
                with requests.get(url, headers=headers, stream=True, timeout=(20, 90)) as response:
                    if response.status_code == 416 and offset:
                        part.unlink()
                        continue
                    response.raise_for_status()
                    append = offset > 0 and response.status_code == 206
                    if not append:
                        offset = 0
                    remaining = int(response.headers.get("Content-Length", 0))
                    total = remaining + offset
                    for redirect in [*response.history, response]:
                        etag = redirect.headers.get("X-Linked-Etag", "").strip('"')
                        if len(etag) == 64:
                            expected_hash = etag
                    emit("progress", done=offset, total=total, message=f"下载 {name}")
                    with part.open("ab" if append else "wb") as writer:
                        reported = offset
                        for data in response.iter_content(1024 * 1024):
                            if data:
                                writer.write(data)
                                offset += len(data)
                                if offset - reported >= 1024 * 1024:
                                    emit("progress", done=offset, total=total, message=f"下载 {name}")
                                    reported = offset
                    if remaining and offset != total:
                        raise RuntimeError("模型文件传输中断，继续下载")
                digest = hashlib.sha256(part.read_bytes()).hexdigest()
                if expected_hash and digest != expected_hash:
                    part.unlink()
                    raise RuntimeError("模型文件 SHA256 不符，重新下载")
                part.replace(path)
                manifest["files"][name] = {"bytes": path.stat().st_size, "sha256": digest}
                done = True
                break
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(1)
        if not done:
            raise RuntimeError(f"下载失败：{name}")
    temporary = directory / "model_manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(directory / "model_manifest.json")


def run(request):
    t0 = time.monotonic()
    emit("stage", stage="download", message="检查 / 下载双人分离模型")
    download_model(request["model_dir"], request.get("source", "hf-mirror"))
    if request.get("download_only"):
        emit("done", download_only=True)
        return
    emit("stage", stage="load", message="载入双人分离模型")
    import numpy as np
    import soundfile as sf
    import torch
    from speechbrain.inference.separation import SepformerSeparation
    from speechbrain.utils.fetching import LocalStrategy
    from core import audio_io
    from core.music import resample_audio
    from core.speakers import MODEL_ID, REVISION, separate_chunks

    torch.set_num_threads(min(4, os.cpu_count() or 1))
    device = request.get("device", "auto")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device not in ("cpu", "cuda") or (device == "cuda" and not torch.cuda.is_available()):
        raise ValueError("当前独立环境无法使用所选设备，请选择 CPU")
    directory = Path(request["model_dir"]).resolve()
    # 模型无需自定义模块。SpeechBrain 1.0.3 对本地 COPY 模式缺省文件
    # 抛 FileNotFoundError 而非忽略；提供空文件兼容该公开加载接口。
    custom = directory / "custom.py"
    if not custom.exists():
        custom.write_text("# No custom modules are required for this model.\n", encoding="utf-8")
    model = SepformerSeparation.from_hparams(source=str(directory), savedir=str(directory / "runtime"),
            run_opts={"device": device}, local_strategy=LocalStrategy.COPY)
    raw, fs = audio_io.load_audio(request["input"])
    if raw.ndim not in (1, 2) or not raw.size or not np.isfinite(raw).all() or (raw.ndim == 2 and raw.shape[1] > 2):
        raise ValueError("需要有效的单声道或立体声双人混合音频")
    input_channels = 1 if raw.ndim == 1 else raw.shape[1]
    mono = audio_io.to_mono(raw)
    work = resample_audio(mono, fs, 8000)[:, 0]
    emit("stage", stage="separate", message="分离两路语音（不识别身份）")
    chunk_gains = []

    def infer(piece):
        if np.max(np.abs(piece)) < 1e-8:
            chunk_gains.append([0.0, 0.0])
            return np.zeros((len(piece), 2), np.float32)
        with torch.inference_mode():
            # separate_batch 不进行 separate_file 的逐轨峰值归一化，保留输出的相对增益。
            raw_estimate = model.separate_batch(torch.from_numpy(piece.copy())[None]).detach().cpu().numpy()[0]
        # SI-SNR 训练不约束输出的绝对增益/极性。以原混合音频最小二乘校准
        # 两路输出，避免未经校准的超满幅结果和相邻块极性翻转。
        gains = np.linalg.lstsq(raw_estimate.astype(np.float64), piece.astype(np.float64), rcond=1e-4)[0]
        chunk_gains.append(gains.tolist())
        return (raw_estimate * gains).astype(np.float32)

    estimates, swaps = separate_chunks(work, infer, chunk_seconds=8, progress=lambda done, total: emit(
        "progress", done=done, total=total, message=f"已处理 {done / 8000:.1f}/{total / 8000:.1f} 秒"))
    estimates = resample_audio(estimates, 8000, fs, length=len(mono))
    emit("stage", stage="export", message="保存两路分离结果及清单")
    output = Path(request["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    files = {}
    for name, audio in {"original": mono, "speaker_1": estimates[:, 0], "speaker_2": estimates[:, 1]}.items():
        path = output / (name + ".wav")
        sf.write(path, audio, fs, subtype="FLOAT")
        files[name] = str(path.resolve())
    result = {"version": 1, "kind": "speaker_separation", "task_id": request["task_id"],
              "input": request["input"], "model": MODEL_ID, "revision": REVISION, "device": device,
              "model_fs": 8000, "fs": fs, "channels": 1, "input_channels": input_channels,
              "samples": len(mono), "seconds": time.monotonic() - t0, "overlap_reorders": swaps,
              "chunk_seconds": 8, "overlap_seconds": 1, "chunk_output_gains": chunk_gains, "files": files}
    temporary = output / "result.json.tmp"
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(output / "result.json")
    emit("done", result=result)


if __name__ == "__main__":
    try:
        run(json.loads(Path(sys.argv[1]).read_text(encoding="utf-8")))
    except Exception as exc:
        emit("error", message=str(exc), detail=traceback.format_exc())
        raise SystemExit(1)
