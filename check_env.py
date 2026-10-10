"""换电脑前的环境自检：确认这台机器能不能跑起来（不需要 MATLAB）。

用法：
    python check_env.py          # 检查环境与文件完整性
    python check_env.py --run     # 检查完顺便启动界面

检查项：
    1. Python 版本（需要 3.10+）
    2. 六个依赖包是否可导入，以及版本
    3. 必需的模型/数据文件是否齐全（缺了就无法开箱即用）
    4. 目录是否可写（训练、导出、截图都要写盘）
    5. 麦克风 / 扬声器是否可用（可选，缺了只影响录音与播放）
"""

from __future__ import annotations

import importlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

OK, WARN, BAD = "  [OK]   ", "  [注意] ", "  [缺失] "

# 依赖包 -> (用途, 是否必需)
PACKAGES = {
    "numpy": ("数值计算（自训练模型）", True),
    "scipy": ("信号处理（滤波/重采样）", True),
    "matplotlib": ("绘图与界面", True),
    "sounddevice": ("录音 / 播放", True),
    "onnxruntime": ("预训练 GTCRN 降噪", True),
    "soundfile": ("WAV/MP3/FLAC 等格式支持", False),
}

# 必需文件 -> 说明（缺了就不算"开箱即用"）
FILES = {
    os.path.join("models", "denoise_mlp.npz"): "自训练降噪模型（⑥ 号按钮用）",
    os.path.join("models", "gtcrn", "gtcrn_simple.onnx"): "GTCRN 预训练模型（⑤ 号按钮用）",
    os.path.join("samples", "demo_speech.wav"): "示例干净语音（启动时自动载入）",
    "app.py": "界面主程序",
    "requirements.txt": "依赖清单",
}


def main() -> int:
    run_after = "--run" in sys.argv
    problems: list[str] = []
    warnings: list[str] = []

    print("=" * 62)
    print("  语音信号处理与机器学习降噪系统 —— 环境自检")
    print("  （语音降噪无需 PyTorch；音乐分离使用 pymss / PyTorch）")
    print("=" * 62)

    # 1) Python
    v = sys.version_info
    line = f"Python {v.major}.{v.minor}.{v.micro}  ({sys.platform})"
    if v >= (3, 10):
        print(OK + line)
    else:
        print(BAD + line + " —— 需要 3.10 或更高版本")
        problems.append("Python 版本过低")

    # 2) 依赖
    print("\n依赖包：")
    for mod, (use, required) in PACKAGES.items():
        try:
            m = importlib.import_module(mod)
            ver = getattr(m, "__version__", "?")
            print(f"{OK}{mod:<14} {ver:<10} {use}")
        except Exception as exc:
            if required:
                print(f"{BAD}{mod:<14} {'':<10} {use} —— {type(exc).__name__}")
                problems.append(f"缺少 {mod}")
            else:
                print(f"{WARN}{mod:<14} {'':<10} {use}（可选，缺失时只支持 WAV）")
                warnings.append(f"{mod} 未安装（可选）")

    print("\n音乐分离（扩展）：")
    from importlib.metadata import version, PackageNotFoundError
    for name in ("pymss", "torch", "av"):
        try:
            print(f"{OK}{name}: {version(name)}")
        except PackageNotFoundError:
            print(f"{WARN}{name} 未安装，音乐分离需 requirements-music.txt")
    music_root = os.path.join(HERE, "models", "music")
    model_files = [p for p in __import__("pathlib").Path(music_root).rglob("*") if p.suffix in (".pth", ".th", ".ckpt")]
    print(f"{OK if model_files else WARN}本地音乐模型 {len(model_files)} 个")

    # 3) 文件
    print("\n必需文件：")
    for rel, desc in FILES.items():
        p = os.path.join(HERE, rel)
        if os.path.isfile(p):
            print(f"{OK}{rel:<38} {os.path.getsize(p) // 1024:>7} KB  {desc}")
        else:
            print(f"{BAD}{rel:<38} {'':>7}     {desc}")
            problems.append(f"缺少文件 {rel}")

    # 4) 可写性
    print("\n目录可写性：")
    for rel in ("recordings", "results", "models"):
        d = os.path.join(HERE, rel)
        try:
            os.makedirs(d, exist_ok=True)
            probe = os.path.join(d, ".write_test")
            with open(probe, "w", encoding="utf-8") as f:
                f.write("ok")
            os.remove(probe)
            print(f"{OK}{rel + '/':<38} 可写")
        except Exception as exc:
            print(f"{BAD}{rel + '/':<38} 不可写 —— {exc}")
            problems.append(f"{rel} 不可写（训练/导出会失败）")

    # 5) 音频设备（可选）
    print("\n音频设备（可选，缺失只影响录音与播放）：")
    try:
        import sounddevice as sd
        ins = [d for d in sd.query_devices() if d.get("max_input_channels", 0) > 0]
        outs = [d for d in sd.query_devices() if d.get("max_output_channels", 0) > 0]
        if ins:
            print(f"{OK}麦克风：{len(ins)} 个，例如 {ins[0]['name'][:40]}")
        else:
            print(f"{WARN}没有检测到麦克风（「① 录音」不可用，其余功能正常）")
            warnings.append("无麦克风")
        if outs:
            print(f"{OK}扬声器：{len(outs)} 个，例如 {outs[0]['name'][:40]}")
        else:
            print(f"{WARN}没有检测到输出设备（播放不可用）")
            warnings.append("无输出设备")
    except Exception as exc:
        print(f"{WARN}无法枚举音频设备：{exc}")
        warnings.append("音频设备查询失败")

    # 结论
    print("\n" + "=" * 62)
    if problems:
        print(f"结论：发现 {len(problems)} 个问题，暂不能正常运行：")
        for p in problems:
            print("   -", p)
        print("\n修复建议：")
        print("   1) 缺依赖 → 运行  pip install -r requirements.txt")
        print("              （慢的话加国内镜像：-i https://pypi.tuna.tsinghua.edu.cn/simple）")
        print("   2) 版本过低 → 安装 Python 3.10+ 后再试")
        print("   3) 缺文件 → 确认整个 dsp_py 目录一起拷贝（尤其 models/ 和 samples/）")
        return 1
    print("结论：环境就绪，可以运行 ✓")
    if warnings:
        print("（提示：" + "；".join(warnings) + "）")
    print("\n启动方式：")
    print("   Windows :  双击 run_windows.bat   或   python app.py")
    print("   macOS/Linux:  bash run_unix.sh      或   python3 app.py")

    if run_after:
        print("\n正在启动界面…")
        os.execv(sys.executable, [sys.executable, os.path.join(HERE, "app.py")])
    return 0


if __name__ == "__main__":
    sys.exit(main())
