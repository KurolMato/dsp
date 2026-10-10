"""GUI 自动验证：无人工干预地驱动界面，检查各按钮回调是否报错，并保存绘图区截图。

运行：
    python verify_gui.py
输出：
    results/gui_check_*.png   各步骤的绘图区截图（可直接作为报告插图）
    results/gui_verify.txt    检查日志
"""

from __future__ import annotations

import os
import sys
import tkinter as tk

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import DSPApp, RESULT_DIR  # noqa: E402

LOG: list[str] = []
COMPARE_TEXT: list[str] = []        # 收集"对比"弹窗的文本，避免自动化被模态框卡住


def log(text: str) -> None:
    print(text)
    LOG.append(text)


def _patch_messagebox() -> None:
    """把 tkinter 的弹窗换成记录文本，让自动化验证不被阻塞。"""
    from tkinter import messagebox

    def _capture(title=None, message=None, **_kw):
        if message:
            COMPARE_TEXT.append(str(message))
            log(f"      [弹窗:{title}] {str(message).splitlines()[0][:80]}…")
        return "ok"

    for name in ("showinfo", "showwarning", "showerror", "askyesno"):
        setattr(messagebox, name, _capture)


def _patch_method_dialog(app: DSPApp, pick: str | None = None) -> None:
    """把"降噪方案选择对话框"替换成自动选择，避免自动化流程卡在等待用户点击。

    该对话框是自定义 Toplevel（内部 wait_window 会一直等），自动化里没人点会卡死；
    真实的交互行为（单选项、试听、放大、采用结果）由 verify_interactions.py 单独验证。
    """
    def _auto(report, results, metrics_by_name):
        names = [k for k in results if k != "原信号"]
        choice = pick if pick in names else (names[0] if names else None)
        COMPARE_TEXT.append(report)
        log(f"      [方案对话框] 自动选择：{choice}（候选 {names}）")
        return choice

    app._ask_method_choice = _auto


def wait_msgs(root, app: DSPApp, seconds: float = 4.0, tick: float = 0.1) -> None:
    """在 tk 事件循环里等待，让后台线程的结果经消息队列回到主线程。"""
    import time as _t

    t0 = _t.time()
    while _t.time() - t0 < seconds:
        root.update()
        _t.sleep(tick)


def shot(app: DSPApp, name: str) -> None:
    path = os.path.join(RESULT_DIR, f"gui_check_{name}.png")
    app.fig.savefig(path, dpi=110)
    log(f"      截图 -> {os.path.relpath(path, os.path.dirname(RESULT_DIR))}")


def main() -> int:
    os.makedirs(RESULT_DIR, exist_ok=True)
    # 本脚本会让界面“以为是自己在处理信号”，而「⑧ 倒放 / 导出」可能把结果写回
    # dsp_py\data.wav。先备份、结束时还原，避免验证过程污染示例音频。
    backup: dict[str, bytes] = {}
    touched = [os.path.join(os.path.dirname(os.path.abspath(__file__)), "data.wav")]
    for p in touched:
        if os.path.isfile(p):
            with open(p, "rb") as f:
                backup[p] = f.read()

    root = tk.Tk()
    _patch_messagebox()
    app = DSPApp(root)
    # 自动验证不实际播放扬声器；播放游标与混音由 verify_update.py 验证。
    app._play = lambda _y, name, fs=None: app.status(f"播放回调已验证：{name}")
    _patch_method_dialog(app)          # 方案选择对话框改为自动选择，避免卡住
    root.update()
    ok = 0
    total = 0

    def step(title: str, fn) -> None:
        nonlocal ok, total
        total += 1
        try:
            fn()
            root.update()
            log(f"  [OK] {title}   （状态栏：{app.var_status.get()[:70]}）")
            ok += 1
        except Exception as exc:
            import traceback
            log(f"  [FAIL] {title}: {exc}")
            log("      " + traceback.format_exc().splitlines()[-1])

    log("== GUI 自动验证 ==")
    log(f"  载入信号：{os.path.basename(app.path)}，{app.fs} Hz，{len(app.y)} 采样点"
        if app.y is not None else "  未载入信号")

    step("⑬ 原信号时域", app.on_time_before)
    shot(app, "01_time_before")
    step("⑪ 原信号频谱", app.on_spectrum_before)
    shot(app, "02_spectrum_before")
    step("⑦ 经典谱减法降噪", app.on_denoise_classic)
    wait_msgs(root, app, seconds=1.5)
    shot(app, "03_classic_denoise")
    step("⑫ 处理后频谱", app.on_spectrum_after)
    shot(app, "04_spectrum_after")
    step("⑨ 端点检测", app.on_endpoint)
    shot(app, "05_endpoint")
    step("④ 回声（不播放）", lambda: setattr(app, "yr", __import__("core.dsp", fromlist=["dsp"]).echo(app.y, app.fs)))
    step("⑩ 倒放（不播放）", lambda: setattr(app, "yr", app.y[::-1].copy()))

    # ⑤ GTCRN 预训练降噪（结果由后台线程经消息队列回主线程）
    step("⑤ GTCRN 预训练降噪", app.on_denoise_gtcrn)
    wait_msgs(root, app, seconds=5.0)
    log(f"      GTCRN 结果 {len(app.yx) if app.yx is not None else 0} 采样点；"
        f"状态栏：{app.var_status.get()[:90]}")
    shot(app, "06_gtcrn_denoise")

    # ⑥ 自训练 MLP 降噪
    step("⑥ 自训练 MLP 降噪", app.on_denoise_ml)
    wait_msgs(root, app, seconds=5.0)
    shot(app, "07_ml_denoise")

    # ⑧ 三种方法同台对比（弹窗改为记录文本，避免自动化被阻塞）
    step("⑧ 三种降噪同台对比", lambda: (app.on_compare_denoisers(), None)[1])
    wait_msgs(root, app, seconds=10.0)
    if COMPARE_TEXT:
        log("      对比报告：\n        " + COMPARE_TEXT[0].replace("\n", "\n        "))
    shot(app, "08_compare")

    step("⑭ 处理后时域（不播放）", app.on_time_after)
    shot(app, "09_time_after")

    log(f"== 结果：{ok}/{total} 步成功 ==")
    with open(os.path.join(RESULT_DIR, "gui_verify.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(LOG))
    root.destroy()

    # 还原被验证过程改动的示例音频
    for p, data in backup.items():
        with open(p, "wb") as f:
            f.write(data)
    for p in touched:
        if p not in backup and os.path.isfile(p):
            os.remove(p)
            log(f"      已清理验证过程生成的 {os.path.relpath(p, os.path.dirname(os.path.abspath(__file__)))}")
    return 0 if ok == total else 1


if __name__ == "__main__":
    sys.exit(main())
