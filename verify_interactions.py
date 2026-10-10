"""界面交互功能的自动化验证：点击子图放大、降噪方案选择对话框。

这两项都是"人点才生效"的交互功能，容易写成"代码在但点了没反应"。
本脚本用程序方式驱动它们，检查：
    1. 点击子图确实会弹出放大窗口，且曲线数据被完整复制（点数/条数一致）；
    2. 对比完成后弹出的选择对话框里，每个方案都有单选项，选中后主界面确实采用了该结果。

运行：
    python verify_interactions.py
"""

from __future__ import annotations

import os
import sys
import time
import tkinter as tk
from tkinter import ttk

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from app import C_DEN, DSPApp, RESULT_DIR  # noqa: E402

LOG: list[str] = []


def log(text: str) -> None:
    print(text)
    LOG.append(text)


def main() -> int:
    os.makedirs(RESULT_DIR, exist_ok=True)
    root = tk.Tk()
    app = DSPApp(root)
    root.update()
    ok = 0
    total = 0

    def check(title: str, cond: bool, extra: str = "") -> None:
        nonlocal ok, total
        total += 1
        if cond:
            ok += 1
        log(f"  {'[OK]  ' if cond else '[FAIL]'} {title}" + (f"  {extra}" if extra else ""))

    log("== 界面交互功能验证 ==")

    # ---------- 1) 点击子图 -> 放大窗口 ----------
    app.on_time_before()
    app.on_spectrum_before()
    root.update()
    n_before = len(app._detail_windows)
    app._enlarge_axes(app.ax1)
    root.update()
    n_after = len(app._detail_windows)
    check("点击子图会弹出放大窗口", n_after == n_before + 1,
          f"放大窗口数 {n_before} -> {n_after}")

    # 放大窗口里的曲线应与原子图一致
    if n_after > n_before:
        info = app._detail_windows[-1]
        fig = info["figure"]
        bax = fig.axes[0]
        n_src = len(app.ax1.get_lines())
        n_dst = len(bax.get_lines())
        same_data = False
        if n_src and n_dst:
            a = np.asarray(app.ax1.get_lines()[0].get_ydata(), dtype=np.float64)
            b = np.asarray(bax.get_lines()[0].get_ydata(), dtype=np.float64)
            same_data = a.shape == b.shape and np.allclose(a, b, atol=1e-9)
        check("放大窗口完整复制曲线（数据逐点一致）", same_data,
              f"原图 {n_src} 条 / 放大图 {n_dst} 条，首条曲线{'一致' if same_data else '不一致'}")
        check("放大窗口标题继承了子图标题", bool(bax.get_title()),
              f"标题：{bax.get_title()[:32]}")
        info["window"].destroy()
    root.update()

    # ---------- 2) 方案选择对话框 ----------
    # 造三份假的"降噪结果"，直接驱动对话框，验证"能选、且选中的会被采用"
    y = app.y
    fake = {
        "原信号": y,
        "方案甲": (y * 0.9).astype(np.float32),
        "方案乙": (y * 0.5).astype(np.float32),
        "方案丙": (y * 0.99).astype(np.float32),
    }
    mets = {k: {"snr_est_improve_db": 1.0 * i, "noise_floor_drop_db": 2.0 * i,
                "speech_keep_db": -0.1 * i, "lsd_self_db": 3.0 * i}
            for i, k in enumerate(("方案甲", "方案乙", "方案丙"), start=1)}

    captured: dict = {}

    def drive() -> None:
        """在对话框弹出后：检查单选项齐全，并选中"方案乙"后确认。"""
        root.update()
        tops = [w for w in root.winfo_children() if isinstance(w, tk.Toplevel)]
        if not tops:
            captured["err"] = "没有弹出选择对话框"
            return
        top = tops[-1]
        radios = []

        def walk(w):
            for c in w.winfo_children():
                if isinstance(c, ttk.Radiobutton):
                    radios.append(c)
                walk(c)

        walk(top)
        captured["n_radios"] = len(radios)
        captured["texts"] = [r.cget("text") for r in radios]
        for r in radios:
            if r.cget("text") == "方案乙":
                r.invoke()                      # 选中它
                for button in r.master.winfo_children():
                    if isinstance(button, ttk.Button) and button.cget("text") == "放大看频谱":
                        button.invoke()
                        root.update()
                        detail = app._detail_windows[-1]
                        captured["spectrum_labels"] = [
                            [line.get_label() for line in ax.get_lines()]
                            for ax in detail["figure"].axes]
                        detail["window"].destroy()
        # 点击"采用所选方案"
        for c in top.winfo_children():
            for cc in c.winfo_children() if hasattr(c, "winfo_children") else []:
                if isinstance(cc, ttk.Button) and "采用" in str(cc.cget("text")):
                    cc.invoke()
                    return
        captured["err"] = "没找到『采用所选方案』按钮"

    root.after(120, drive)
    chosen = app._ask_method_choice("测试报告", fake, mets)
    check("对话框为每个方案提供单选项", captured.get("n_radios") == 3,
          f"单选项 {captured.get('n_radios')} 个：{captured.get('texts')}")
    check("选中的方案被正确返回", chosen == "方案乙", f"返回 {chosen!r}，错误 {captured.get('err')}")
    check("方案行的频谱按钮只显示该方案与原信号",
          captured.get("spectrum_labels") == [["原信号", "方案乙（当前查看）"], ["原信号", "方案乙"]],
          f"图例 {captured.get('spectrum_labels')}")

    # ---------- 3) 采用所选结果 ----------
    app._apply_denoised(fake[chosen], chosen) if chosen else None
    root.update()
    used = np.allclose(app.yx[:100], fake["方案乙"][:100], atol=1e-6) if app.yx is not None else False
    check("主界面确实采用了所选结果", used,
          f"当前结果前 100 点与『方案乙』{'一致' if used else '不一致'}")

    # ---------- 4) 单方案 / 全部方案的放大窗口 ----------
    app._plot_single_method_spectrum(fake, "方案甲")
    root.update()
    info = app._detail_windows[-1]
    check("可单独放大查看某方案频谱", len(app._detail_windows) >= 1,
          f"标题：{info['title'][:28]}")
    check("放大窗口为置顶子窗口（不会掉到主窗口后面）",
          bool(info["window"].transient()) is not None and bool(info["window"].transient()),
          "已设置 transient（跟随主窗口并置前）")
    axs = info["figure"].axes
    check("单方案大图含频谱+时域两个子图", len(axs) == 2, f"子图数 {len(axs)}")
    check("单方案频谱仅含原信号及指定方案",
          [line.get_label() for line in axs[0].get_lines()] == ["原信号", "方案甲（当前查看）"],
          f"曲线 {len(axs[0].get_lines())} 条")
    check("单方案波形仅含原信号及指定方案",
          [line.get_label() for line in axs[1].get_lines()] == ["原信号", "方案甲"],
          f"曲线 {len(axs[1].get_lines())} 条")

    app._plot_single_method_spectrum(fake, None)
    root.update()
    info2 = app._detail_windows[-1]
    check("『放大查看频谱』可显示全部方案", "全部方案" in info2["title"],
          f"标题：{info2['title']}")
    check("全部方案窗口保留全部频谱和波形",
          all([line.get_label() for line in ax.get_lines()] == list(fake)
              for ax in info2["figure"].axes))
    for d in app._detail_windows:
        d["window"].destroy()
    root.update()

    # ---------- 5) 对比对话框里能触发放大窗口 ----------
    app._last_compare = {"results": fake, "metrics": mets}
    n_before = len([d for d in app._detail_windows if d["window"].winfo_exists()])
    app._show_last_compare_enlarged()
    root.update()
    n_after = len([d for d in app._detail_windows if d["window"].winfo_exists()])
    check("对比对话框的『放大查看频谱』按钮可用", n_after >= n_before + 1,
          f"可见详情窗口 {n_before} -> {n_after}")
    for d in app._detail_windows:
        if d["window"].winfo_exists():
            d["window"].destroy()

    # ---------- 滚轮缩放 ----------
    app.on_time_before()
    root.update()
    canvas = app.canvas
    canvas.get_tk_widget().update()
    h = canvas.get_tk_widget().winfo_height()
    ax = app.ax1
    bb = ax.get_window_extent()
    cx, cy = (bb.x0 + bb.x1) / 2, (bb.y0 + bb.y1) / 2      # matplotlib 显示坐标（原点左下）
    y_tk = h - cy                                          # 转成 Tk 坐标（原点左上）

    class _Ev:
        pass

    ev = _Ev(); ev.x = int(cx); ev.y = int(y_tk)
    base = abs(ax.get_xlim()[1] - ax.get_xlim()[0])
    center_before = ax.transData.inverted().transform((cx, cy))[0]

    ev.delta = 120                                          # 向上滚 = 放大
    app._on_wheel_event(canvas, ev)
    zoomed = abs(ax.get_xlim()[1] - ax.get_xlim()[0])
    check("滚轮向上滚 = 放大（坐标范围变窄）", zoomed < base * 0.95,
          f"跨度 {base:.0f} -> {zoomed:.0f}")

    center_after = ax.transData.inverted().transform((cx, cy))[0]
    check("缩放围绕光标位置（光标下的数据点基本不动）",
          abs(center_after - center_before) < base * 0.01,
          f"光标处数据坐标 {center_before:.0f} -> {center_after:.0f}")

    ev.delta = -120                                         # 向下滚 = 缩小
    app._on_wheel_event(canvas, ev)
    back = abs(ax.get_xlim()[1] - ax.get_xlim()[0])
    check("滚轮向下滚 = 缩小（基本还原）", abs(back - base) / base < 0.02,
          f"跨度回到 {back:.0f}（原 {base:.0f}）")

    ev.x = 2; ev.y = 2; ev.delta = 120                      # 光标在子图外
    before_lim = ax.get_xlim()
    app._on_wheel_event(canvas, ev)
    check("光标不在子图上时忽略滚轮", ax.get_xlim() == before_lim)

    for _ in range(80):
        app._zoom_axes(ax, 0.8, cx, cy)
    s_in = abs(ax.get_xlim()[1] - ax.get_xlim()[0]) / base
    check("滚轮缩放有极限保护（不会被无限放大）", 0.01 <= s_in <= 100.0,
          f"连续放大 80 次后为初始的 {s_in:.1f}×（上限 100×）")

    # ---------- 拖动平移 + 不再窗口套窗口 ----------
    app.on_time_before()
    root.update()
    ax = app.ax1

    class _Ev:
        def __init__(self, a, x, y, btn=1):
            self.inaxes = a; self.button = btn; self.x = x; self.y = y
            self.canvas = app.canvas

    xa0, xb0 = ax.get_xlim()
    app._on_plot_click(_Ev(ax, 300, 300), False)      # 按下（不开窗）
    app._on_plot_drag(_Ev(ax, 380, 340))             # 拖动
    app._on_plot_release(_Ev(ax, 380, 340))          # 松开
    xa1, xb1 = ax.get_xlim()
    check("按住左键拖动可平移视图", abs(xa1 - xa0) > 1 and abs((xb1 - xa1) - (xb0 - xa0)) < 1,
          f"X 范围 {xa0:.0f}~{xb0:.0f} -> {xa1:.0f}~{xb1:.0f}（跨度不变 {xb1 - xa1:.0f}）")

    app.on_time_before()
    root.update()
    n_win = len([d for d in app._detail_windows if d["window"].winfo_exists()])
    app._on_plot_click(_Ev(app.ax1, 300, 300), True)   # 主界面点击 -> 开窗
    root.update()
    n_win2 = len([d for d in app._detail_windows if d["window"].winfo_exists()])
    check("主界面单击可弹出放大窗口", n_win2 == n_win + 1, f"{n_win} -> {n_win2}")

    app._on_plot_click(_Ev(app.ax1, 300, 300), False)  # 放大窗口内点击
    app._on_plot_click(_Ev(app.ax1, 310, 310), False)
    root.update()
    n_win3 = len([d for d in app._detail_windows if d["window"].winfo_exists()])
    check("放大窗口内再点击不会『窗口套窗口』", n_win3 == n_win2,
          f"{n_win2} -> {n_win3}（应不变）")

    if "--plots-only" in sys.argv:
        log(f"== 结果：{ok}/{total} 项通过 ==")
        with open(os.path.join(RESULT_DIR, "spectrum_interaction_verify.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(LOG))
        root.destroy()
        return 0 if ok == total else 1

    # ---------- 批量导入文件夹（GUI 路径，打桩文件对话框与弹窗） ----------
    import shutil as _sh
    import tempfile
    from tkinter import filedialog, messagebox

    from core import audio_io as _aio

    captured: list[str] = []
    tmpd = tempfile.mkdtemp(prefix="dspy_import_")
    y0, sr0 = _aio.load_audio(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                           "samples", "demo_speech.wav"))
    _aio.save_audio(os.path.join(tmpd, "clean_a.wav"), _aio.to_mono(y0), sr0)      # 正常素材
    _aio.save_audio(os.path.join(tmpd, "silent_b.wav"),
                    np.zeros(len(y0), dtype=np.float32), sr0)                      # 近乎静音

    record_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "recordings")
    previous_records = set(os.listdir(record_dir)) if os.path.isdir(record_dir) else set()
    orig_dir, orig_info, orig_ask = filedialog.askdirectory, messagebox.showinfo, messagebox.askyesno
    filedialog.askdirectory = lambda **_k: tmpd
    messagebox.showinfo = lambda title=None, message=None, **_k: captured.append(str(message))
    messagebox.askyesno = lambda *a, **k: True
    try:
        app.on_import_folder()
        for _ in range(150):               # 最多等 15 秒
            root.update()
            time.sleep(0.1)
            if captured:
                break
        root.update()
    finally:
        filedialog.askdirectory, messagebox.showinfo, messagebox.askyesno = orig_dir, orig_info, orig_ask

    check("GUI 批量导入：能完成并弹出结果", bool(captured),
          captured[0].splitlines()[0] if captured else "未收到结果弹窗")
    check("GUI 批量导入：结果里报告了成功数量", bool(captured) and "成功" in captured[0],
          captured[0].splitlines()[1] if captured and len(captured[0].splitlines()) > 1 else "")
    check("GUI 批量导入：导入了文件夹里的素材",
          any(f.startswith("bulk_") for f in os.listdir(record_dir)) if os.path.isdir(record_dir) else False,
          f"recordings/ 中 bulk_* 文件数 "
          f"{len([f for f in os.listdir(record_dir) if f.startswith('bulk_')]) if os.path.isdir(record_dir) else 0}")

    # 清理本次测试导入的内容
    if os.path.isdir(record_dir):
        for f in os.listdir(record_dir):
            if f.startswith("bulk_") and f not in previous_records:
                try:
                    os.remove(os.path.join(record_dir, f))
                except Exception:
                    pass
    _sh.rmtree(tmpd, ignore_errors=True)

    log(f"== 结果：{ok}/{total} 项通过 ==")
    with open(os.path.join(RESULT_DIR, "interaction_verify.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(LOG))
    root.destroy()
    return 0 if ok == total else 1


if __name__ == "__main__":
    sys.exit(main())
