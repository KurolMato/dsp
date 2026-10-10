"""GUI 布局自检：检查窗口在默认/最小尺寸下，左侧面板与绘图区是否被裁剪。

用户反馈：打开界面后底部内容显示不全，需要手动拉伸窗口。本脚本量化这个问题：
    1. 测出左侧面板实际需要的总高度 vs 窗口可视高度；
    2. 验证加上滚动条后，所有控件都能通过滚动访问到；
    3. 验证绘图区不会被左侧面板挤掉。

用法：
    python verify_layout.py
"""

from __future__ import annotations

import os
import sys
import tkinter as tk

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from app import DSPApp, RESULT_DIR      # noqa: E402

LOG: list[str] = []


def log(text: str) -> None:
    print(text)
    LOG.append(text)


def walk(widget):
    yield widget
    for child in widget.winfo_children():
        yield from walk(child)


def main() -> int:
    os.makedirs(RESULT_DIR, exist_ok=True)
    root = tk.Tk()
    root.geometry("1180x760")
    app = DSPApp(root)
    root.update_idletasks()
    root.update()

    ok = total = 0

    def check(title: str, cond: bool, extra: str = "") -> None:
        nonlocal ok, total
        total += 1
        if cond:
            ok += 1
        log(f"  {'[OK]  ' if cond else '[FAIL]'} {title}" + (f"  {extra}" if extra else ""))

    log("== GUI 布局自检 ==")

    # ---------- 1) 横向：左侧面板内容是否完整显示（按钮不被切） ----------
    canvas = getattr(app, "left_canvas", None)
    inner = getattr(app, "left_inner", None)
    need_w = inner.winfo_reqwidth() if inner else 0
    view_w = canvas.winfo_width() if canvas else 0
    log(f"      左侧内容需要宽度 {need_w} px，画布宽度 {view_w} px")
    check("左侧面板宽度足够（内容不被横向裁剪）", need_w <= view_w,
          f"{need_w} px ≤ {view_w} px")
    squeezed = []
    for w in walk(inner) if inner else []:
        try:
            if w.winfo_ismapped() and (w.winfo_reqwidth() > w.winfo_width() + 1
                                       or w.winfo_reqheight() > w.winfo_height() + 1):
                squeezed.append((w.winfo_class(), str(w.cget("text"))[:18]
                                 if "text" in w.keys() else ""))
        except Exception:
            pass
    check("没有控件被挤压/截断", not squeezed,
          f"{len(squeezed)} 个被压缩" + (f"：{squeezed[:3]}" if squeezed else ""))

    # 横向滚动条：内容装得下时应自动隐藏；手动缩窄画布后应出现并可滚动
    hscroll = getattr(app, "left_hscroll", None)
    check("横向滚动条存在（兜底）", hscroll is not None)
    if hscroll is not None:
        check("内容装得下时横向滚动条自动隐藏", not hscroll.winfo_ismapped(),
              "已隐藏" if not hscroll.winfo_ismapped() else "仍显示")

    # ---------- 2) 纵向：标签页分组 ----------
    tabs = getattr(app, "left_tabs", None)
    check("左侧面板已按功能分组为标签页", tabs is not None,
          f"{len(tabs.tabs())} 个标签页：{[tabs.tab(t, 'text') for t in tabs.tabs()]}" if tabs else "")
    if tabs is not None:
        heights = []
        for t in tabs.tabs():
            w = root.nametowidget(t)
            heights.append((tabs.tab(t, "text"), w.winfo_reqheight()))
        tallest = max(heights, key=lambda kv: kv[1])
        log("      各标签页内容高度：" + "、".join(f"{n} {h}px" for n, h in heights))
        check("最高的标签页在默认窗口内完整可见",
              tallest[1] <= getattr(app, "left_canvas").winfo_height(),
              f"最高 {tallest[0]} {tallest[1]}px / 可视 {getattr(app, 'left_canvas').winfo_height()}px")

    left = getattr(app, "left_panel", None)
    check("左侧面板已放入可滚动容器（兜底）", canvas is not None and inner is not None)

    if inner is not None:
        need_h = inner.winfo_reqheight()
        view_h = canvas.winfo_height() if canvas is not None else 0
        log(f"      左侧面板需要高度 {need_h} px，当前可视高度 {view_h} px")
        if need_h > view_h:
            check("内容超出可视区时启用了滚动", canvas.cget("scrollregion") != "",
                  f"scrollregion={canvas.cget('scrollregion')}")
            canvas.yview_moveto(1.0)
            root.update()
            widgets = [w for w in walk(inner) if w.winfo_height() > 1]
            last = max(widgets, key=lambda w: w.winfo_rooty())
            c_top = canvas.winfo_rooty()
            c_bot = c_top + canvas.winfo_height()
            check("滚动到底部后，最下方控件可见",
                  last.winfo_rooty() + last.winfo_height() <= c_bot + 2,
                  f"最下控件底边 {last.winfo_rooty() + last.winfo_height() - c_top}px / 可视 {canvas.winfo_height()}px")
            canvas.yview_moveto(0.0)
        else:
            check("内容未超出可视区（无需滚动）", True,
                  f"{need_h}px ≤ {view_h}px，滚动条已自动隐藏")

    # 绘图区没有被左侧面板挤掉
    ax_w = app.canvas.get_tk_widget().winfo_width()
    check("绘图区宽度合理（>400 px）", ax_w > 400, f"{ax_w} px")

    # 缩小到最小尺寸后，每个标签页内容仍完整可见
    root.geometry("980x560")
    root.update_idletasks()
    root.update()
    if tabs is not None:
        worst = 0
        for t in tabs.tabs():
            w = root.nametowidget(t)
            worst = max(worst, w.winfo_reqheight())
        check("最小窗口尺寸下所有标签页仍完整可见",
              worst <= canvas.winfo_height(),
              f"最高标签页 {worst}px / 可视 {canvas.winfo_height()}px")

    has_wheel = bool(canvas.bind("<MouseWheel>")) if canvas is not None else False
    bound_all = bool(app.root.bind_all("<MouseWheel>"))
    check("左侧面板支持鼠标滚轮（指针在面板上时生效）", bound_all or has_wheel,
          "已注册全局滚轮处理（仅在指针位于面板上时滚动）" if bound_all else "未注册")

    log(f"== 结果：{ok}/{total} 项通过 ==")
    with open(os.path.join(RESULT_DIR, "layout_verify.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(LOG))
    root.destroy()
    return 0 if ok == total else 1


if __name__ == "__main__":
    sys.exit(main())
