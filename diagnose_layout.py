"""左侧面板宽度诊断：找出到底是哪个控件横向溢出、文字是否被截断。

用法：python diagnose_layout.py
"""

from __future__ import annotations

import os
import sys
import tkinter as tk
from tkinter import ttk

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from app import DSPApp      # noqa: E402


def walk(w):
    yield w
    for c in w.winfo_children():
        yield from walk(c)


def main() -> int:
    root = tk.Tk()
    app = DSPApp(root)
    root.update_idletasks()
    root.update()

    canvas = app.left_canvas
    cw = canvas.winfo_width()
    print(f"左侧画布宽度 = {cw} px")
    print(f"内容 frame 实际宽度 = {app.left_inner.winfo_width()} px，"
          f"请求宽度 = {app.left_inner.winfo_reqwidth()} px")
    print(f"scrollregion = {canvas.cget('scrollregion')}")
    print()

    # 逐个控件检查：需要的宽度 vs 实际分配到的宽度
    print("=== 横向是否被挤压/截断（req > 实际 即被压缩）===")
    print(f"{'控件':<34}{'类型':<14}{'需要宽':>7}{'实际宽':>7}{'需要高':>7}{'实际高':>7}")
    bad = []
    for w in walk(app.left_inner):
        if not w.winfo_ismapped() and w is not app.left_inner:
            continue
        try:
            rq_w, rq_h = w.winfo_reqwidth(), w.winfo_reqheight()
            ac_w, ac_h = w.winfo_width(), w.winfo_height()
        except Exception:
            continue
        if rq_w > ac_w + 1 or rq_h > ac_h + 1:
            txt = ""
            try:
                txt = str(w.cget("text"))[:22]
            except Exception:
                pass
            bad.append((w.winfo_class(), txt, rq_w, ac_w, rq_h, ac_h))

    for cls, txt, rq_w, ac_w, rq_h, ac_h in bad:
        print(f"{txt or '(无文字)':<34}{cls:<14}{rq_w:>7}{ac_w:>7}{rq_h:>7}{ac_h:>7}")
    if not bad:
        print("  （没有控件被压缩）")
    print()

    # 按钮文字是否放得下
    print("=== 功能按钮：文字宽度 vs 按钮宽度 ===")
    for key, btn in list(app.buttons.items()):
        txt_w = btn.winfo_reqwidth()
        act_w = btn.winfo_width()
        flag = "  ← 被压缩" if txt_w > act_w + 1 else ""
        print(f"  {key:<22} 需要 {txt_w:>4} px  实际 {act_w:>4} px{flag}")

    # 各标签页内容宽度
    print("\n=== 各标签页内容需要的宽度 ===")
    for t in app.left_tabs.tabs():
        w = root.nametowidget(t)
        print(f"  {app.left_tabs.tab(t, 'text'):<6} 需要 {w.winfo_reqwidth():>4} px "
              f"/ 画布 {cw} px" + ("  ← 超出！" if w.winfo_reqwidth() > cw else ""))
    root.destroy()
    return 0


if __name__ == "__main__":
    sys.exit(main())
