"""带音频版本号的后台任务，旧结果不能应用到新素材。"""
from __future__ import annotations
import threading
from uuid import uuid4
import numpy as np


class SignalTaskRunner:
    def __init__(self, emit):
        self.emit = emit
        self.revision = 0
        self.current_id = None
        self.busy = False
        self.closed = False
        self._lock = threading.Lock()

    def invalidate(self):
        self.revision += 1

    def submit(self, name, audio, fs, work):
        with self._lock:
            if self.closed or self.busy:
                raise RuntimeError("已有音频任务正在处理，请等待完成")
            self.busy = True
            task_id = self.current_id = uuid4().hex
            revision = self.revision
        snapshot = np.array(audio, dtype=np.float32, copy=True)
        sample_rate = int(fs)

        def run():
            event = {"type": "signal_result", "task_id": task_id, "revision": revision,
                     "name": name, "audio": snapshot, "fs": sample_rate}
            try:
                event["result"] = work(snapshot, sample_rate)
            except Exception as exc:
                event["error"] = str(exc)
            finally:
                with self._lock:
                    self.busy = False
                if not self.closed:
                    self.emit(event)
        threading.Thread(target=run, daemon=True).start()
        return task_id

    def accepts(self, event):
        return (not self.closed and event.get("revision") == self.revision
                and event.get("task_id") == self.current_id)

    def close(self):
        self.closed = True
        self.invalidate()
