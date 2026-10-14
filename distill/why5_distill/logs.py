"""Shared console/file logging for Why5 pipelines.

Console lines: [HH:MM:SS][LEVEL][tag] message
Optional file sink: call start_file_log(path); every log() line is appended.
"""

import os
import time

_sink = None


def _ts():
    return time.strftime("%H:%M:%S")


def log(tag, msg, level="INFO"):
    line = f"[{_ts()}][{level:>5}][{tag}] {msg}"
    print(line, flush=True)
    if _sink:
        _sink.write(line + "\n")
        _sink.flush()


def start_file_log(path):
    global _sink
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    _sink = open(path, "a")
    log("logger", f"file log started: {path}")
    return path
