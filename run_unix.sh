#!/usr/bin/env bash
set -eu
cd "$(dirname "$0")"
if [ -x .venv/bin/python ]; then
  PY=.venv/bin/python
else
  PY=python3
fi
"$PY" -X utf8 check_env.py
"$PY" -X utf8 app.py
