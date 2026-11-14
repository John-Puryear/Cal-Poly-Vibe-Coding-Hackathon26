#!/bin/sh
set -eu
cd "$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
if [ -n "${BDHNCA_PYTHON:-}" ]; then
  demo_python="$BDHNCA_PYTHON"
elif [ -x .venv/bin/python ]; then
  demo_python=.venv/bin/python
elif [ -x ../../work/venv/bin/python ]; then
  demo_python=../../work/venv/bin/python
else
  echo 'Create .venv using the README instructions, or set BDHNCA_PYTHON.' >&2
  exit 1
fi
if [ "$#" -eq 0 ]; then
  set -- evaluate --checkpoint runs/demo/checkpoint.pt --out runs/my_evaluation
fi
exec "$demo_python" -m bdh_nca "$@"
