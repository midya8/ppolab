#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
PYTHON=${PYTHON:-python3}
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
"$PYTHON" -m venv --system-site-packages "$TMP/venv"
if command -v uv >/dev/null 2>&1; then
  uv pip install --python "$TMP/venv/bin/python" --quiet --no-deps "$ROOT" --target "$TMP/site"
else
  "$TMP/venv/bin/python" -m pip install --quiet --no-deps "$ROOT" --target "$TMP/site"
fi
cd "$TMP"
PYTHONPATH="$TMP/site" "$TMP/venv/bin/python" - <<'PY'
import importlib,json,pathlib
from importlib import resources
for name in ['ppolab.clipface','ppolab.clipverify']:
 module=importlib.import_module(name)
 assert '/site/' in module.__file__,module.__file__
from ppolab.clipface import audit
from ppolab.clipverify import verify
fixtures=json.loads(resources.files('ppolab').joinpath('data/clip-fixtures.json').read_text())
assert len(fixtures)==10
for f in fixtures:
 r=audit(f['input']['old'],f['input']['advantages'],f['input']['epsilon'])
 assert r==f['report'] and verify(r)
pathlib.Path('input.json').write_text(json.dumps(fixtures[0]['input']))
pathlib.Path('expected.json').write_text(json.dumps(fixtures[0]['report'],sort_keys=True))
print('Two installed module origins and all ten complete independently replayed reports pass')
PY
PYTHONPATH="$TMP/site" "$TMP/venv/bin/python" -m ppolab.clipface input.json > report.json
PYTHONPATH="$TMP/site" "$TMP/venv/bin/python" -m ppolab.clipverify report.json
PYTHONPATH="$TMP/site" "$TMP/venv/bin/python" - <<'PY'
import json
assert json.load(open('report.json'))==json.load(open('expected.json'))
PY
printf '%s\n' 'Exact producer and independent verifier CLI reproduce outside checkout.'
