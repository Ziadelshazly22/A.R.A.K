#!/usr/bin/env bash
set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

if ! command -v python >/dev/null 2>&1; then
  echo -e "${RED}Python not found. Please install Python 3.8+ and retry.${NC}"
  exit 1
fi

PYVER=$(python - <<'PY'
import sys
print("%d.%d" % sys.version_info[:2])
PY
)
MAJ=${PYVER%%.*}
MIN=${PYVER##*.}
if [ "$MAJ" -lt 3 ] || { [ "$MAJ" -eq 3 ] && [ "$MIN" -lt 8 ]; }; then
  echo -e "${RED}Python >=3.8 required. Found ${PYVER}.${NC}"
  exit 1
fi

python -m venv venv
. venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt

echo -e "${GREEN}Environment ready. Activate with: source venv/bin/activate${NC}"
