#!/bin/sh
# Run HUPHY's OWN test suite against the patched files, without touching the HUPHY repo.
#   1. git archive origin/biped (read-only) into a temp dir      -> "orig" tree
#   2. copy it, lay the two huphy_patch files over it            -> "patched" tree
#   3. pytest both (needs numpy, pyyaml, pytest; PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 avoids broken plugins)
# Usage: ./run_upstream_suite.sh [READER_DEFAULT]      READER_DEFAULT=false also flips CanBus(reader=...)
#        back to False in the patched copy (upstream tests assume the thread is off by default).
# Takes ~5 minutes per tree.  Output: <tmp>/orig.txt, <tmp>/patched.txt (path printed).
set -e
HUPHY=${HUPHY:-/home/syaro/external_repos/HUPHY}
HERE=$(cd "$(dirname "$0")" && pwd)
T=$(mktemp -d)
mkdir -p "$T/orig"
git -C "$HUPHY" archive origin/biped | tar -x -C "$T/orig"
cp -r "$T/orig" "$T/patched"
cp "$HERE/huphy_patch/canbus.py" "$T/patched/src/huphy/motors/canbus.py"
cp "$HERE/huphy_patch/robstride/bus.py" "$T/patched/src/huphy/motors/robstride/bus.py"
if [ "$1" = "false" ]; then
  sed -i 's/        reader: bool = True,/        reader: bool = False,/' "$T/patched/src/huphy/motors/canbus.py"
fi
for v in orig patched; do
  (cd "$T/$v" && PYTHONPATH=src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q -p no:cacheprovider tests --tb=short > "$T/$v.txt" 2>&1) || true
  echo "== $v: $(tail -1 "$T/$v.txt")"
done
echo "logs in $T"
