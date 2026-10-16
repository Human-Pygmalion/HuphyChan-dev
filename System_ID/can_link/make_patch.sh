#!/bin/sh
# Regenerate huphy_patch/PATCH.diff: unified diff of the two patched files against the vendored
# origin/biped a1b67b1 originals, with paths relative to the HUPHY repo root (apply with patch -p1).
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
T=$(mktemp -d)
mkdir -p "$T/a/src/huphy/motors/robstride" "$T/b/src/huphy/motors/robstride"
V="$HERE/vendor/huphy_motors_biped_a1b67b1"
cp "$V/canbus.py" "$T/a/src/huphy/motors/canbus.py"
cp "$V/robstride/bus.py" "$T/a/src/huphy/motors/robstride/bus.py"
cp "$HERE/huphy_patch/canbus.py" "$T/b/src/huphy/motors/canbus.py"
cp "$HERE/huphy_patch/robstride/bus.py" "$T/b/src/huphy/motors/robstride/bus.py"
cd "$T"
git diff --no-index -- a b | sed -e 's#a/a/#a/#g' -e 's#b/b/#b/#g' > "$HERE/huphy_patch/PATCH.diff" || true
rm -rf "$T"
wc -l "$HERE/huphy_patch/PATCH.diff"
