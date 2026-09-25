#!/usr/bin/env bash
# Size every top-level entry of the shared volume, gently: idle I/O class, lowest CPU
# priority, one entry at a time, 20 minutes max per entry. Results stream to a TSV so a
# partial run is still useful. Metadata only - no file is opened or read.
OUT=${1:-/tmp/du_volume.tsv}
: > "$OUT"
cd /home/syaro/MikuchanRemote || exit 1
for e in * .[!.]* '#recycle'; do
  [ -e "$e" ] || continue
  t0=$(date +%s)
  s=$(timeout 1200 ionice -c3 nice -n 19 du -sb -- "$e" 2>/dev/null | cut -f1)
  rc=$?
  printf '%s\t%s\t%s\t%s\n' "${s:-NA}" "$e" "$(( $(date +%s) - t0 ))" "$rc" >> "$OUT"
done
echo DONE >> "$OUT"
