"""Walk the Human-Pygmalion repo once and account every byte to a category.

Categories are chosen to answer "what could go, and what must stay":
checkpoints are split into the ones the thinning tool would remove and the ones it keeps,
using the same rule (last per run + every 1000th, runs active in the last 6 h untouched).
"""
import json, os, re, time
ROOT = "/home/syaro/MikuchanRemote/Human-Pygmalion"
CK = re.compile(r"model_(\d+)\.pt$")
now = time.time()
week = now - 7 * 86400

agg = {}          # (area, kind) -> [bytes, files, bytes_last7d]
run_ck = {}       # run dir -> list of (iter, size, mtime, path)
biggest = []      # (size, path)

def area_of(rel):
    p = rel.split(os.sep)
    if p[0] == "mujoco-sim" and len(p) > 2 and p[1] == "mjlab":
        if p[2] in ("logs", ".venv"): return f"mjlab/{p[2]}"
        if p[2] == "analysis" and len(p) > 3 and p[3] == "out": return "mjlab/analysis/out"
        return "mjlab/(code 등)"
    if p[0] == "pygmalion_locomotion" and len(p) > 1 and p[1] == "logs": return "pygmalion_locomotion/logs"
    if p[0] in (".git", "docs", "tools", "refs", "System_ID", "WebPage"): return p[0]
    return "기타"

def kind_of(name):
    if CK.search(name): return "checkpoint"
    n = name.lower()
    if n.endswith((".mp4", ".webm", ".gif", ".avi")): return "video"
    if n.startswith("events.out.tfevents"): return "tensorboard"
    if n.endswith(".npz") or n.endswith(".npy"): return "measurement npz"
    if n.endswith(".mjb") or n.endswith(".xml") or n.endswith(".stl") or n.endswith(".obj"): return "model/mesh"
    if n.endswith((".png", ".jpg", ".svg")): return "image"
    if n.endswith((".pack", ".idx")): return "git pack"
    if n.endswith(".onnx"): return "onnx"
    if n.endswith((".so", ".pyc")) or "site-packages" in n: return "python env"
    return "other"

stack = [ROOT]
while stack:
    d = stack.pop()
    try:
        it = os.scandir(d)
    except OSError:
        continue
    with it:
        for e in it:
            try:
                if e.is_symlink(): continue
                if e.is_dir(follow_symlinks=False):
                    stack.append(e.path); continue
                st = e.stat(follow_symlinks=False)
            except OSError:
                continue
            rel = os.path.relpath(e.path, ROOT)
            a = area_of(rel)
            k = kind_of(e.name)
            if a == "mjlab/.venv": k = "python env"
            key = f"{a}\t{k}"
            v = agg.setdefault(key, [0, 0, 0])
            v[0] += st.st_size; v[1] += 1
            if st.st_mtime > week: v[2] += st.st_size
            m = CK.search(e.name)
            if m and "logs" in rel:
                run_ck.setdefault(os.path.dirname(e.path), []).append((int(m.group(1)), st.st_size, st.st_mtime))
            if st.st_size > 200 * 2**20:
                biggest.append((st.st_size, rel))

thin = keep = 0; per_run = []
for d, cks in run_ck.items():
    if now - max(t for _, _, t in cks) < 6 * 3600:
        keep += sum(s for _, s, _ in cks); continue
    last = max(i for i, _, _ in cks)
    rb = 0
    for i, s, _ in cks:
        if i == last or i % 1000 == 0: keep += s
        else: thin += s; rb += s
    per_run.append((rb, sum(s for _, s, _ in cks), os.path.basename(d)))
per_run.sort(reverse=True)
biggest.sort(reverse=True)
out = {
    "agg": {k: v for k, v in agg.items()},
    "thin_bytes": thin, "keep_bytes": keep,
    "per_run_top": per_run[:25],
    "biggest_files": biggest[:25],
    "measured_at": time.strftime("%Y-%m-%d %H:%M"),
}
json.dump(out, open(os.environ.get("DU_REPO_JSON", "/tmp/du_repo.json"), "w"), ensure_ascii=False, indent=1)
tot = sum(v[0] for v in agg.values())
print(f"저장소 합계 {tot/2**30:.1f} GiB, 최근 7일 증가 {sum(v[2] for v in agg.values())/2**30:.1f} GiB")
byarea = {}
for k, v in agg.items():
    a = k.split("\t")[0]; byarea[a] = byarea.get(a, 0) + v[0]
for a, b in sorted(byarea.items(), key=lambda x: -x[1]):
    print(f"  {b/2**30:6.1f} GiB  {a}")
print(f"솎을 수 있는 저장본 {thin/2**30:.1f} GiB / 남길 저장본 {keep/2**30:.1f} GiB")
