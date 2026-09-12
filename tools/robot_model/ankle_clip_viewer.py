#!/usr/bin/env python3
"""발목 클리핑이 **정말 도는지 눈으로 보는** 화면. 모터로 나가는 길은 없다.

2026-09-13 · 사용자 요청. 알고리즘 자체는 `ankle_clip.py` 한 곳에 있고, 이 파일은
그것을 **부르기만** 합니다. 그래서 알고리즘을 고치면 화면은 그대로 두고 다시 켜기만
하면 됩니다.

    왼쪽 3차원   발목(크랭크 A·B — 로드 — 발판)이 실제로 어떤 자세가 되는지
    오른쪽 위    A·B 슬라이더. 클리핑을 껐다 켤 수 있음
    오른쪽 가운데 클리핑이 걸렸는가 · **어느 코드 블럭이** 반응했는가 · 무슨 값에
    오른쪽 아래  갈 수 있는 영역 그림 두 장 (크랭크 공간 / 발판 자세 공간)과 지금 자리

★ **모터로 나가지 않습니다.** 이 파일에는 통신선을 여는 코드도, 명령을 보내는 코드도
없습니다. 보는 것과 계산하는 것뿐입니다.

★ 못 막았을 때(`inside=False`)는 상태 칸이 **빨갛게** 되고 그림에도 빨간 점으로
찍힙니다. 그 상태를 만들 수 있으면 클리핑에 구멍이 있다는 뜻입니다.

    ankle_clip_viewer.py --port 8112
    ankle_clip_viewer.py --headless --a 62 --b 40      # 화면 없이 한 번만
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
  sys.path.insert(0, str(HERE))

from ankle_clip import AnkleClipper, ClipResult  # noqa: E402

XML = ("mujoco-sim/mjlab/src/mjlab/asset_zoo/robots/pygmalion/xmls/"
       "LegOnly_prototype-tempmass-motormeasured-armfix_v30_proxyfix_loop.xml")


# ---------------------------------------------------------------- 그림
def region_png(clipper: AnkleClipper, r: ClipResult, width: int = 560):
  """지금 상태를 그린 png 바이트. 왼쪽 크랭크 공간, 오른쪽 발판 자세 공간."""
  import matplotlib
  matplotlib.use("Agg")
  import matplotlib.pyplot as plt
  import numpy as np

  t = clipper.table
  fig, ax = plt.subplots(1, 2, figsize=(width / 100.0, width / 200.0 + 0.6), dpi=100)

  if t:
    axis = np.array(t["axis"])
    blo = np.array([np.nan if v is None else v for v in t["b_lo"]], dtype=float)
    bhi = np.array([np.nan if v is None else v for v in t["b_hi"]], dtype=float)
    m = ~np.isnan(blo)
    ax[0].fill_between(axis[m], blo[m], bhi[m], color="tab:green", alpha=.30)
    ax[0].plot(axis[m], blo[m], color="tab:green", lw=1.2)
    ax[0].plot(axis[m], bhi[m], color="tab:green", lw=1.2)
  lim = clipper.limit
  ax[0].add_patch(plt.Rectangle((-lim, -lim), 2 * lim, 2 * lim, fill=False,
                                ec="0.5", ls="--", lw=1.0))
  col = "tab:red" if not r.inside else "black"
  ax[0].plot([r.a_in], [r.b_in], "x", ms=9, mew=2.0, color="tab:orange")
  ax[0].plot([r.a_out], [r.b_out], "o", ms=7, color=col)
  ax[0].annotate("", xy=(r.a_out, r.b_out), xytext=(r.a_in, r.b_in),
                 arrowprops=dict(arrowstyle="->", color="tab:orange", lw=1.3))
  ax[0].set_xlabel("crank A [deg]"); ax[0].set_ylabel("crank B [deg]")
  ax[0].set_title("crank space", fontsize=9)
  ax[0].set_xlim(-lim - 4, lim + 4); ax[0].set_ylim(-lim - 4, lim + 4)
  ax[0].set_aspect("equal"); ax[0].grid(alpha=.25)

  p0, p1 = clipper.rom_pitch
  r0, r1 = clipper.rom_roll
  ax[1].add_patch(plt.Rectangle((p0, r0), p1 - p0, r1 - r0, fc="tab:green",
                                alpha=.25, ec="tab:green", lw=1.4))
  if r.pitch == r.pitch:                       # NaN 이 아니면
    ax[1].plot([r.pitch], [r.roll], "o", ms=8, color=col)
  ax[1].set_xlabel("foot pitch [deg]"); ax[1].set_ylabel("foot roll [deg]")
  ax[1].set_title("foot pose space", fontsize=9)
  ax[1].set_xlim(p0 - 30, p1 + 30); ax[1].set_ylim(r0 - 20, r1 + 20)
  ax[1].grid(alpha=.25)
  if not r.inside:
    for a_ in ax:
      for s in a_.spines.values():
        s.set_color("tab:red"); s.set_linewidth(2.4)

  fig.tight_layout(pad=0.4)
  buf = io.BytesIO()
  fig.savefig(buf, format="png")
  plt.close(fig)
  buf.seek(0)
  import PIL.Image
  return np.asarray(PIL.Image.open(buf).convert("RGB"))


def status_markdown(r: ClipResult) -> str:
  head = ("### ✅ 막았습니다" if r.inside else "### ❌ **못 막았습니다**")
  if not r.changed and r.inside:
    head = "### ⬜ 손댈 것 없음 (명령이 이미 안전)"
  lines = [
    head,
    "",
    f"| | A | B |",
    f"|---|---|---|",
    f"| 명령 | `{r.a_in:+.2f}` | `{r.b_in:+.2f}` |",
    f"| 나감 | `{r.a_out:+.2f}` | `{r.b_out:+.2f}` |",
    "",
    f"발판 자세 **숙임 {r.pitch:+.2f}° · 기울임 {r.roll:+.2f}°**",
    "",
    f"반응한 코드 블럭: **{' → '.join(r.stages) if r.stages else '없음'}**  "
    f"(반복 {r.rounds}회)",
  ]
  if r.notes:
    lines.append("")
    lines += [f"* {n}" for n in r.notes]
  return "\n".join(lines)


# ---------------------------------------------------------------- 화면
def run(host: str, port: int, leg: str) -> int:
  import mujoco
  import numpy as np
  import viser
  from mjviser import ViserMujocoScene

  clipper = AnkleClipper(leg)

  # 모델을 여기서 다시 세우지 않습니다. `tools/viewer/mjcf_joint_viewer.py` 가 이미
  # 같은 파일(legonly/loop)에 크랭크 위치 서보를 붙이고 중력을 끄는 일을 하고
  # 있으므로 **그 함수를 그대로 불러 씁니다**. 예전에는 여기서 원본 XML 을 직접
  # 열었는데, 그 파일에는 액추에이터가 하나도 없어(nu=0) 크랭크가 전혀 안 움직였고
  # 게다가 중력에 떠 있는 다리가 그대로 떨어져 화면에서 사라졌습니다 (2026-09-13).
  sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "viewer"))
  import mjcf_joint_viewer as JV

  model, crank_actuators = JV.build_model("legonly", "loop")
  data = mujoco.MjData(model)

  side = "L" if leg == "left" else "R"

  def jid(name: str) -> int:
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)

  act = {tag: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR,
                                crank_actuators[f"{side}_crank_{tag}_joint"])
         for tag in "AB"}
  foot_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_foot_link")

  # 크랭크와 로드·발목만 움직이게 두고 나머지(떠 있는 몸통 포함)는 매 스텝 붙잡습니다.
  # 이것도 `mjcf_joint_viewer` 의 규약을 그대로 가져온 것입니다.
  free_names = JV.LOOP_HIDDEN + JV.CRANK_JOINTS + JV.LOOP_PASSIVE_READOUT
  free_q = {int(model.jnt_qposadr[jid(n)]) for n in free_names}
  free_v = {int(model.jnt_dofadr[jid(n)]) for n in free_names}
  pin_q = sorted(set(range(model.nq)) - free_q)
  pin_v = sorted(set(range(model.nv)) - free_v)

  # 지운 것: geom 을 안 보이는 그림표(5번)로 옮겨 발목 위쪽을 잘라내려 했는데,
  # 그러면 **남긴 8개까지 통째로 안 그려집니다** (2026-09-13 화면으로 확인). 잘라내는
  # 대신 아래처럼 카메라를 발목에 붙이는 쪽으로 갑니다.

  # 화면을 발목에 맞추는 두 번째 손질: **카메라 대신 모델을 옮깁니다.**
  # `client.camera` 로 넣은 값은 접속·1.2초 뒤·단추 세 경우 모두 화면에 반영되지
  # 않았습니다 (예외도 안 납니다). 그래서 몸통을 띄우는 높이를 골라 **발판이 원점에
  # 오게** 합니다 — viser 의 기본 카메라가 원점을 보므로 이러면 항상 발목이 잡힙니다.
  # 몸통은 매 스텝 붙잡히는 자리라 물리에는 아무 영향이 없습니다.
  data.qpos[:] = 0.0
  data.qpos[0:3] = [0, 0, 1.0]
  data.qpos[3:7] = [1, 0, 0, 0]
  mujoco.mj_forward(model, data)
  off = data.xpos[foot_bid].copy()
  data.qpos[0:3] = [-off[0], -off[1], 1.0 - off[2]]
  home = data.qpos.copy()
  mujoco.mj_forward(model, data)

  server = viser.ViserServer(host=host, port=port, label="발목 클리핑 확인")
  scene = ViserMujocoScene(server, model, num_envs=1)
  # ★ 이것 때문에 화면이 계속 엉덩이였습니다. 이 장면 객체는 기본값으로 **따라다니는
  # 몸통을 원점에 붙여 놓습니다** (`update_from_mjdata` 안에서 scene_offset =
  # -tracked_pos). 그래서 모델을 아무리 옮기고 카메라를 발판에 맞춰도, 그릴 때마다
  # 몸통이 원점으로 끌려와 발목이 화면 밖으로 나갔습니다 (2026-09-13 화면으로 확인).
  # 여기서는 다리가 매달린 채 가만히 있으므로 따라다닐 이유가 없습니다.
  scene.camera_tracking_enabled = False

  state = dict(a=0.0, b=0.0, bypass=False, last=None)

  def settle(a_deg: float, b_deg: float) -> None:
    """크랭크를 그 각으로 몰아넣고 닫힌 고리를 풀어 준다.

    `mj_forward` 만으로는 로드 끝을 발판에 붙이는 구속이 안 풀립니다 -- 그것은
    `mj_step` 의 구속 해석기가 합니다 (`tools/viewer/mjcf_joint_viewer.py` 가 같은
    이유로 같은 방법을 씁니다). 여기서는 크랭크 위치 서보의 목표를 천천히 올리면서 스텝을 돌립니다.
    """
    goal = {tag: np.radians(a_deg if tag == "A" else b_deg) for tag in "AB"}
    start = {tag: float(data.ctrl[act[tag]]) for tag in "AB"}
    steps = JV.SETTLE_STEPS
    for k in range(steps):
      f = min(1.0, k / (steps / 2))          # 절반에 걸쳐 올리고 나머지는 유지
      for tag in "AB":
        data.ctrl[act[tag]] = start[tag] + f * (goal[tag] - start[tag])
      data.qpos[pin_q] = home[pin_q]
      data.qvel[pin_v] = 0.0
      mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)

  def apply() -> None:
    r = clipper.clip(state["a"], state["b"])
    if state["bypass"]:
      out_a, out_b = state["a"], state["b"]
    else:
      out_a, out_b = r.a_out, r.b_out
    settle(out_a, out_b)
    scene.update_from_mjdata(data)
    state["last"] = r
    status.content = status_markdown(r) + (
      "\n\n> ⚠ **클리핑을 꺼 둔 상태입니다** — 화면의 3차원은 명령 그대로입니다."
      if state["bypass"] else "")
    image.image = region_png(clipper, r)
    # 시뮬레이터가 구속을 풀어 실제로 세운 발판 각도. 우리 기구학이 낸 값과
    # **다르면** 둘 중 하나가 틀린 것이므로 같이 보여 줍니다.
    pj = {n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_ankle_{n}_joint")
          for n in ("pitch", "roll")}
    sim = {n: (np.degrees(data.qpos[model.jnt_qposadr[j]]) if j >= 0 else float("nan"))
           for n, j in pj.items()}
    # 예전에는 **숙임만** 비교했습니다. 그래서 기울임이 3도씩 어긋나 있는데도 화면은
    # "차이 0.29도" 라고 안심시켰습니다 (2026-09-13 발견). 둘 다 봅니다.
    gp = abs(sim["pitch"] - r.pitch) if r.pitch == r.pitch else float("nan")
    gr = abs(sim["roll"] - r.roll) if r.roll == r.roll else float("nan")
    worst = max([g for g in (gp, gr) if g == g], default=float("nan"))
    pose.content = (
      f"시뮬레이터가 세운 발판: 숙임 `{sim['pitch']:+.2f}°` · 기울임 `{sim['roll']:+.2f}°`  \n"
      f"우리 기구학과의 차이: 숙임 `{gp:.2f}°` · 기울임 `{gr:.2f}°`"
      + ("  ⚠ 둘이 다릅니다" if worst == worst and worst > 1.0 else ""))

  with server.gui.add_folder("AB 모터"):
    sa = server.gui.add_slider("크랭크 A [도]", min=-90.0, max=90.0, step=0.25, initial_value=0.0)
    sb = server.gui.add_slider("크랭크 B [도]", min=-90.0, max=90.0, step=0.25, initial_value=0.0)
    bypass = server.gui.add_checkbox("클리핑 끄기 (막는 것을 눈으로 확인)", False)
    reset = server.gui.add_button("0, 0 으로")
    look = server.gui.add_button("발목으로 화면 맞추기")

  with server.gui.add_folder("클리핑"):
    status = server.gui.add_markdown(" ")
    pose = server.gui.add_markdown(" ")

  with server.gui.add_folder("갈 수 있는 영역"):
    image = server.gui.add_image(region_png(clipper, clipper.clip(0.0, 0.0)),
                                 label="크랭크 공간 / 발판 자세 공간")

  @sa.on_update
  def _(_):
    state["a"] = float(sa.value); apply()

  @sb.on_update
  def _(_):
    state["b"] = float(sb.value); apply()

  @bypass.on_update
  def _(_):
    state["bypass"] = bool(bypass.value); apply()

  @reset.on_click
  def _(_):
    sa.value = 0.0; sb.value = 0.0

  # ★ 카메라를 **발목에** 붙입니다. 요청은 "발목 부분만 잘라낸 3차원" 이었는데,
  # 모델에서 몸통을 잘라내려면 파일을 새로 만들어야 하고 그러면 기구가 달라질 위험이
  # 있습니다. 그래서 자르는 대신 **그 부분만 크게** 봅니다 -- 보이는 것은 같고,
  # 모델은 학습·실물과 한 글자도 다르지 않습니다.
  def frame_ankle(client: viser.ClientHandle) -> None:
    tgt = data.xpos[foot_bid].copy()
    # ★ 순서가 중요합니다. `look_at` 을 먼저 넣고 `position` 을 넣으면 viser 가
    # **보는 방향을 유지한 채** 눈 위치만 옮겨서, 되읽어 보면 look_at 이
    # (-2.85, -2.51, -2.48) 처럼 엉뚱한 데로 가 있습니다 (2026-09-13 실측).
    # 눈 위치를 먼저 놓고, 그 다음에 어디를 볼지 말해야 합니다.
    client.camera.up_direction = (0.0, 0.0, 1.0)
    # 발판(원점)에서 크랭크 모터까지가 0.30 m 입니다. 그 한가운데를 보고 0.55 m
    # 뒤에서 잡으면 크랭크 두 개 - 로드 - 발판이 한 화면에 다 들어옵니다.
    client.camera.position = (tgt[0] + 0.40, tgt[1] + 0.32, tgt[2] + 0.26)
    client.camera.look_at = (tgt[0], tgt[1], tgt[2] + 0.15)

  @server.on_client_connect
  def _(client: viser.ClientHandle) -> None:
    # 접속 직후에 한 번 넣으면 화면 쪽이 아직 제 카메라를 세우는 중이라 덮어써
    # 버립니다 (2026-09-13 실측: 발목 대신 엉덩이가 잡혔음). 조금 기다렸다 다시
    # 넣습니다.
    frame_ankle(client)
    threading.Timer(1.2, lambda: frame_ankle(client)).start()

  @look.on_click
  def _(event) -> None:
    frame_ankle(event.client)

  apply()
  print(f"  화면: http://{host}:{port}/   (모터로 나가는 길 없음)")
  while True:
    time.sleep(1.0)


def main(argv=None) -> int:
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--host", default="0.0.0.0")
  ap.add_argument("--port", type=int, default=8112)
  ap.add_argument("--leg", default="left", choices=("left", "right"))
  ap.add_argument("--headless", action="store_true", help="화면 없이 한 번 계산만")
  ap.add_argument("--a", type=float, default=0.0)
  ap.add_argument("--b", type=float, default=0.0)
  a = ap.parse_args(argv)

  if a.headless:
    r = AnkleClipper(a.leg).clip(a.a, a.b)
    print(json.dumps(r.as_dict(), ensure_ascii=False, indent=1))
    return 0 if r.inside else 1
  return run(a.host, a.port, a.leg)


if __name__ == "__main__":
  raise SystemExit(main())
