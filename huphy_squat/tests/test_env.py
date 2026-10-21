"""GPU checks of the assembled task: reset, ankle linkage, DR, delays, push, actuator, rewards."""

from __future__ import annotations

import math

import pytest
import torch

from huphy_squat import robot
from huphy_squat.env_cfg import FOOT_FRICTION, OBS_MAX_LAG, PAYLOAD_KG, PUSH
from huphy_squat.mdp import commands, events, observations, rewards

from conftest import loop_residual, make_env, sole_corners_z


def _zero(env):
  return torch.zeros(env.num_envs, 12, device=env.device)


def _ids(r, names):
  return [r.joint_names.index(n) for n in names]


def _term(env, name):
  return env.reward_manager.get_term_cfg(name)


def _eval_term(env, name):
  """Raw (unweighted, clamped) value of reward term `name` for the current sim state."""
  cfg = _term(env, name)
  return cfg.func(env, **cfg.params)


# --- reset ----------------------------------------------------------------------------------

def test_reset_state(env):
  env.reset()
  env.sim.forward()
  r = env.scene["robot"]
  mi = _ids(r, robot.MOTOR_JOINT_NAMES)
  dq = r.data.joint_pos[:, mi] - r.data.default_joint_pos[:, mi]
  assert dq.abs().max() <= 0.05 + 1e-6 and dq.abs().max() > 0.04  # noise present and bounded
  assert loop_residual(env).max() < 1e-4  # passive joints consistent with the cranks (< 0.1 mm)
  z = sole_corners_z(env).min(dim=1).values
  assert z.min() > -1e-4 and z.max() < 2.5e-3, (z.min(), z.max())  # lowest corner ~1 mm above the floor
  assert torch.allclose(r.data.projected_gravity_b[:, 2], torch.full_like(z, -1.0), atol=1e-5)
  assert r.data.joint_vel.abs().max() == 0 and r.data.root_link_lin_vel_w.abs().max() == 0
  h = commands.base_height(env)
  assert 0.875 < h.min() and h.max() < 0.925  # default 0.8935 +- the noise


def test_reset_settling(env):
  """Zero action right after reset: the robot rocks onto its soles within the 0.2 s vz_speed
  grace window, then |vz| is small; loops stay closed; no non-foot contact."""
  env.reset()
  vz, res, nonfoot, pen = [], [], 0, []
  for _ in range(15):
    env.step(_zero(env))
    vz.append(float(env.scene["robot"].data.root_link_lin_vel_w[:, 2].abs().max()))
    res.append(float(loop_residual(env).max()))
    nonfoot += int(env.termination_manager.get_term("nonfoot_contact").sum())
    pen.append(float(_eval_term(env, "vz_speed").max()))
  assert max(vz[8:]) < 0.12, vz  # after the grace window (10 steps)
  assert max(pen[:9]) == 0.0, pen  # grace: no vz penalty for the reset transient
  assert max(res) < 5e-4, res
  assert nonfoot == 0


def test_command_first_draw_and_keep(env):
  env.reset()
  c = env.command_manager.get_term("height")
  h = c.command[:, 0]
  assert h.min() >= 0.72 and h.max() <= 0.895 and h.std() > 0.03
  old = h.clone()
  ids = torch.arange(env.num_envs, device=env.device)
  c._resample(ids)  # noqa: SLF001
  kept = (c.command[:, 0] == old).float().mean().item()
  assert 0.1 < kept < 0.3, kept  # keep_prob 0.2
  c.command_counter[:] = 0
  c._resample_command(ids)  # noqa: SLF001
  assert (c.command[:, 0] == old).float().mean() < 0.02  # first draw of an episode never keeps


# --- ankle linkage ---------------------------------------------------------------------------

def test_crank_to_ankle_mapping():
  """Cranks in phase move ankle pitch, in anti-phase ankle roll (robot fixed in the air)."""
  env = make_env(16, edit=_no_dr_hanging)
  env.reset()
  r = env.scene["robot"]
  ap, ar = _ids(r, ("R_ankle_pitch_joint",)), _ids(r, ("R_ankle_roll_joint",))
  base = r.data.joint_pos[0, ap + ar].clone()
  out = {}
  for name, a in (("pitch", (1.0, -1.0)), ("roll", (1.0, 1.0))):
    env.reset()
    act = _zero(env)
    act[:, 4], act[:, 5] = a  # R crank A, R crank B (action order = robot.MOTOR_JOINT_NAMES)
    for _ in range(40):
      env.step(act)
    out[name] = (r.data.joint_pos[0, ap + ar] - base).tolist()
  assert abs(out["pitch"][0]) > 0.1 and abs(out["pitch"][1]) < 0.03, out
  assert abs(out["roll"][1]) > 0.1 and abs(out["roll"][0]) < 0.03, out


def _no_dr_hanging(cfg):
  for k in list(cfg.events):
    if k not in ("reset_pose", "motor_step_end", "obs_lag"):
      cfg.events.pop(k)
  cfg.events["reset_pose"].params["motor_noise"] = 0.0
  cfg.terminations = {"time_out": cfg.terminations["time_out"]}

  def lift(env, env_ids, **kw):  # keep the base 1 m above the floor, fixed (weld-like hold)
    r = env.scene["robot"]
    st = torch.zeros(len(env_ids), 13, device=env.device)
    st[:, :3] = env.scene.env_origins[env_ids]
    st[:, 2] += 1.5
    st[:, 3] = 1.0
    r.write_root_state_to_sim(st, env_ids=env_ids)

  from mjlab.managers.event_manager import EventTermCfg
  cfg.events["hold_base"] = EventTermCfg(mode="step", func=lambda env, env_ids: lift(env, torch.arange(env.num_envs, device=env.device)))


def test_extreme_passive_dr_stays_closed():
  """Worst passive DR (friction 0.5 N*m, damping 0) + random actions + pushes: no blow-up, loops closed."""
  def edit(cfg):
    cfg.events["passive_friction"].params["ranges"] = (0.5, 0.5)
    cfg.events["passive_damping"].params["ranges"] = (0.0, 0.0)
  env = make_env(256, edit=edit)
  env.reset()
  worst, blow, rod = 0.0, 0, 0
  g = torch.Generator(device=env.device).manual_seed(0)
  for i in range(250):
    a = torch.randn(env.num_envs, 12, device=env.device, generator=g) * 0.5
    env.step(a)
    worst = max(worst, float(loop_residual(env).max()))
    blow += int(env.termination_manager.get_term("physics_blowup").sum())
    rod += int((rewards.contact_count(env, "rod_contact") > 0).sum())
  assert blow == 0
  assert worst < 2e-3, worst  # < 2 mm even with random actions and falls
  print(f"extreme passive DR: max loop residual {worst * 1e3:.3f} mm, rod-contact env-steps {rod}")


# --- DR ---------------------------------------------------------------------------------------

def _dof(env, field, names):
  r = env.scene["robot"]
  adr = r.indexing.joint_v_adr[_ids(r, names)].long()
  return getattr(env.sim.model, field)[:, adr], env.sim.get_default_field(field)[adr]


def test_dr_ranges_and_resample(env):
  env.reset()
  r = env.scene["robot"]
  snap = {}
  for field in ("dof_frictionloss", "dof_damping", "dof_armature"):
    live, nom = _dof(env, field, robot.MOTOR_JOINT_NAMES)
    ratio = live / nom
    assert ratio.min() >= 0.8 - 1e-5 and ratio.max() <= 1.2 + 1e-5 and ratio.std() > 0.05, field
    snap[field] = live.clone()
  for field, hi in (("dof_frictionloss", 0.5), ("dof_damping", 0.05)):
    live, _ = _dof(env, field, robot.ROD_JOINT_NAMES + robot.ANKLE_JOINT_NAMES)
    assert live.min() >= 0 and live.max() <= hi + 1e-6 and live.std() > hi / 5, field
  fr = observations.foot_friction(env, _cfg(env, geom_names=(robot.FOOT_GEOM_EXPR,)))
  assert fr.min() >= FOOT_FRICTION[0] - 1e-6 and fr.max() <= FOOT_FRICTION[1] + 1e-6
  m = events.get_motor(env)
  kp, kd = m.stiffness / m.default_stiffness, m.damping / m.default_damping
  assert kp.min() >= 0.8 - 1e-5 and kp.max() <= 1.2 + 1e-5 and kd.min() >= 0.8 - 1e-5 and kd.max() <= 1.2 + 1e-5
  assert torch.allclose(m.default_stiffness[0], torch.tensor([135, 135, 75, 130, 60, 60] * 2, dtype=torch.float, device=env.device))
  assert torch.allclose(m.default_damping[0], torch.tensor([6, 6, 4, 6, 2.7, 2.7] * 2, dtype=torch.float, device=env.device))
  b = r.data.encoder_bias
  mi = _ids(r, robot.MOTOR_JOINT_NAMES)
  others = [i for i in range(len(r.joint_names)) if i not in mi]
  assert b[:, mi].abs().max() <= 0.015 and b[:, mi].std() > 0.005 and b[:, others].abs().max() == 0
  q = env.sim.model.site_quat[:, r.indexing.site_ids[list(r.site_names).index(robot.IMU_SITE)].long()]
  q0 = env.sim.get_default_field("site_quat")[r.indexing.site_ids[list(r.site_names).index(robot.IMU_SITE)].long()]
  ang = 2 * torch.acos((q * q0).sum(-1).abs().clamp(max=1))
  assert ang.max() <= math.radians(2.0) * math.sqrt(3) + 1e-4 and ang.mean() > math.radians(0.5)
  # masses: pelvis = pseudo-inertia mass + payload, payload never accumulates over resets
  pel = r.indexing.body_ids[r.body_names.index("base_link")].long()
  nom = float(env.sim.get_default_field("body_mass")[pel])
  for _ in range(3):
    env.reset()
    pl = env.huphy_payload
    base = env.sim.model.body_mass[:, pel] - pl
    assert pl.min() >= PAYLOAD_KG[0] and pl.max() <= PAYLOAD_KG[1] and pl.std() > 1.0
    assert base.min() >= nom * math.exp(2 * -0.0256) - 1e-4 and base.max() <= nom * math.exp(2 * 0.0699) + 1e-4
  # everything is redrawn at reset
  for field in snap:
    assert not torch.allclose(_dof(env, field, robot.MOTOR_JOINT_NAMES)[0], snap[field])


def _cfg(env, **kw):
  from mjlab.managers.scene_entity_config import SceneEntityCfg

  c = SceneEntityCfg("robot", **kw)
  c.resolve(env.scene)
  return c


def test_encoder_bias_same_in_obs_and_action(env):
  env.reset()
  env.step(_zero(env))
  r = env.scene["robot"]
  mi = _ids(r, robot.MOTOR_JOINT_NAMES)
  b = r.data.encoder_bias[:, mi]
  # zero action -> processed target = default (encoder frame) -> sent target = default - bias
  tgt = r.data.joint_pos_target[:, mi]
  assert torch.allclose(tgt, r.data.default_joint_pos[:, mi] - b, atol=1e-6)
  enc = env.observation_manager._group_obs_term_cfgs["actor"][2].func  # motor_pos  # noqa: SLF001
  raw = enc.raw(env, asset_cfg=_cfg(env, joint_names=robot.MOTOR_JOINT_NAMES))
  assert torch.allclose(raw, r.data.joint_pos[:, mi] + b - r.data.default_joint_pos[:, mi], atol=1e-6)


def test_lags(env):
  env.reset()
  for _ in range(3):
    env.step(_zero(env))
  act = events.get_motor(env)._delay_buffer.current_lags  # noqa: SLF001
  assert set(act.unique().tolist()) == set(range(robot.ACTUATOR.delay_max_lag + 1))
  first = act.clone()
  was_reset = torch.zeros_like(first, dtype=torch.bool)
  for _ in range(20):
    env.step(_zero(env))
    was_reset |= env.reset_buf
  assert ((act == first) | was_reset).all()  # held for the episode
  assert set(env.huphy_obs_lag.unique().tolist()) == set(range(OBS_MAX_LAG + 1))
  terms = {n: c.func for n, c in zip(env.observation_manager._group_obs_term_names["actor"],  # noqa: SLF001
                                     env.observation_manager._group_obs_term_cfgs["actor"], strict=True)}  # noqa: SLF001
  for n in ("base_ang_vel", "projected_gravity", "motor_pos"):
    assert isinstance(terms[n], observations._SharedLagTerm)  # noqa: SLF001


def test_shared_lag_term_unit():
  class Dummy(observations._SharedLagTerm):  # noqa: SLF001
    def raw(self, env):
      return env.value.clone()

  class E:
    num_envs, device = 3, "cpu"

  e = E()
  t = Dummy(None, e)
  e.huphy_obs_lag = torch.tensor([0, 1, 1])
  outs = []
  for k in range(4):
    e.value = torch.full((3, 2), float(k))
    if k == 2:
      t.reset(torch.tensor([2]))
    outs.append(t(e)[:, 0].tolist())
  assert outs[0] == [0, 0, 0]  # first frame fills the buffer
  assert outs[1] == [1, 0, 0]
  assert outs[2] == [2, 1, 2]  # env 2 was reset: no frame from before the reset
  assert outs[3] == [3, 2, 2]


def _no_fall_termination(cfg):
  cfg.terminations = {"time_out": cfg.terminations["time_out"]}


def test_push_force():
  env = make_env(256, edit=_no_fall_termination)
  env.reset()
  p = env.huphy_push
  r = env.scene["robot"]
  pel = int(r.indexing.body_ids[r.body_names.index("base_link")])
  assert (p.pause >= 2.0 - 1e-6).all() and (p.force == 0).all()  # first push only after a full pause
  seen, forces = 0, []
  for _ in range(400):  # 8 s
    env.step(_zero(env))
    xf = env.sim.data.xfrc_applied
    assert torch.allclose(xf[:, pel, :3], p.force, atol=1e-5)
    other = torch.cat((xf[:, :pel], xf[:, pel + 1 :]), dim=1)
    assert other.abs().max() == 0 and xf[:, pel, 2:].abs().max() == 0  # horizontal, pelvis only
    on = p.remaining > 0
    seen += int(on.sum())
    forces.append(p.force[on, :2].norm(dim=1))
  f = torch.cat(forces)
  assert f.max() <= PUSH["force_range"][1] + 1e-4 and f.mean() > 10
  frac = seen / (400 * env.num_envs)
  assert 0.02 < frac < 0.08, frac  # duration 0.15 s / (0.15 + 3.5 s) ~ 4 %


# --- actuator ---------------------------------------------------------------------------------

def test_single_torque_clip_and_tn(env):
  env.reset()
  m = events.get_motor(env)
  g = torch.Generator(device=env.device).manual_seed(1)
  for _ in range(30):
    env.step(torch.randn(env.num_envs, 12, device=env.device, generator=g) * 3.0)
    applied = m.applied_effort
    # MuJoCo never clips a second time: the <motor> force equals the T-N-clamped torque
    assert torch.allclose(env.sim.data.actuator_force[:, m.global_ctrl_ids], applied, atol=1e-4)
    lo, hi = m.bounds(env.scene["robot"].data.joint_vel[:, m.target_ids])
  sat = (m.requested_effort.abs() > m.applied_effort.abs() + 1e-3).float().mean()
  assert sat > 0.001  # large random targets do saturate
  assert (m.step_max_ratio >= 0).all()
  assert torch.allclose(m.peak, torch.tensor([120.09, 120.09, 59.68, 120.09, 59.68, 59.68] * 2, device=env.device))
  del lo, hi


def test_tn_quadrants():
  from huphy_squat.actuator import tn_limit

  curve = robot.tn_curve("RS04")
  w = torch.tensor([c[0] for c in curve])
  t = torch.tensor([c[1] for c in curve])
  assert float(tn_limit(w, t, torch.tensor([0.0]))) == pytest.approx(120.09)
  assert float(tn_limit(w, t, torch.tensor([200 * 2 * math.pi / 60]))) == 0.0
  mid = float(tn_limit(w, t, torch.tensor([150 * 2 * math.pi / 60])))
  assert 0 < mid < 120.09


# --- rewards ----------------------------------------------------------------------------------

def _set_pose_at_reference(env, h_star: float):
  """Write the squat-table pose for h* (pelvis upright, feet flat on the floor); h* = h_star."""
  r = env.scene["robot"]
  ref = rewards.SquatReference(env.device)
  hip, knee, ankle = ref(torch.full((env.num_envs,), h_star, device=env.device))
  q = r.data.joint_pos.clone()
  for s, sg in (("R", 1.0), ("L", -1.0)):
    q[:, r.joint_names.index(f"{s}_hip_pitch_joint")] = sg * hip
    q[:, r.joint_names.index(f"{s}_knee_joint")] = sg * knee
    q[:, r.joint_names.index(f"{s}_ankle_pitch_joint")] = ankle
    for j in ("hip_roll", "hip_yaw", "ankle_roll"):
      q[:, r.joint_names.index(f"{s}_{j}_joint")] = 0.0
  r.write_joint_state_to_sim(q, torch.zeros_like(q))
  env.sim.forward()
  st = torch.zeros(env.num_envs, 13, device=env.device)
  st[:, :7] = r.data.root_link_pose_w
  st[:, 2] -= sole_corners_z(env).min(dim=1).values  # lowest sole corner onto the floor
  r.write_root_state_to_sim(st)
  env.sim.forward()
  env.command_manager.get_term("height").target[:, 0] = h_star


def test_reward_reference_pose(env):
  env.reset()
  ones = torch.ones(env.num_envs, device=env.device)
  for h_star in (0.72, 0.80, 0.89):
    _set_pose_at_reference(env, h_star)
    assert torch.allclose(_eval_term(env, "squat_posture"), ones, atol=1e-6), h_star
    assert torch.allclose(_eval_term(env, "pitch_symmetry"), ones, atol=1e-6), h_star
    h = commands.base_height(env)
    assert (h - h_star).abs().max() < 0.01, (h_star, h[:4])  # table pose reaches h* (feet flat)
    assert (sole_corners_z(env).max(dim=1).values < 2e-3).all()  # feet flat
  env.command_manager.get_term("height").target[:, 0] = commands.base_height(env)
  assert torch.allclose(_eval_term(env, "track_height"), ones)
  assert _eval_term(env, "knee_guidance").abs().max() < 1e-6  # h == h*: gate 0


def test_reward_formulas(env):
  env.reset()
  env.sim.forward()
  r = env.scene["robot"]
  c = env.command_manager.get_term("height")
  h = commands.base_height(env)
  c.target[:, 0] = h - 0.05
  assert torch.allclose(_eval_term(env, "track_height"), torch.full_like(h, math.exp(-1.0)), atol=1e-5)
  st = torch.zeros(env.num_envs, 13, device=env.device)
  st[:, :7] = r.data.root_link_pose_w
  env.episode_length_buf[:] = 50  # past the 0.2 s grace window
  for vz, want in ((0.05, 0.0), (-0.1, 0.0), (0.3, 0.04), (-0.35, 0.0625)):
    st[:, 9] = vz
    r.write_root_state_to_sim(st)
    env.sim.forward()
    assert torch.allclose(_eval_term(env, "vz_speed"), torch.full_like(h, want), atol=1e-5), vz
  # clamp: |w f| <= 50 -> f <= 50 / 600
  st[:, 9] = -2.0
  r.write_root_state_to_sim(st)
  env.sim.forward()
  assert torch.allclose(_eval_term(env, "vz_speed"), torch.full_like(h, 50 / 600), atol=1e-6)


def test_knee_guidance_direction(env):
  env.reset()
  _set_pose_at_reference(env, 0.85)
  c = env.command_manager.get_term("height")
  c.target[:, 0] = 0.89  # going up: straightening the knees must lower the cost
  base = _eval_term(env, "knee_guidance").mean()
  r = env.scene["robot"]
  q = r.data.joint_pos.clone()
  for s, sg in (("R", 1.0), ("L", -1.0)):
    q[:, r.joint_names.index(f"{s}_knee_joint")] -= sg * 0.1
  r.write_joint_state_to_sim(q, torch.zeros_like(q))
  env.sim.forward()
  assert _eval_term(env, "knee_guidance").mean() < base


def test_reward_weights_registered(env):
  from huphy_squat.env_cfg import REWARD_WEIGHTS

  for name, w in REWARD_WEIGHTS.items():
    assert _term(env, name).weight == w
  assert env.cfg.scale_rewards_by_dt
