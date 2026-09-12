#!/usr/bin/env python3
"""Freeze the Pygmalion RP/AB runtime contract as machine-checkable JSON.

Run this in a fresh process for each ankle mode because the model-selection flags are
resolved at import time.  This is intentionally a one-environment CPU test: it validates
the policy boundary, not training throughput.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import mujoco

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
import mjlab.tasks  # noqa: F401


UPPER = {
    "waist_yaw_joint", "L_shoulder_pitch_joint", "R_shoulder_pitch_joint",
    "L_shoulder_roll_joint", "R_shoulder_roll_joint",
}


def expected(mode: str) -> list[str]:
    ankle = ("crank_A", "crank_B") if mode == "AB" else ("ankle_pitch", "ankle_roll")
    return [f"{side}_{joint}_joint" for side in "LR"
            for joint in ("hip_pitch", "hip_roll", "hip_yaw", "knee", *ankle)]


def main() -> None:
    mode = os.environ.get("PYG_ANKLE_MODE", "").upper()
    assert mode in ("AB", "RP")
    assert os.environ.get("PYG_STUDENT_TEACHER"), "student/teacher contract must be enabled"
    assert not os.environ.get("PYG_UPPER_DOF"), "welded upper body requires PYG_UPPER_DOF unset"

    cfg = load_env_cfg("Mjlab-Velocity-Flat-Pygmalion", play=True)
    cfg.scene.num_envs = 1
    cfg.events.pop("push_robot", None)
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    robot = env.scene["robot"]
    model: mujoco.MjModel = env.sim.mj_model
    action_term = env.action_manager.get_term("joint_pos")
    action = list(action_term._target_names)
    def first_row(value):
        if isinstance(value, (int, float)):
            return [float(value)] * len(action)
        return [float(x) for x in value.detach().cpu().reshape(-1, len(action))[0]]
    action_scale = first_row(action_term.scale)
    action_offset = first_row(action_term.offset)
    exp = expected(mode)
    joints = list(robot.joint_names)
    actor_terms = list(env.observation_manager.active_terms["actor"])
    actor_term_dims = [list(x) for x in env.observation_manager._group_obs_term_dim["actor"]]
    body_names = [model.body(i).name for i in range(model.nbody)]
    upper_mass = sum(float(model.body_mass[i]) for i, name in enumerate(body_names)
                     if name.endswith("torso_link") or "shoulder" in name or "arm_link" in name)
    eq_names = [model.equality(i).name for i in range(model.neq)]
    expected_mass = float(os.environ.get("PYG_EXPECTED_TOTAL_MASS_KG", "35.707"))

    checks = {
        "action_dim_12": env.action_manager.total_action_dim == 12,
        "action_order_exact": action == exp,
        "upper_not_actuated": not (set(action) & UPPER),
        "upper_joints_welded": not (set(joints) & UPPER),
        "upper_mass_retained": upper_mass > 10.0,
        "actor_terms_exact": actor_terms == [
            "base_ang_vel", "projected_gravity", "motor_pos_history", "actions", "command"
        ],
        "actor_dim_45": env.observation_manager.group_obs_dim["actor"] == (45,),
        "action_scale_0_25rad": all(abs(x - 0.25) < 1e-8 for x in action_scale),
        "policy_period_20ms": abs(float(env.step_dt) - 0.02) < 1e-9,
        "closed_loop_equalities_4": (model.neq == 4) if mode == "AB" else (model.neq == 0),
        "total_mass_matches_expected": abs(float(model.body_mass.sum()) - expected_mass) < 0.002,
    }
    report = {
        "schema": "pygmalion-runtime-contract-v1",
        "ankle_mode": mode,
        "model_selector": os.environ.get("PYG_MODEL_TAG", os.environ.get("PYG_V2_XML")),
        "total_mass_kg": round(float(model.body_mass.sum()), 6),
        "expected_total_mass_kg": expected_mass,
        "upper_retained_mass_kg": round(upper_mass, 6),
        "model_counts": {"body": model.nbody, "joint": model.njnt, "actuator": model.nu,
                         "equality": model.neq},
        "action_names": action,
        "expected_action_names": exp,
        "action": {
            "type": "position target = default_offset + raw_policy_action * scale",
            "scale_rad": action_scale,
            "default_offset_rad": action_offset,
            "raw_action_clip": action_term.cfg.clip,
            "warning": "runner clip_actions is not captured here; current Pygmalion runner leaves it unset",
        },
        "timing": {
            "physics_dt_s": float(model.opt.timestep),
            "policy_dt_s": float(env.step_dt),
            "decimation": round(float(env.step_dt) / float(model.opt.timestep)),
            "action_hold": "zero-order hold for one policy period",
        },
        "actor": {
            "terms": actor_terms,
            "term_dims": actor_term_dims,
            "group_dim": list(env.observation_manager.group_obs_dim["actor"]),
            "meaning": {
                "base_ang_vel": "raw-deployable EBIMU gyro, 3",
                "projected_gravity": "derived from EBIMU attitude, 3",
                "motor_pos_history": "q(t-1), q(t) for exactly 12 motor encoders, 24",
                "actions": "previous 12 motor commands",
                "command": "vx, vy, yaw-rate command, 3",
            },
        },
        "critic_dim": list(env.observation_manager.group_obs_dim["critic"]),
        "entity_joint_names": joints,
        "equality_names": eq_names,
        "checks": checks,
        "verdict": "PASS" if all(checks.values()) else "FAIL",
    }
    output = Path(os.environ["PYG_CONTRACT_OUT"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    env.close()
    print(json.dumps(report, ensure_ascii=False, indent=2))
    assert report["verdict"] == "PASS", checks


if __name__ == "__main__":
    main()
