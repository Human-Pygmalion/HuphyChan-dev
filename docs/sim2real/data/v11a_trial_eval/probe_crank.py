import os, sys
sys.path.insert(0, "analysis")
import mjlab.tasks
from mjlab.tasks.registry import load_env_cfg
from mjlab.envs import ManagerBasedRlEnv
cfg = load_env_cfg("Mjlab-Velocity-Flat-Pygmalion", play=True)
cfg.scene.num_envs = 1
env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
m = env.unwrapped.sim.mj_model if hasattr(env,"unwrapped") else env.sim.mj_model
import mujoco, numpy as np
print("PYG_CRANK_KP=", os.environ.get("PYG_CRANK_KP"), "KD=", os.environ.get("PYG_CRANK_KD"))
for i in range(m.nu):
    n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
    print("ACT", n, m.actuator_gainprm[i,0], m.actuator_biasprm[i,:3])
ent = env.unwrapped.scene["robot"] if hasattr(env,"unwrapped") else env.scene["robot"]
for a in ent.actuators if hasattr(ent,"actuators") else []:
    print("ACTUATOR", type(a).__name__, getattr(a,"joint_names",None))
    for k in ("stiffness","damping"):
        v = getattr(a,k,None)
        if v is not None: print("  ",k, v.flatten().tolist()[:6])
os._exit(0)
