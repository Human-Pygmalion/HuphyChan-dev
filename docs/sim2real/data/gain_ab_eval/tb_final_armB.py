import numpy as np, json
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
B="logs/rsl_rl/pygmalion_velocity/"
runs={"armB_resumed":B+"2026-10-17_02-09-17_rough_gain_mountain_v11a"}
keys=["Train/mean_reward","Train/mean_episode_length","Episode_Termination/fell_over","Episode_Termination/low_base","Episode_Termination/stagnation",
"Episode_Reward/track_linear_velocity","Episode_Reward/track_angular_velocity","Metrics/twist/error_vel_xy_steady_moving","Metrics/twist/error_ang_vel_steady_moving","Curriculum/dr_levels/dr_factor"]
out={}
for r,p in runs.items():
    acc=EventAccumulator(p,size_guidance={"scalars":0}); acc.Reload()
    tags=acc.Tags()["scalars"]
    out[r]={}
    for k in keys:
        if k not in tags: out[r][k]=None; continue
        ev=acc.Scalars(k); s=np.array([e.step for e in ev]); v=np.array([e.value for e in ev])
        m=s>=s.max()-199
        out[r][k]={"last_step":int(s.max()),"final200":float(v[m].mean()),"n":int(m.sum()),"last":float(v[-1])}
    print(r, "max step", max(out[r][k]["last_step"] for k in keys if out[r][k]))
for k in keys:
    print(k.ljust(48), *(f"{out[r][k]['final200']:9.4f}" if out[r][k] else "      nan" for r in runs))
json.dump(out,open("/home/syaro/v11a_eval/gain_ab/tb_final_armB.json","w"),indent=1)
