"""Load a real HUPHY policy log (Expr_Data/*.csv) as a stage-6 record (see s6_metrics.py)."""
import numpy as np, pandas as pd

J = ["hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_a", "ankle_b"]
JN = [f"{l}/{j}" for l in ("right_leg", "left_leg") for j in J]


def load_real(path):
  with open(path, encoding="utf-8") as f:
    skip = 1 if f.readline().startswith("#") else 0
  df = pd.read_csv(path, skiprows=skip)
  c = lambda p: df[[f"{p}/{n}" for n in JN]].to_numpy()  # noqa: E731
  g = df[["obs/grav/x", "obs/grav/y", "obs/grav/z"]].to_numpy()
  gyro = df[["obs/gyro/x", "obs/gyro/y", "obs/gyro/z"]].to_numpy()
  return dict(t=df.t.to_numpy() - df.t.iloc[0], tgt=c("tgt"), pos=c("pos"), act=c("act"), gyro=gyro,
              tilt=np.degrees(np.arccos(np.clip(-g[:, 2] / np.linalg.norm(g, axis=1), -1, 1))), grav=g)
