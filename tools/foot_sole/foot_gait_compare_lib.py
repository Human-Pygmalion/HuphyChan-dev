"""Shared helpers for the foot gait scripts."""
import mujoco
import numpy as np


def quat_mat(q):
  w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
  return np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                   2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                   2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1).reshape(q.shape[:-1] + (3, 3))


def foot_geoms(m, body):
  """List of (vertices in foot frame [m], kind) for every collision geom of the foot."""
  out = []
  b = m.body(body).id
  for g in range(m.ngeom):
    if m.geom_bodyid[g] != b or m.geom_group[g] != 3:
      continue
    mat = np.zeros(9)
    mujoco.mju_quat2Mat(mat, m.geom_quat[g])
    R, p = mat.reshape(3, 3), m.geom_pos[g]
    t = m.geom_type[g]
    if t == mujoco.mjtGeom.mjGEOM_BOX:
      s = m.geom_size[g]
      v = np.array([[sx, sy, sz] for sx in (-s[0], s[0]) for sy in (-s[1], s[1]) for sz in (-s[2], s[2])])
    elif t == mujoco.mjtGeom.mjGEOM_MESH:
      mid = m.geom_dataid[g]
      v = m.mesh_vert[m.mesh_vertadr[mid]: m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
    elif t == mujoco.mjtGeom.mjGEOM_SPHERE:
      r = m.geom_size[g][0]
      a = np.linspace(0, 2 * np.pi, 13)[:-1]
      v = np.array([[r * np.cos(u), 0, r * np.sin(u)] for u in a])
    else:
      continue
    v = v @ R.T + p
    kind = "upper" if v[:, 2].min() > -0.044 else "sole"
    out.append((v, kind))
  return out


def foot_vertices(m, body):
  return np.concatenate([g for g, _ in foot_geoms(m, body)])
