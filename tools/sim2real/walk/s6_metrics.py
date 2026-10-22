#!/usr/bin/env python3
"""Metrics for the stage-6 5 Hz study, one code path for the real log and the simulator runs
(docs/sim2real/2026-10-22_stage6_5hz_reproduction.md).

A record is a dict of arrays sampled at the (non-uniform) policy-call times t [s]:
  tgt, pos, act : (N, 12) in deg / deg / raw policy units, joint order R hip_pitch, hip_roll, hip_yaw, knee, ankle_a, ankle_b, then L.
  gyro (N,3) rad/s, tilt (N,) deg (angle between body z and vertical).

Definitions (identical for real and sim):
  band signal    linear interpolation onto a uniform 100 Hz grid, mean removed over the window, Butterworth 4th order 3-8 Hz band-pass (zero phase)
  band power     mean over the 12 joints of the band-signal variance [deg^2];  band amp = sqrt(2 * band power) = sinusoid amplitude of the rms
  LS peak        Lomb-Scargle (raw times, no resampling) peak frequency in 1-9 Hz of the sum of the normalised spectra of the R and L joint of one type
  fit            least-squares sinusoid at the LS peak of the target of that joint: amplitude of tgt and pos, lag = phase(tgt) - phase(pos) / (2 pi f)
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, lombscargle, sosfiltfilt

JT = ["hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_a", "ankle_b"]
JIDX = {"hip_pitch": (0, 6), "hip_roll": (1, 7), "knee": (3, 9)}
FS = 100.0
_SOS = butter(4, [3.0, 8.0], btype="band", fs=FS, output="sos")
_FR = np.arange(1.0, 9.0001, 0.02)


def uniform(t, y, t0, t1):
  g = np.arange(t0, t1, 1.0 / FS)
  y = np.atleast_2d(y.T).T if y.ndim == 1 else y
  return g, np.stack([np.interp(g, t, y[:, j]) for j in range(y.shape[1])], 1)


def band_signal(t, y, t0, t1):
  """band-passed signal of y (N,J) on the 100 Hz grid of [t0,t1); the filter runs on the window plus 1 s margin each side."""
  a, b = max(t0 - 1.0, t[0]), min(t1 + 1.0, t[-1])
  g, u = uniform(t, y, a, b)
  u = u - u.mean(0)
  f = sosfiltfilt(_SOS, u, axis=0)
  m = (g >= t0) & (g < t1)
  return g[m], f[m]


def band_power(t, y, t0, t1):
  """mean over columns of the variance of the 3-8 Hz band signal [unit^2]; NaN if the window has < 1 s of data"""
  if t1 - t0 < 1.0 or ((t >= t0) & (t < t1)).sum() < 8:
    return float("nan")
  _, f = band_signal(t, y, t0, t1)
  return float((f ** 2).mean())


def ls_spec(t, y, fr=_FR):
  y = y - y.mean()
  p = lombscargle(t, y, 2 * np.pi * fr, normalize=True)
  return p


def ls_peak(t, ys, t0, t1):
  """peak frequency [Hz] and normalised height of the summed LS spectrum of the columns ys (N,J) in [t0,t1)"""
  m = (t >= t0) & (t < t1) & np.all(np.isfinite(ys), 1)
  if m.sum() < 16:
    return float("nan"), float("nan")
  s = sum(ls_spec(t[m], ys[m][:, j]) for j in range(ys.shape[1])) / ys.shape[1]
  i = int(np.argmax(s))
  return float(_FR[i]), float(s[i])


def sine_fit(t, y, f):
  """amplitude and phase of y ~ a cos + b sin at frequency f (least squares, mean removed)"""
  X = np.stack([np.cos(2 * np.pi * f * t), np.sin(2 * np.pi * f * t), np.ones_like(t)], 1)
  c, *_ = np.linalg.lstsq(X, y, rcond=None)
  return float(np.hypot(c[0], c[1])), float(np.arctan2(c[1], c[0]))


def peak_fit(rec, t0, t1, joint="hip_pitch"):
  """at the LS peak (1-9 Hz) of the target of `joint` (R+L): f, tgt amp, pos amp, ratio, lag [ms] (pos behind tgt = positive)"""
  t = rec["t"]
  i0, i1 = JIDX[joint]
  m = (t >= t0) & (t < t1) & np.all(np.isfinite(rec["tgt"][:, [i0, i1]]), 1)
  if m.sum() < 16:
    return None
  f, _ = ls_peak(t, rec["tgt"][:, [i0, i1]], t0, t1)
  out = []
  for j in (i0, i1):
    sgn = 1.0
    at, pt = sine_fit(t[m], rec["tgt"][m, j], f)
    ap, pp = sine_fit(t[m], rec["pos"][m, j], f)
    lag = ((pp - pt + np.pi) % (2 * np.pi) - np.pi) / (2 * np.pi * f) * 1000.0
    out.append((at, ap, lag))
  at, ap, lag = np.mean(out, 0)
  return dict(f=f, tgt_amp=float(at), pos_amp=float(ap), ratio=float(ap / at) if at > 0 else float("nan"), lag_ms=float(lag))


def sliding_band_amp(rec, key, t0, t1, win=2.0, hop=0.5):
  """band amplitude (sqrt(2 * power), deg or raw) of the 12-joint mean in sliding windows -> (centres, amps)"""
  t = rec["t"]
  cs, am = [], []
  a = t0
  while a + win <= t1 + 1e-9:
    p = band_power(t, rec[key], a, a + win)
    cs.append(a + win / 2)
    am.append(np.sqrt(2 * p) if np.isfinite(p) else np.nan)
    a += hop
  return np.array(cs), np.array(am)


def decay_time(cs, am, t_start, thr, sustained_last=5.0):
  """time after t_start until the amplitude stays below thr for the rest of the record; inf if it never does ('sustained'),
  0 if it was never above thr after t_start."""
  m = cs >= t_start
  c, a = cs[m], am[m]
  ok = np.isfinite(a)
  c, a = c[ok], a[ok]
  if len(a) == 0:
    return float("nan")
  above = a > thr
  if not above.any():
    return 0.0
  last = np.where(above)[0].max()
  if last >= len(a) - 1:
    return float("inf")
  return float(c[last + 1] - t_start)
