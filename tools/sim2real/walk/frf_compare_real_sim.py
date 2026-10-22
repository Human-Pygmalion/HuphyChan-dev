#!/usr/bin/env python3
"""tgt->pos frequency response (H = Pxy/Pxx, coherence) per joint: real log vs sim CSV export, same method.
Closed-loop estimate (the policy closes the loop), so differences mix plant and excitation; use it as a screen.
Run from the repo root: mujoco-sim/mjlab/.venv/bin/python3 tools/sim2real/walk/frf_compare_real_sim.py"""
import numpy as np,io,glob,sys
from scipy.signal import csd,welch
J=['hip_pitch','hip_roll','hip_yaw','knee','ankle_a','ankle_b']
def load(f):
    L=[l for l in open(f,encoding='utf-8').read().splitlines() if not l.startswith('#')]
    c=L[0].split(','); a=np.genfromtxt(io.StringIO('\n'.join(L[1:])),delimiter=','); return c,a
FS=20.0
def frf(files):
    out={}
    for leg in ['right_leg','left_leg']:
      for j in J:
        Pxy=Pxx=Pyy=0; n=0
        for f in files:
            c,a=load(f); t=a[:,0]; tu=np.arange(t[0],t[-1],1/FS)
            x=np.interp(tu,t,a[:,c.index(f'tgt/{leg}/{j}')]); y=np.interp(tu,t,a[:,c.index(f'pos/{leg}/{j}')])
            x-=x.mean(); y-=y.mean()
            fr,sxy=csd(x,y,fs=FS,nperseg=128); _,sxx=welch(x,fs=FS,nperseg=128); _,syy=welch(y,fs=FS,nperseg=128)
            Pxy=Pxy+sxy; Pxx=Pxx+sxx; Pyy=Pyy+syy
        H=Pxy/Pxx; coh=np.abs(Pxy)**2/(Pxx*Pyy)
        out[f'{leg[0].upper()}_{j}']=(fr,H,coh)
    return out
real=frf(['Expr_Data/1021vz0.5ob.csv'])
sim=frf(sorted(glob.glob('mujoco-sim/mjlab/analysis/out/s2r_walk/csv/W5_real_env*.csv')))
fq=[0.6,1.3,2.0,3.0,4.0,5.0]
print("tgt->pos FRF  |H| (phase deg) [coh]   real wz0.5 vs sim W5_real (8 envs), fs 20 Hz uniform interp, nperseg 128")
print(f"{'joint':14s}"+''.join(f"{q:>26.1f}Hz" for q in fq))
for k in real:
    fr,Hr,cr=real[k]; _,Hs,cs=sim[k]
    row_r=''; row_s=''
    for q in fq:
        i=np.argmin(abs(fr-q))
        row_r+=f"{abs(Hr[i]):6.2f}({np.degrees(np.angle(Hr[i])):5.0f})[{cr[i]:.2f}]".rjust(28)
        row_s+=f"{abs(Hs[i]):6.2f}({np.degrees(np.angle(Hs[i])):5.0f})[{cs[i]:.2f}]".rjust(28)
    print(f"{k+' real':14s}{row_r}"); print(f"{k+' sim':14s}{row_s}")
