import re, json, numpy as np, trimesh, mujoco, sys
sys.path.insert(0,'/home/syaro/MikuchanRemote/Human-Pygmalion/tools/robot_model/foot_overlay')
R='/home/syaro/MikuchanRemote/Human-Pygmalion/'
U=R+'pygmalion_locomotion/assets/pygmalion_v2/'
NAME='LegOnly_prototype-tempmass-motormeasured-armfix_v30_proxyfix'
urdf=open(U+NAME+'_footv5_loop.urdf').read()
base=open(U+NAME+'_loop.urdf').read()
mj=open(R+'Feet_design/Feet_Design_Workspace/mjcf_sim/union_branch/mjlab_policy/pygmalion_mjlab/assets/pygmalion_legonly_ab_v5fullm.xml').read()
def pin(blk):
    o=np.array(re.search(r'<origin xyz="([^"]+)"',blk).group(1).split(),float)
    m=float(re.search(r'<mass value="([^"]+)"',blk).group(1))
    I={k:float(v) for k,v in re.findall(r'(i[xyz]{2})="([^"]+)"',blk)}
    T=np.array([[I['ixx'],I['ixy'],I['ixz']],[I['ixy'],I['iyy'],I['iyz']],[I['ixz'],I['iyz'],I['izz']]])
    return m,o,T
def link(t,S): return re.search(rf'  <link name="{S}_foot_link">.*?</link>\n',t,re.S).group(0)
ok=lambda c:'PASS' if c else 'FAIL'
rows=[]
def chk(name,val,ref,tol,unit=''):
    d=np.max(np.abs(np.asarray(val)-np.asarray(ref)))
    rows.append((name,d<=tol,f'{d:.3g}{unit} (tol {tol})')); print(ok(d<=tol),name,f'diff={d:.3g}{unit} tol={tol}')
# STEP truth (from step_props.py, plate frame) -> foot frame (R): x=145-y, y=x+123.7, z=z+800
step={'carrier':(160.12118169,[-124.62122959,97.31572220,-848.13862859]),
      'tape':(26.66428743,[-123.02277152,91.12716272,-851.72375166]),
      'sheet':(106.64583433,[-123.02300271,90.12161461+1.0,-854.23284492]),  # placeholder fixed below
      'feet_up':(488.36774435,[-123.70012924,42.79775205,-816.09765603])}
step['sheet']=(106.64583433,[-123.02300271,91.12161461,-854.23284492])
f=lambda p:np.array([145-p[1],p[0]+123.7,p[2]+800])
mesh={}
for p,(v,c) in step.items():
    m=trimesh.load(U+f'meshes_footv5/R_v5_{p}.stl')
    wt=m.is_watertight
    mesh[p]=m
    chk(f'{p}: STL volume vs STEP [cm3]',m.volume*1e6,v,0.1*v/100,' cm3')
    chk(f'{p}: STL COM vs STEP COM, ankle frame [mm]',m.center_mass*1000 if wt else m.centroid*1000,f(c),0.3,' mm')
    print('   watertight',wt,'euler',m.euler_number, 'bodies',len(m.split(only_watertight=False)))
# recombine
b=pin(link(base,'R')); n=pin(link(urdf,'R')); nl=pin(link(urdf,'L'))
dens={'carrier':887.4*1.04/1.24,'tape':720.,'sheet':1150.,'feet_up':None}
items=[b]
for p,m in mesh.items():
    mm=m.copy()
    if not mm.is_watertight: mm.merge_vertices(); trimesh.repair.fill_holes(mm)
    rho=dens[p] if dens[p] else 0.080/mm.volume
    mm.density=rho
    items.append((mm.mass,mm.center_mass.copy(),mm.moment_inertia.copy()))
M=sum(i[0] for i in items); c=sum(i[0]*i[1] for i in items)/M
T=sum(I+m*(np.dot(p-c,p-c)*np.eye(3)-np.outer(p-c,p-c)) for m,p,I in items)
chk('recombined mass vs URDF R [kg]',M,n[0],1e-5,' kg'); chk('recombined COM vs URDF R [mm]',c*1000,n[1]*1000,0.01,' mm')
chk('recombined inertia vs URDF R [rel]',T/np.abs(n[2]).max(),n[2]/np.abs(n[2]).max(),2e-4)
print('R foot: base mass',b[0],'new',n[0],'add',n[0]-b[0],'; parts masses',[round(i[0],5) for i in items[1:]])
# mirror
S=np.diag([1,-1,1]); chk('L inertia = mirror(R) [kg m2]',nl[2],S@n[2]@S,1e-9); chk('L COM = mirror(R) [mm]',nl[1]*1000,(S@n[1])*1000,1e-4,' mm'); chk('L mass = R mass',nl[0],n[0],1e-9)
chk('baseline foot COM mirror (original model)',pin(link(base,'L'))[1],S@pin(link(base,'R'))[1],1e-9)
print('principal moments URDF R',np.linalg.eigvalsh(n[2]),'triangle',(lambda w:(w[0]+w[1]>=w[2]))(np.linalg.eigvalsh(n[2])))
# MJCF v5fullm
for Sd in 'RL':
    mjb=re.search(rf'<body name="{Sd}_foot_link"[^>]*>(.*?)(?=<geom)',mj,re.S)
    ln=[l for l in mj.splitlines() if 'mass="1.18617"' in l]
idx=0
for l,Sd in zip([l for l in mj.splitlines() if 'mass="1.18617"' in l],'RL'):
    pos=np.array(re.search(r'pos="([^"]+)"',l).group(1).split(),float); m=float(re.search(r'mass="([^"]+)"',l).group(1))
    fi=np.array(re.search(r'fullinertia="([^"]+)"',l).group(1).split(),float)
    T2=np.array([[fi[0],fi[3],fi[4]],[fi[3],fi[1],fi[5]],[fi[4],fi[5],fi[2]]])
    u=pin(link(urdf,Sd))
    chk(f'MJCF v5fullm vs URDF mass {Sd}',m,u[0],1e-5,' kg'); chk(f'MJCF vs URDF COM {Sd} [mm]',pos*1000,u[1]*1000,0.001,' mm')
    chk(f'MJCF vs URDF inertia {Sd} [rel]',T2/np.abs(u[2]).max(),u[2]/np.abs(u[2]).max(),2e-3)
# MuJoCo load
def load(path,urdfmode):
    t=open(path).read()
    if urdfmode:
        i=t.index('>',t.index('<robot'))+1
        t=t[:i]+f'\n<mujoco><compiler discardvisual="false" meshdir="{U}"/></mujoco>'+t[i:]
    return mujoco.MjModel.from_xml_string(t) if urdfmode else mujoco.MjModel.from_xml_path(path)
mu=load(U+NAME+'_footv5_loop.urdf',True); mo=load(U+NAME+'_loop.urdf',True)
for Sd in 'RL':
    b1=mu.body(f'{Sd}_foot_link'); b0=mo.body(f'{Sd}_foot_link')
    print(Sd,'mass',b1.mass[0],'base',b0.mass[0],'ipos',b1.ipos*1000)
print('total mass new/base',mu.body_mass.sum(),mo.body_mass.sum())
json.dump([(a,bool(b),c) for a,b,c in rows],open(sys.argv[1] if len(sys.argv)>1 else '/dev/null','w'),ensure_ascii=False,indent=1)
