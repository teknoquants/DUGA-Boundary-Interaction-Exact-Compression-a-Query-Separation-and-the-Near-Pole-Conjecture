#!/usr/bin/env python3
"""Near-pole research diagnostics; Python 3 + NumPy.
Finite checks support the preliminary propositions, not the conjecture.
No quantum circuit is implemented and no asymptotic speedup is measured.
"""
import math
import itertools
from fractions import Fraction as F
import numpy as np
def family(k=3):
    # Prism C_k x K_2 has m=2k vertices, each replaced by a triangle.
    if k<3:raise ValueError('k >= 3')
    m=2*k
    base=sorted({tuple(sorted(e)) for layer in range(2) for i in range(k) for e in [(layer*k+i,layer*k+(i+1)%k)]} | {(i,k+i) for i in range(k)})
    neighbors=[sorted(b if a==v else a for a,b in base if v in (a,b)) for v in range(m)]
    edges=[];marked=[]
    for v in range(m):
        marked.append(len(edges));edges.extend([(3*v,3*v+1),(3*v+1,3*v+2),(3*v+2,3*v)])
    for a,b in base:edges.append((3*a+neighbors[a].index(b),3*b+neighbors[b].index(a)))
    return m,edges,marked

def matrix(bits,u,k=3):
    m,E,marked=family(k);lengths=np.ones(len(E),dtype=int)
    for i,e in enumerate(marked):lengths[e]+=bits[i]
    W=np.eye(3*m);scalar=0.0
    for (a,b),L in zip(E,lengths):
        t=u**int(L);q=1-t*t
        W[a,a]+=t*t/q;W[b,b]+=t*t/q
        W[a,b]-=t/q;W[b,a]-=t/q
        scalar+=math.log1p(-t*t)
    sign,ld=np.linalg.slogdet(W)
    if sign<=0:raise ArithmeticError('outside tested positive domain')
    return W,-(scalar+ld)/m,scalar
def degree_bound(u, eta):
    a=(1-2*u)/(1+u); gap2=(a/8)**2
    # Sufficient bound; log1p avoids catastrophic cancellation in 1-gap2.
    return max(1,math.ceil(math.log(1/(2*gap2*eta))/(-math.log1p(-gap2))))

def run():
    exact=numeric=0
    def check(name,ok,kind='numerical'):
        nonlocal exact,numeric
        if not ok:raise AssertionError(name)
        if kind=='exact':exact+=1
        else:numeric+=1
        print(f'[PASS {kind}] {name}')
    m,E,marked=family(3);n=3*m;tol=2e-10
    check('known skeleton is cubic',all(sum(v in e for e in E)==3 for v in range(n)),'exact')
    for uq in [F(1,3),F(49,100),F(4999,10000)]:
        a=(1-2*uq)/(1+uq);beta0=1+3*uq/(1-uq)
        for bits in [(0,)*m,(1,)*m,tuple(i%2 for i in range(m))]:
            lengths=[1]*len(E)
            for i,e in enumerate(marked):lengths[e]+=bits[i]
            rows=[F(1)]*n;tops=[F(1)]*n;scalar=F(1)
            for (v,w),L in zip(E,lengths):
                t=uq**L;scalar*=1-t*t
                for j in [v,w]:rows[j]-=t/(1+t);tops[j]+=t/(1-t)
            check(f'u={uq}, bits={bits}: rational spectral enclosure',min(rows)>=a>0 and max(tops)<=beta0,'exact')
            if not any(bits):check(f'u={uq}: constant eigenvector has exact eigenvalue a',all(r==a for r in rows),'exact')
            W,value,_=matrix(bits,float(uq));ev=np.linalg.eigvalsh(W)
            check(f'u={uq}, bits={bits}: numerical spectrum agrees',ev[0]>=float(a)-tol and ev[-1]<=float(beta0)+tol)
            check(f'u={uq}, bits={bits}: finite normalized log zeta',math.isfinite(value) and value>=-tol)
    # Enumerate the smallest family to test monotonicity, not query hardness.
    u=.49;vals={b:matrix(b,u)[1] for b in itertools.product([0,1],repeat=m)}
    c0=2*math.log((1-u**4)/(1-u**3))
    drops=[]
    for b in vals:
        for i in range(m):
            if not b[i]:
                c=list(b);c[i]=1;drops.append(m*(vals[b]-vals[tuple(c)]))
    check('all 192 one-bit changes obey triangle lower bound near pole',min(drops)>=c0-tol)
    print(f'Near-pole influence range times m: {min(drops):.9g} .. {max(drops):.9g}')
    # Numerically evaluate the polynomial at a feasible baseline degree.
    eta=1e-6;u=1/3;d=degree_bound(u,eta)
    W,f,scalar=matrix(tuple(i%2 for i in range(m)),u)
    H=W/8;R=np.eye(n)-H@H;power=np.eye(n);P=np.zeros_like(H)
    for j in range(1,d+1):power=power@R;P-=power/(2*j)
    approx=-(scalar+n*math.log(8)+np.trace(P))/m
    gap2=(((1-2*u)/(1+u))/8)**2
    tail=math.exp((d+1)*math.log1p(-gap2))/(2*(d+1)*gap2)
    check('baseline dense polynomial agrees within proven trace tail',abs(approx-f)<=3*tail+tol)
    print(f'Executed baseline polynomial: d={d}, degree={2*d}, trace error={abs(approx-f):.6g}, bound={3*tail:.6g}')
    print('Conservative sufficient degrees for eta=1e-6 (NOT executed near the pole):')
    for u in [1/3,.45,.49,.499,.4999]:
        print(f'  u={u:.8g}, delta={1-2*u:.8g}, polynomial degree={2*degree_bound(u,eta)}')
    # Test the proved simple-pole asymptotic on the known zero-bit input.
    residual=[]
    for delta in [1e-4,1e-5,1e-6]:
        value=matrix((0,)*m,(1-delta)/2)[1]
        residual.append(value+math.log(delta)/m)
    check('zero-input simple-pole residual stabilizes numerically',abs(residual[-1]-residual[-2])<abs(residual[-2]-residual[-3]) and abs(residual[-1]-residual[-2])<1e-3)
    print('Simple-pole residual F_0 + log(delta)/m:',residual)
    print(f'ALL CHECKS PASS: {exact} exact, {numeric} numerical; numerical tolerance={tol:g}')
    print('NOT VERIFIED: the near-pole query conjecture, QSVT circuits, DS-NAQHA realization, or originality.')
    print('Degree bounds are conservative analytic upper bounds for one method, not algorithmic lower bounds.')

if __name__=='__main__':run()
