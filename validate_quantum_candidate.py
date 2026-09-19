#!/usr/bin/env python3
"""Numerical diagnostics for the proposed boundary-query problem (NumPy).
No quantum circuit, query lower bound, or conjecture is verified by this script.
Run: python validate_quantum_candidate.py
"""
import itertools
import math
import numpy as np
TOL=2e-10

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

def run():
    count=0
    def test(name,ok):
        nonlocal count
        if not ok:raise AssertionError(name)
        count+=1;print('[PASS numerical]',name)
    m,E,marked=family();u=1/3
    test('specified graph is cubic and has one marked edge per triangle',len(E)==9*m//2 and all(sum(v in e for e in E)==3 for v in range(3*m)) and len(set(marked))==m)
    values={}
    for bits in itertools.product([0,1],repeat=m):
        W,f,_=matrix(bits,u);values[bits]=f
        eig=np.linalg.eigvalsh(W)
        test('spectral bounds bits='+''.join(map(str,bits)),eig[0]>=.25-TOL and eig[-1]<=2.5+TOL)
    c0=2*math.log((1-u**4)/(1-u**3));diffs=[]
    for bits in values:
        for i in range(m):
            if bits[i]==0:
                changed=list(bits);changed[i]=1
                drop=m*(values[bits]-values[tuple(changed)]);diffs.append(drop)
    test('all 192 one-bit influences lie in the proved interval',min(diffs)>=c0-TOL and max(diffs)<=2+TOL)
    print(f'Influence range after multiplying by m: [{min(diffs):.12g}, {max(diffs):.12g}]; lower bound c0={c0:.12g}')
    # Explicit even polynomial for log(H), H=W/beta, beta=8.
    beta=8.;a=.25;z=1-(a/beta)**2;eta=1e-6
    d=1
    while z**(d+1)/(2*(d+1)*(1-z))>eta:d+=1
    tail=z**(d+1)/(2*(d+1)*(1-z));harm=sum(1/j for j in range(1,d+1))
    grid=np.linspace(-1,1,1001);term=np.ones_like(grid);poly=np.zeros_like(grid)
    for j in range(1,d+1):term*=1-grid*grid;poly-=term/(2*j)
    test('normalized polynomial has even parity and is bounded on sampled grid',np.max(np.abs(poly/harm))<=.5+TOL and np.max(np.abs(poly-poly[::-1]))<TOL)
    for bits in [(0,)*m,(1,)*m,tuple(i%2 for i in range(m))]:
        W,f,scalar=matrix(bits,u);n=len(W);H=W/beta
        R=np.eye(n)-H@H;power=np.eye(n);P=np.zeros_like(H)
        for j in range(1,d+1):power=power@R;P-=power/(2*j)
        estimate=-(scalar+n*math.log(beta)+np.trace(P))/m
        test('trace polynomial error fits proved tail for '+str(bits),abs(estimate-f)<=3*tail+TOL)
        size=1<<(n-1).bit_length();Hp=np.eye(size);Hp[:n,:n]=H
        eig,V=np.linalg.eigh(Hp);root=(V*np.sqrt(np.maximum(0,1-eig*eig)))@V.T
        U=np.block([[Hp,root],[root,-Hp]])
        test('qubit-sized dense dilation unitary within tolerance for '+str(bits),np.linalg.norm(U.T@U-np.eye(2*size),ord=2)<TOL)
        test('dense dilation corner equals padded H for '+str(bits),np.array_equal(U[:size,:size],Hp))
    print(f'Even polynomial degree={2*d}; scalar tail <= {tail:.6g}; normalized log-zeta error <= {3*tail:.6g}')
    for u in [.4,.49,.4999]:
        for bits in [(0,)*m,(1,)*m,tuple(i%2 for i in range(m))]:
            W,f,_=matrix(bits,u);e=np.linalg.eigvalsh(W);a=(1-2*u)/(1+u)
            test(f'near-pole SPD diagnostic u={u}, bits={bits}',e[0]>=a-TOL and e[-1]<=1+3*u/(1-u)+TOL)
        print(f'Near-pole u={u}: delta={1-2*u:.6g}; conservative polynomial gap a/8={a/8:.6g}')
    print(f'ALL {count} NUMERICAL CHECKS PASS (tolerance {TOL:g})')
    print('NOT VERIFIED: QSVT phases/circuit synthesis, oracle query counts, fault tolerance, near-critical conjecture, originality.')
    print('Dense dilation is a numerical identity check and is not an efficient oracle implementation.')

if __name__=='__main__':run()
