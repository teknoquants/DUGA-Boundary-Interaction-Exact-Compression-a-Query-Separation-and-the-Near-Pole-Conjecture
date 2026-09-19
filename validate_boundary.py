#!/usr/bin/env python3
"""Exact rational tests of Theorem 1; standard library only.
Run: python validate_boundary.py. No external files required.
Finite validation is not a proof or a test of computational lower bounds.
"""
from fractions import Fraction as F
from itertools import product

def det(A):
    A=[list(map(F,row)) for row in A]; ans=F(1)
    for j in range(len(A)):
        pivot=next((i for i in range(j,len(A)) if A[i][j]),None)
        if pivot is None:return F(0)
        if pivot!=j:A[j],A[pivot]=A[pivot],A[j];ans=-ans
        v=A[j][j];ans*=v
        for i in range(j+1,len(A)):
            t=A[i][j]/v
            for k in range(j+1,len(A)):A[i][k]-=t*A[j][k]
            A[i][j]=F(0)
    return ans

def inverse(A):
    n=len(A); B=[list(map(F,row))+[F(i==j) for j in range(n)] for i,row in enumerate(A)]
    for j in range(n):
        p=next(i for i in range(j,n) if B[i][j]);B[j],B[p]=B[p],B[j]
        t=B[j][j]; B[j]=[v/t for v in B[j]]
        for i in range(n):
            if i!=j:
                t=B[i][j]; B[i]=[x-t*y for x,y in zip(B[i],B[j])]
    return [r[n:] for r in B]

def mm(A,B):
    return [[sum((x*y for x,y in zip(row,col)),F(0)) for col in zip(*B)] for row in A]

def expand(n,edges,lengths):
    N=n; E=[]
    for (a,b),L in zip(edges,lengths):
        if L<1:raise ValueError('positive lengths required')
        path=[a]+list(range(N,N+L-1))+[b];N+=L-1
        E.extend(zip(path,path[1:]))
    return N,E

def bass(n,edges,u):
    A=[[F(0) for _ in range(n)] for _ in range(n)]
    degree=[0]*n
    for a,b in edges:A[a][b]-=u;A[b][a]-=u;degree[a]+=1;degree[b]+=1
    for a in range(n):A[a][a]=1+u*u*(degree[a]-1)
    return A

def edge_determinant(edges,u):
    arcs=edges+[(b,a) for a,b in edges]
    return det([[F(i==j)-u*int(b==c and d!=a) for j,(c,d) in enumerate(arcs)] for i,(a,b) in enumerate(arcs)])

def compressed(n,edges,lengths,u):
    W=[[F(i==j) for j in range(n)] for i in range(n)]
    hair=F(1); factor=F(1)
    for (a,b),L in zip(edges,lengths):
        t=u**L; q=1-t*t
        hair*=q/(1-u*u);factor*=q
        W[a][a]+=t*t/q;W[b][b]+=t*t/q
        W[a][b]-=t/q;W[b][a]-=t/q
    return W,hair,factor

def run():
    checks=0
    def test(name,ok):
        nonlocal checks
        if not ok:raise AssertionError(name)
        checks+=1;print('[PASS]',name)
    cases=[(3,[(0,1),(1,2),(2,0)],list(L)) for L in product([1,2],repeat=3)]
    cases += [(4,[(0,1),(1,2),(2,3),(3,0),(0,2)],[2,1,3,1,2]),
              (2,[(0,1)],[5]),(4,[(0,1),(1,2),(2,3)],[1,2,3])]
    for idx,(n,E,L) in enumerate(cases):
        N,Fedges=expand(n,E,L)
        for u in [F(0),F(1,3),F(-1,4)]:
            M=bass(N,Fedges,u);H=[row[n:] for row in M[n:]]
            S=[row[:n] for row in M[:n]]
            if H:
                K=[row[:n] for row in M[n:]]
                corr=mm(list(zip(*K)),mm(inverse(H),K))
                S=[[S[i][j]-corr[i][j] for j in range(n)] for i in range(n)]
            W,hair,pref=compressed(n,E,L,u)
            target=[[(1-u*u)*v for v in row] for row in W]
            test(f'T1a case={idx} u={u} hair determinant',det(H)==hair)
            test(f'T1b case={idx} u={u} Schur matrix',S==target)
            test(f'T1c case={idx} u={u} independent directed-edge determinant',edge_determinant(Fedges,u)==pref*det(W))
            test(f'T1d case={idx} u={u} full Schur determinant',det(M)==hair*det(S))
            if u>=0:
                degree=[sum(v in e for e in E) for v in range(n)];D=max(degree)
                lower=(1-(D-1)*u)/(1+u);upper=1+D*u/(1-u)
                test(f'T1e case={idx} u={u} exact Gershgorin bounds',all(W[i][i]-sum(abs(W[i][j]) for j in range(n) if i!=j)>=lower and W[i][i]+sum(abs(W[i][j]) for j in range(n) if i!=j)<=upper for i in range(n)))
    E=[(0,1),(1,2),(2,0)];u=F(1,3)
    _,_,p=compressed(3,E,[1,1,1],u);W,_,_=compressed(3,E,[1,1,1],u)
    W2,_,p2=compressed(3,E,[2,1,1],u)
    test('cycle-sensitive input: subdividing one edge changes zeta',p*det(W)!=(p2*det(W2)))
    test('cycle length formulas for 3 and 4 edges',p*det(W)==(1-u**3)**2 and p2*det(W2)==(1-u**4)**2)
    test('local update is supported only on its two endpoints',all(W[i][j]==W2[i][j] for i in range(3) for j in range(3) if i==2 or j==2))
    print(f'ALL {checks} EXACT CHECKS PASS')
    print('Scope: finite instances of Theorem 1 only; no claim of originality or quantum separation verification.')

if __name__=='__main__':run()
