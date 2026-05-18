import numpy as np
from scipy.integrate import quad
from scipy.interpolate import AAA
from scipy.optimize import minimize_scalar
from scipy.optimize import bisect


def v_generate(k, beta):
    v = np.array([1/p**beta for p in range(1, k+1)])
    return v/np.sqrt(np.sum(v**2))

def align(u,v):
    return np.sum(u*v)/np.sqrt(np.sum(u**2)*np.sum(v**2))

'''
Gaussian integrals
'''
def Ez(f):
    return quad(lambda x: f(x)*np.exp(-x**2/2)/np.sqrt(2*np.pi), -6, 6)[0]

def hermite(x, n):
    if n==0:
        return 1
        
    a, b = 1, x
    for k in range(2, n + 1):
        a, b = b, x*b - (k - 1)*a
    return b

def hermite_coeff(f, n):
    return Ez(lambda x: f(x)*hermite(x, n))

def h(q, sig): # E[sig(u) sig(v)], where u,v~N(0,1), E[uv]=q
    f = lambda u: Ez( lambda v: sig(np.sqrt(q)*u + np.sqrt(1-q)*v) )
    return Ez( lambda u: f(u)**2 )

def hh(sig):
    qs = np.linspace(0, 1, 21)
    fs = np.array([h(q, sig) for q in qs])
    f = AAA(qs, fs)
    return f


def diff(g): # g function on [0, 1]
    e1, e2 = 1e-4, 1e-8
    def g_prime(x):
        if (x>e1)&(x<1-e1):
            return (g(x+e1)-g(x-e1))/(2*e1)
        if x<e1:
            return (g(x+e2)-g(x))/e2
        if x>1-e1:
            return (g(x)-g(x-e2))/e2
    return np.vectorize(g_prime)


def thres(f, max_expand=15, max_bisect=30, f_tol=1e-2, x_tol=1e-7): # smallest positive x such that f(x)>0
    x = 1
    for _ in range(max_expand):
        if f(x) > f_tol:
            break
        else:
            x *= 2
    a, b = x/2, x
    for _ in range(max_bisect):
        c = (a+b)/2
        if f(c) > f_tol:
            b=c
        else:
            a=c
        if (b-a<x_tol):
            break
    return b

def sig_related(sig):
    '''
    compute related functions to sigma
    g: g_sigma in paper
    dg: g'
    m_e: m_sigma in paper
    l_e: = l_sigma in paper
    m_s:  metastable solution near 1 of the RS potential
    l_s: phase transition for m_s>0
    '''
    g = hh(sig)
    dg = diff(g)
    def m(l, tol=1e-9):
        q = 1
        for _ in range(1000):
            q_old = q
            r = l*dg(q)
            q = r/(r+1)
            if np.max(np.abs(q - q_old))<tol:
                break
        # pot = l*g(q)/2 - q*r/2 + r/2 - np.log(1+r)/2
        pot = l*g(q) + q + np.log(1-q)
        if pot>l*g(0):
            qe = q
        else:
            qe = 0
        return [qe, q]
    
    l_e = thres(lambda l: m(l)[0])
    l_data_e = np.linspace(l_e, l_e*5, 50)
    m_data_e = np.array([m(l)[0] for l in l_data_e])
    m_e_ = AAA(l_data_e, m_data_e)

    def m_e(ls): # ls is an array
        mvals = np.zeros_like(ls)
        mask = (ls >= l_e)
        mvals[mask] = m_e_(ls[mask])
        return mvals

    l_s = thres(lambda l: m(l)[1])
    l_data_s = np.linspace(l_s, l_s*5, 50)
    m_data_s = np.array([m(l)[1] for l in l_data_s])
    m_s_ = AAA(l_data_s, m_data_s)

    def m_s(ls): # ls is an array
        mvals = np.zeros_like(ls)
        mask = (ls >= l_s)
        mvals[mask] = m_s_(ls[mask])
        return mvals
    
    return g, dg, l_e, m_e, l_s, m_s


'''
solving fixed-point equations
g, m: functions that work on arrays
'''

def solve(n, d, Delta, v, g, m): # for general v
    def f(x):
        q = m((n/d)*v**2/(Delta + x)) 
        return  g(1)*np.sum(v**2) - np.sum(v**2*g(q))

    mmse = bisect(lambda x: f(x)-x, 0, np.sum(v**2)*g(1), xtol=1e-6, maxiter=50)
    qw = m((n/d)*v**2/(Delta + mmse))
    return mmse, qw

def solve_dense(alpha, gamma, Delta, v, pv, g, m): # for dense readouts with density Pv
    def f(x):
        return g(1) - np.sum(pv*v**2*g(m(alpha*v**2/gamma/(Delta + x))))
    mmse = bisect(lambda x: f(x)-x, 0, np.sum(v**2)*g(1), xtol=1e-6, maxiter=50)
    qw = m(alpha*v**2/gamma/(Delta + mmse))
    return mmse, qw


def one_shot_BO(n, d, Delta, v, g, m): 
    '''
    generalization error of one-shot BO estimator, defined as
    \hat y(x) = sum_{i<=k_c} v_i \sigma(w'_i \cdot x), where W' is a posterior sample
    '''
    def f(x):
        q = m((n/d)*v**2/(Delta + x)) 
        return  g(1)*np.sum(v**2) - np.sum(v**2*g(q))

    mmse = bisect(lambda x: f(x)-x, 0, np.sum(v**2)*g(1), xtol=1e-6, maxiter=50)
    rw = (n/d)*v**2/(Delta+mmse)
    qw = m(rw)

    k = len(v)
    p = np.argmin(qw>0) # number of learned features
    if p==0: # all features are learned
        p = k

    one_shot_err = 2*np.sum([v[i]**2*(g(1)-g(qw[i])) for i in range(p)]) +  np.sum([ v[i]**2*(g(1)-g(qw[i])) for i in range(p, k)])
    return one_shot_err








