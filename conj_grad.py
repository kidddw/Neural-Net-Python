#!/usr/bin/python -tt

import numpy as np

def nonlinear_cg(func, x_init, gradfunc, args):
    """Nonlinear Conjugate Gradient Method
       alpha: float (scalar) step size
       beta : float (scalar) 
       d    : numpy matrix (column vector)
       r    : numpy matrix (column vector)
       x    : numpy matrix (column vector)"""
    d = - gradfunc(x_init, *args) 
    r = d
    x = x_init
    alpha = 0.01
    for i in range(1000):
        J = func(x, *args)
        if i > 0 and J >= J_old: break
        alpha = argmin_sec(gradfunc, x, d, alpha, args)
        x = x + alpha * d
        r_new = - gradfunc(x, *args)
    #    beta = Fletcher_Reeves(r_new, r)
        beta = Polak_Ribiere(r_new, r)
        r = r_new
        d = r + beta * d
        J_old = J
  return x



def Fletcher_Reeves(r_new, r):
    num = np.dot(r_new.transpose(), r_new).item()
    denom = np.dot(r.transpose(), r).item()
    return num / denom


def Polak_Ribiere(r_new, r):
    num = np.dot(r_new.transpose(), (r_new - r)).item()  
    denom = np.dot(r.transpose(), r).item()
    if (num / denom) <= 0.0: return 0.0
    if (num / denom) >= 0.0: return num / denom


def argmin_sec(func_grad,x,d,sigma,args):
    """find the parameter alpha which minimizes f(x+alpha d), using Secant method"""
    num = np.dot(func_grad(x,*args).transpose(), d).item()
    denom = np.dot(func_grad(x + sigma * d,*args).transpose(), d).item() - num
    return - sigma * num / denom 
  






