
from FrozenOrbits.boundary_conditions import *
from FrozenOrbits.coordinate_transforms import *
from FrozenOrbits.LPE import LPE

import numpy as np
import pandas as pd

from GravNN.Support.transformations import cart2sph, invert_projection
from scipy.integrate import solve_bvp

from FrozenOrbits.ivp import solve_ivp_pos_problem, solve_ivp_oe_problem

def bc(ya, yb, p=None):
    periodic_res = yb - ya 
    bc_res = np.hstack((periodic_res))#, ic_res))
    return bc_res

def solve_bvp_pos_problem(T, state, lpe, initial_nodes=100, t_eval=None, tol=1e-4, bc_tol=1e0, max_nodes=300000, model_includes_pm=False):

    def fun(x,y,p=None):
        "Return the first-order system"
        # print(x)
        R, V = y[0:3], y[3:6]
        r = np.linalg.norm(R, axis=0)
        a_pm_sph = np.vstack((-lpe.mu/r**2, np.zeros((len(r),)),np.zeros((len(r),)))).T
        r_sph = cart2sph(R.T)
        a = lpe.model.generate_acceleration(R.T).numpy()
        if model_includes_pm:
            dxdt = np.vstack((V, (a).T))
        else:
            a_pm_xyz = invert_projection(r_sph, a_pm_sph)
            dxdt = np.vstack((V, (a_pm_xyz - a).T))
        return dxdt.reshape((6,-1))
    
    t_mesh = np.linspace(0, T, initial_nodes)
    sol = solve_ivp_pos_problem(T, state, lpe, t_eval=t_mesh)
    y_guess = sol.y

    p = None # Initial guess for unknown parameters
    results = solve_bvp(fun, bc, t_mesh, y_guess, p=p, verbose=2, max_nodes=max_nodes, tol=tol, bc_tol=bc_tol)
    return results


def solve_bvp_oe_problem(T, OE, lpe, initial_nodes=100, tol=1e-4, max_nodes=300000):
    bc = get_bc(lpe.element_set)
    def fun(x,y,p=None):
        dxdt = np.array([v.numpy() for v in lpe(y.T).values()])
        return dxdt.reshape((6,-1))

    t_mesh = np.linspace(0, T, initial_nodes)
    y_guess = []
    for _ in range(0,len(t_mesh)) : y_guess.append(OE)
    y_guess = np.array(y_guess).squeeze().T
    sol = solve_ivp_pos_problem(T, np.hstack(oe2cart_tf(OE, lpe.mu, lpe.element_set)), lpe, t_eval=t_mesh)
    states = sol.y
    y_guess = []
    for i in range(len(states.T)):
        state = states[:,i].reshape((6,1))
        y_guess.append(cart2oe_tf(state.T, lpe.mu, lpe.element_set).numpy())
    y_guess = np.array(y_guess).squeeze().T
    # y_guess = solve_ivp_problem(T, OE, lpe, t_eval=t_mesh)

    p = None # Initial guess for unknown parameters
    results = solve_bvp(fun, bc, t_mesh, y_guess, p=p, verbose=2, max_nodes=max_nodes, tol=tol)     
    return results


def solve_bvp_pos_problem_tf(T, state, lpe, initial_nodes=100, t_eval=None, tol=1e-3, bc_tol=1e3, max_nodes=300000):
    import tensorflow as tf

    def fun_jac(x,y,p=None):

        fun_tf()
        "Return the first-order system"
        # print(x)
        R, V = y[0:3], y[3:6]
        r = np.linalg.norm(R, axis=0)
        a_pm_sph = np.vstack((-lpe.mu/r**2, np.zeros((len(r),)),np.zeros((len(r),)))).T
        r_sph = cart2sph(R.T)
        a_pm_xyz = invert_projection(r_sph, a_pm_sph)
        a = lpe.model.generate_acceleration(R.T).numpy()
        dxdt = np.vstack((V, (a_pm_xyz - a).T))
        return dxdt.reshape((6,-1))
    

    def fun_tf(x,y,p=None):
        "Return the first-order system"
        # print(x)
        R, V = y[0:3], y[3:6]
        r = tf.math.l2_normalize(R, axis=0)
        a_pm_sph = tf.stack((-lpe.mu/r**2, np.zeros((len(r),)),np.zeros((len(r),))),0)
        a_pm_sph = tf.transpose(a_pm_sph)
        r_sph = cart2sph(R.T)
        a_pm_xyz = invert_projection(r_sph, a_pm_sph)
        a = lpe.model.generate_acceleration(R.T).numpy()
        dxdt = np.vstack((V, (a_pm_xyz - a).T))
        return dxdt.reshape((6,-1))
    
    t_mesh = np.linspace(0, T, initial_nodes)
    sol = solve_ivp_pos_problem(T, state, lpe, t_eval=t_mesh)
    y_guess = sol.y

    y_guess_tf = tf.data.from_tensor_slices(y_guess)


    p = None # Initial guess for unknown parameters
    results = solve_bvp(fun, bc, t_mesh, y_guess, p=p, verbose=2, max_nodes=max_nodes, tol=tol, bc_tol=bc_tol)
    return results