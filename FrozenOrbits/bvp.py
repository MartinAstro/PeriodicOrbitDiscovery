
from FrozenOrbits.boundary_conditions import *
from FrozenOrbits.coordinate_transforms import *
from FrozenOrbits.constraints import *
from FrozenOrbits.LPE import LPE

import numpy as np
np.set_printoptions(formatter={'float': "{0:0.2e}".format})
import pandas as pd
import copy
import time
from FrozenOrbits.dynamics import *
from FrozenOrbits.utils import calc_angle_diff

from GravNN.Support.transformations import cart2sph, invert_projection
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares, root
from GravNN.Support.ProgressBar import ProgressBar


def debug_plot(sol, lpe, element_set):
    from GravNN.CelestialBodies.Asteroids import Eros
    import matplotlib.pyplot as plt
    from FrozenOrbits.visualization import plot_cartesian_state_3d, plot_OE_1d
    
    plot_OE_1d(sol.t, sol.y.T, element_set)

    OE_dim = lpe.dimensionalize_state(sol.y.T).numpy()
    cart = oe2cart_tf(OE_dim, Eros().mu, element_set).numpy()

    sol_cart = copy.deepcopy(sol)
    sol_cart.y = cart.T
    plot_cartesian_state_3d(sol_cart.t, sol_cart.y.T, Eros().obj_8k)
    plt.show()
    return


def variable_time_mirror_bvp(T, x0, model):
    T_i_p1 = copy.deepcopy(T) / 2
    x_i_p1 = copy.deepcopy(x0)

    k = 0
    tol = 1
    while tol > 1E-6 and k < 10: 
        start_time = time.time()
        T_i = copy.deepcopy(T_i_p1)
        x_i = copy.deepcopy(x_i_p1)
        N = len(x_i)
        phi_0 = np.identity(N)
        z_i = np.hstack((x_i, phi_0.reshape((-1))))
        pbar = ProgressBar(T_i, enable=True)
        sol = solve_ivp(dynamics_cart_w_STM, [0, T_i], z_i, args=(model,pbar),atol=1E-6, rtol=1E-6)
        pbar.close()
        z_f = sol.y[:,-1]
        x_f = z_f[:N]

        x_dot_f = np.hstack((x_f[3:6], model.generate_acceleration(x_f[0:3])))
        phi_t0_tf = z_f[N:].reshape((N,N))

        V_i = np.array([x_i[0], x_i[4], T_i])
        C = np.array([
            [x_f[1]],
            [x_f[3]]
            ])
        D = np.array([
            [phi_t0_tf[1,0], phi_t0_tf[1,4], x_dot_f[1]],
            [phi_t0_tf[3,0], phi_t0_tf[3,4], x_dot_f[3]]
            ])
        
        V_i_p1 = V_i - np.transpose(D.T@np.linalg.pinv(D@D.T)@C).squeeze()
        x_i_p1[0] = V_i_p1[0]
        x_i_p1[4] = V_i_p1[1]
        T_i_p1 = V_i_p1[2]
        tol = np.linalg.norm(C)
        dx = np.linalg.norm((x_i_p1 - x_i)[0:6])
        print(f"Iteration {k}: tol = {tol} \t dx_k = {dx} \t dT = {T_i_p1 - T_i} \t Time Elapsed: {time.time() - start_time}")
        k += 1

    return x_i_p1, T_i_p1*2

def general_variable_time_bvp(T, x0, model):
    T_i_p1 = copy.deepcopy(T) 
    x_i_p1 = copy.deepcopy(x0)

    k = 0
    tol = 1
    while tol > 1E-6 and k < 10: 
        start_time = time.time()
        T_i = copy.deepcopy(T_i_p1)
        x_i = copy.deepcopy(x_i_p1)
        N = len(x_i)
        phi_0 = np.identity(N)
        z_i = np.hstack((x_i, phi_0.reshape((-1))))
        pbar = ProgressBar(T_i, enable=True)
        sol = solve_ivp(dynamics_cart_w_STM, [0, T_i], z_i, args=(model,pbar),atol=1E-6, rtol=1E-6)
        pbar.close()
        z_f = sol.y[:,-1]
        x_f = z_f[:N]

        r_f = x_f[0:3]
        v_f = x_f[3:6]
        a_f = model.generate_acceleration(r_f)
        x_dot_f = np.hstack((v_f, a_f)).reshape((6,1))
        phi_t0_tf = z_f[N:].reshape((N,N))

        V_i = np.hstack((x_i, T_i))
        C = x_f - x_i
        D = np.block([phi_t0_tf - np.eye(len(x0)), x_dot_f])
        V_i_p1 = V_i - np.transpose(D.T@np.linalg.pinv(D@D.T)@C).squeeze()
        x_i_p1 = V_i_p1[0:len(x0)]
        T_i_p1 = V_i_p1[len(x0)]
        
        tol = np.linalg.norm(C)
        dx = np.linalg.norm((x_i_p1 - x_i)[0:len(x0)])
        print(f"Iteration {k}: tol = {tol} \t dx_k = {dx} \t dT = {T_i_p1 - T_i} \t Time Elapsed: {time.time() - start_time}")
        k += 1

    return x_i_p1, T_i_p1

def general_variable_time_bvp_trad_OE(T, x0_dim, model, constraint):

    x0 = model.non_dimensionalize_state(x0_dim).numpy()
    T = model.non_dimensionalize_time(T).numpy()
    print(f"Total Time {T} \n Dim State {x0_dim} \n Non Dim State {x0}")
    T_i_p1 = copy.deepcopy(T) 
    x_i_p1 = copy.deepcopy(x0.reshape((-1,)))

    k = 0
    tol = 1
    while tol > 1E-6 and k < 10: 
        start_time = time.time()
        T_i = copy.deepcopy(T_i_p1) 
        x_i = copy.deepcopy(x_i_p1)
        N = len(x_i)
        phi_0 = np.identity(N)
        z_i = np.hstack((x_i.reshape((-1,)), phi_0.reshape((-1))))
        pbar = ProgressBar(T_i, enable=True)
        sol = solve_ivp(dynamics_OE_w_STM, 
                        [0, T_i],
                        z_i,
                        args=(model,pbar),
                        atol=1E-3, rtol=1E-3)
        pbar.close()
        print(f"{sol.success} \t {sol.message}")
        z_f = sol.y[:,-1]
        x_i_p1, T_i_p1 = constraint(z_f, x_i, x_i_p1, 
                                        T_i, start_time, model, k)
        k += 1

    x_i_p1 = model.dimensionalize_state(np.array([x_i_p1])).numpy()
    T_i_p1 = model.dimensionalize_time(T_i_p1).numpy()

    return x_i_p1, T_i_p1

######################
## Scipy algorithms ##
######################


def F_milankovitch(x, T, lpe):
    pbar = ProgressBar(T, enable=False)
    sol = solve_ivp(dynamics_OE, 
            [0, T],
            x,
            args=(lpe, pbar),
            atol=1E-7, rtol=1E-7,
            t_eval=np.linspace(0,T, 100)
            )
    pbar.close()
    dx = sol.y[:,-1] - x 
    dx[6] = calc_angle_diff(x[6], sol.y[6,-1])

    # for debugging
    # debug_plot(sol, lpe, 'milankovitch')

    print(f"dx_norm {np.linalg.norm(dx)} \t dH_mag {np.linalg.norm(dx[0:3])} \t de_mag = {np.linalg.norm(dx[3:6])} \t dL = {dx[6]}")# \n {dx}")
    return dx

def F_traditional(x, T, lpe):
    pbar = ProgressBar(T, enable=False)
    sol = solve_ivp(dynamics_OE, 
            [0, T],
            x,
            args=(lpe, pbar),
            t_eval=np.linspace(0, T, 100),
            atol=1E-7, rtol=1E-7)
    pbar.close()
    dx = sol.y[:,-1] - x # return R^6 where m = 6

    # # for debugging
    # debug_plot(sol, lpe, 'traditional')

    # dx[2] = calc_angle_diff(x[2], sol.y[2,-1])
    dx[3] = calc_angle_diff(x[3], sol.y[3,-1])
    dx[4] = calc_angle_diff(x[4], sol.y[4,-1])
    dx[5] = calc_angle_diff(x[5], sol.y[5,-1])

    print(f"dx_norm {np.linalg.norm(dx)} \t {dx}")
    return dx

def jac(x, T, lpe):
    pbar = ProgressBar(T, enable=True)
    N = len(x)
    phi_0 = np.identity(N)
    z_i = np.hstack((x.reshape((-1,)), phi_0.reshape((-1))))
    sol = solve_ivp(dynamics_OE_w_STM, 
                    [0, T],
                    z_i,
                    args=(lpe,pbar),
                    atol=1E-7, rtol=1E-7)
    pbar.close()
    z_f = sol.y[:,-1]
    phi_ti_t0 = np.reshape(z_f[N:], (N,N))
    D = phi_ti_t0 - np.eye(N)
    return D

def scipy_periodic_orbit_algorithm(T_dim, OE_0_dim, lpe, solution_bounds, element_set):

    OE_0 = lpe.non_dimensionalize_state(OE_0_dim).numpy()
    T = lpe.non_dimensionalize_time(T_dim).numpy()
    print(f"Total Time {T} \nDim State {OE_0_dim} \nNon Dim State {OE_0}")

    if element_set == 'traditional':
        F = F_traditional
    elif element_set == 'milankovitch':
        F = F_milankovitch
    else:
        raise NotImplementedError(f"The element set '{element_set}' is not defined!")

    result = least_squares(F, OE_0.reshape((-1,)), jac, 
                            args=(T, lpe),
                            bounds=solution_bounds,
                            verbose=2,
                            )
    
    print(f"""
    Success? ({result.success} \t Status: {result.status} 
    Message: {result.message}
    OE_0 = {result.x} 
    Number of Function Evals: {result.nfev} \t Number of Jacobian Evals: {result.njev}")
    """)

    OE_0_sol = lpe.dimensionalize_state(np.array([result.x])).numpy()
    T_sol = lpe.dimensionalize_time(T).numpy()

    X_0_sol = oe2cart_tf(OE_0_sol, lpe.mu_tilde, element_set).numpy()[0]

    return OE_0_sol, X_0_sol, T_sol
