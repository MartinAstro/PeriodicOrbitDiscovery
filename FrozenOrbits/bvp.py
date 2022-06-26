
from FrozenOrbits.boundary_conditions import *
from FrozenOrbits.coordinate_transforms import *
from FrozenOrbits.constraints import *
from FrozenOrbits.LPE import LPE

import numpy as np
import pandas as pd
import copy
import time
from FrozenOrbits.dynamics import *
from FrozenOrbits.utils import calc_angle_diff

from GravNN.Support.transformations import cart2sph, invert_projection
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares, root
from GravNN.Support.ProgressBar import ProgressBar

np.set_printoptions(formatter={'float': "{0:0.2e}".format})

######################
## Helper Functions ##
######################

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

def evolve_state_w_STM(t, x_0, F, model, tol=1E-6):
    N = len(x_0)
    phi_0 = np.identity(N)
    z_i = np.hstack((x_0, phi_0.reshape((-1))))
    pbar = ProgressBar(t, enable=True)
    sol = solve_ivp(F, [0, t], z_i, args=(model,pbar),atol=tol, rtol=tol)
    pbar.close()
    if not sol.success:
        print(f"{sol.success} \t {sol.message}")
    x_f = sol.y[:N,-1]
    phi_f = sol.y[N:,-1].reshape((N,N))
    return x_f, phi_f


#######################
## Custom algorithms ##
#######################

def variable_time_mirror_bvp(T, x0, model):
    T_i_p1 = copy.deepcopy(T) / 2
    x_i_p1 = copy.deepcopy(x0)

    k = 0
    tol = 1
    while tol > 1E-6 and k < 10: 
        start_time = time.time()
        T_i = copy.deepcopy(T_i_p1)
        x_i = copy.deepcopy(x_i_p1)
        x_f, phi_f = evolve_state_w_STM(T_i, x_i, dynamics_cart_w_STM, model, tol=1E-6)

        x_dot_f = np.hstack((x_f[3:6], model.generate_acceleration(x_f[0:3])))

        V_i = np.array([x_i[0], x_i[4], T_i])
        C = np.array([
            [x_f[1]],
            [x_f[3]]
            ])
        D = np.array([
            [phi_f[1,0], phi_f[1,4], x_dot_f[1]],
            [phi_f[3,0], phi_f[3,4], x_dot_f[3]]
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

def general_variable_time_cartesian_bvp(T, x0, model):
    T_i_p1 = copy.deepcopy(T) 
    x_i_p1 = copy.deepcopy(x0)

    k = 0
    tol = 1
    while tol > 1E-6 and k < 10: 
        start_time = time.time()
        T_i = copy.deepcopy(T_i_p1)
        x_i = copy.deepcopy(x_i_p1)
        N = len(x_i)

        # Propagate dynamics 
        x_f, phi_f = evolve_state_w_STM(T_i, x_i, dynamics_cart_w_STM, model, tol=1E-6)

        # Get X_dot @ t_f
        r_f = x_f[0:3]
        v_f = x_f[3:6]
        a_f = model.generate_acceleration(r_f)
        x_dot_f = np.hstack((v_f, a_f)).reshape((6,1))

        # Build BVP "state" vector
        V_i = np.hstack((x_i, T_i))

        # Define constraint vector and corresponding partials w.r.t. BVP state vector (dC/dV)
        C = x_f - x_i
        D = np.block([phi_f - np.eye(len(x0)), x_dot_f])

        # Compute correction term and apply 
        dV = np.transpose(D.T@np.linalg.pinv(D@D.T)@C).squeeze()
        V_i_p1 = V_i - dV

        # Map back to state
        x_i_p1 = V_i_p1[0:len(x0)]
        T_i_p1 = V_i_p1[len(x0)]
        
        tol = np.linalg.norm(C)
        dx = np.linalg.norm((x_i_p1 - x_i)[0:len(x0)])
        print(f"Iteration {k}: tol = {tol} \t dx_k = {dx} \t dT = {T_i_p1 - T_i} \t Time Elapsed: {time.time() - start_time}")
        k += 1

    return x_i_p1, T_i_p1

def general_variable_time_bvp_trad_OE(T_dim, OE_0_dim, model):

    OE_0 = model.non_dimensionalize_state(OE_0_dim).numpy()
    T = model.non_dimensionalize_time(T_dim).numpy()
    print(f"Total Time {T} \n Dim State {OE_0_dim} \n Non Dim State {OE_0}")
    T_i_p1 = copy.deepcopy(T) 
    OE_i_p1 = copy.deepcopy(OE_0.reshape((-1,)))

    k = 0
    tol = 1
    while tol > 1E-6 and k < 10: 
        T_i = copy.deepcopy(T_i_p1) 
        x_i = copy.deepcopy(OE_i_p1)
        x_f, phi_f = evolve_state_w_STM(T_i, x_i, dynamics_OE_w_STM, model, tol=1E-3)
        OE_i_p1, T_i_p1 = OE_constraint(x_f, phi_f, x_i, T_i, model, k, 
                            decision_variable_mask=None,
                            constraint_variable_mask=None)
        k += 1

    OE_i_p1 = model.dimensionalize_state(np.array([OE_i_p1])).numpy()
    T_i_p1 = model.dimensionalize_time(T_i_p1).numpy()
    x_i_p1 = trad2cart_tf(OE_i_p1, model.mu_tilde).numpy()

    return OE_i_p1, x_i_p1, T_i_p1


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
    print(f"x0: {x}")
    print(f"xf: {sol.y[:,-1]}")
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
