
from FrozenOrbits.boundary_conditions import *
from FrozenOrbits.coordinate_transforms import *
from FrozenOrbits.constraints import *
from FrozenOrbits.LPE import LPE

import numpy as np
import pandas as pd
import copy
import time
from FrozenOrbits.dynamics import *

from GravNN.Support.transformations import cart2sph, invert_projection
from scipy.integrate import solve_ivp



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

        sol = solve_ivp(dynamics_w_STM, [0, T_i], z_i, args=(model,),atol=1E-6, rtol=1E-6)
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

        sol = solve_ivp(dynamics_w_STM, [0, T_i], z_i, args=(model,),atol=1E-6, rtol=1E-6)
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

def general_variable_time_bvp_OE(T, x0, model):
    T_i_p1 = copy.deepcopy(T) 
    x_i_p1 = copy.deepcopy(x0.reshape((-1,)))

    k = 0
    tol = 1
    while tol > 1E-6 and k < 10: 
        start_time = time.time()
        T_i = copy.deepcopy(T_i_p1) / 10
        x_i = copy.deepcopy(x_i_p1)
        N = len(x_i)
        phi_0 = np.identity(N)
        z_i = np.hstack((x_i.reshape((-1,)), phi_0.reshape((-1))))

        sol = solve_ivp(dynamics_w_STM_OE, [0, T_i], z_i, args=(model,),atol=1E-6, rtol=1E-6)
        print(f"{sol.success} \t {sol.message}")
        z_f = sol.y[:,-1]
        x_f = z_f[:N]

        x_dot_f = model.dOE_dt(x_f)
        phi_t0_tf = z_f[N:].reshape((N,N))

        V_i = np.hstack((x_i, T_i))
        C = x_f - x_i # F(V) in pdf
        D = np.hstack([phi_t0_tf - np.eye(N), x_dot_f.reshape((N,-1))])
        V_i_p1 = V_i - np.transpose(D.T@np.linalg.pinv(D@D.T)@C).squeeze()
        x_i_p1 = V_i_p1[0:N]
        T_i_p1 = V_i_p1[N]
        
        tol = np.linalg.norm(C)
        dx = np.linalg.norm((x_i_p1 - x_i)[:N])
        print(f"Iteration {k}: tol = {tol} \t dx_k = {dx} \t dT = {T_i_p1 - T_i} \t Time Elapsed: {time.time() - start_time}")
        k += 1

    return x_i_p1, T_i_p1

def general_variable_time_bvp_OE_ND(T, x0, model):
    T = model.non_dimensionalize_time(T).numpy()
    x0 = model.non_dimensionalize_state(x0).numpy()

    T_i_p1 = copy.deepcopy(T) 
    x_i_p1 = copy.deepcopy(x0.reshape((-1,)))

    k = 0
    tol = 1
    while tol > 1E-6 and k < 1: 
        start_time = time.time()
        T_i = copy.deepcopy(T_i_p1)
        x_i = copy.deepcopy(x_i_p1)
        N = len(x_i)
        phi_0 = np.identity(N)
        z_i = np.hstack((x_i.reshape((-1,)), phi_0.reshape((-1))))

        sol = solve_ivp(dynamics_w_STM_OE, [0, T_i], z_i, args=(model,),atol=1E-3, rtol=1E-3, method='LSODA')
        z_f = sol.y[:,-1]
        x_f = z_f[:N]

        x_dot_f = model.dOE_dt(x_f)
        phi_t0_tf = z_f[N:].reshape((N,N))

        V_i = np.hstack((x_i, T_i))
        C = x_f - x_i
        D = np.hstack([phi_t0_tf - np.eye(N), x_dot_f.reshape((N,-1))])
        V_i_p1 = V_i - np.transpose(D.T@np.linalg.pinv(D@D.T)@C).squeeze()
        x_i_p1 = V_i_p1[0:N]
        T_i_p1 = V_i_p1[N]
        
        tol = np.linalg.norm(C)
        dx = np.linalg.norm((x_i_p1 - x_i)[:N])
        print(f"Iteration {k}: tol = {tol} \t dx_k = {dx} \t dT = {T_i_p1 - T_i} \t Time Elapsed: {time.time() - start_time}")
        k += 1

    T_i_p1 = model.dimensionalize_time(T_i_p1).numpy()
    x_i_p1 = model.dimensionalize_state(np.array([x_i_p1])).numpy()

    return x_i_p1, T_i_p1

def general_variable_time_bvp_trad_OE(T, x0, model):
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

        sol = solve_ivp(dynamics_w_STM_OE, [0, T_i], z_i, args=(model,),atol=1E-6, rtol=1E-6)
        print(f"{sol.success} \t {sol.message}")
        z_f = sol.y[:,-1]
        x_f = z_f[:N]

        x_dot_f = model.dOE_dt(x_f)
        phi_t0_tf = z_f[N:].reshape((N,N))

        # The entire state 
        # V_i = np.hstack((x_i, T_i))
        # C = x_f - x_i # F(V) in pdf
        # D = np.hstack([phi_t0_tf - np.eye(N), x_dot_f.reshape((N,-1))])

        # V_i_p1 = V_i - np.transpose(D.T@np.linalg.pinv(D@D.T)@C).squeeze()
        # x_i_p1 = V_i_p1[0:N]
        # T_i_p1 = V_i_p1[N]

        # Option 1: Remove a variable from the state
        # Removing mean anomaly from the state (doesn't matter where on the orbit it is)
        # M = 5
        # V_i = np.hstack((x_i[:M], T_i))
        # C = C[:M]
        # D_original = np.hstack([phi_t0_tf - np.eye(N), x_dot_f.reshape((N,-1))])
        # D = np.hstack((D_original[:M, :M], D_original[:M,-1:]))

        # V_i_p1 = V_i - np.transpose(D.T@np.linalg.pinv(D@D.T)@C).squeeze()
        # x_i_p1 = V_i_p1[:M]
        # T_i_p1 = V_i_p1[-1]

        # Option 2: Remove a variable from the state
        # Remove the period from the state.

        V_i = x_i
        C = x_f - x_i
        D = phi_t0_tf - np.eye(N)

        V_i_p1 = V_i - np.transpose(D.T@np.linalg.pinv(D@D.T)@C).squeeze()
        x_i_p1 = V_i_p1
        T_i_p1 = T_i

        tol = np.linalg.norm(C)
        dx = np.linalg.norm((x_i_p1 - x_i)[:N])
        print(f"Iteration {k}: tol = {tol} \t dx_k = {dx} \t dT = {T_i_p1 - T_i} \t Time Elapsed: {time.time() - start_time}")
        k += 1

    return x_i_p1, T_i_p1

def general_variable_time_bvp_trad_OE_ND(T, x0_dim, model):

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

        sol = solve_ivp(dynamics_w_STM_OE, 
                        [0, T_i],
                        z_i,
                        args=(model,),
                        atol=1E-3, rtol=1E-3)
        print(f"{sol.success} \t {sol.message}")
        z_f = sol.y[:,-1]
        x_f = z_f[:N]

        x_dot_f = model.dOE_dt(x_f)
        phi_t0_tf = z_f[N:].reshape((N,N))
        
        # x_i_p1, T_i_p1 = OE_wo_a_e_i(z_f, x_i, x_i_p1, 
        #                                 T_i, start_time, model, k)
       
        # x_i_p1, T_i_p1 = OE_wo_a_e_i__w_T(z_f, x_i, x_i_p1, 
        #                                 T_i, start_time, model, k)
       
        x_i_p1, T_i_p1 = OE_wo_a_e_i__w_T_inv(z_f, x_i, x_i_p1, 
                                        T_i, start_time, model, k)


        k += 1

    x_i_p1 = model.dimensionalize_state(np.array([x_i_p1])).numpy()
    T_i_p1 = model.dimensionalize_time(T_i_p1).numpy()

    return x_i_p1, T_i_p1


