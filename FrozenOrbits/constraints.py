import copy
import numpy as np
from FrozenOrbits.dynamics import *

from GravNN.Support.ProgressBar import ProgressBar
from FrozenOrbits.utils import calc_angle_diff
from scipy.integrate import solve_ivp

def update_state(X_0, X_subset, decision_variable_mask):
    k = 0
    X_updated = copy.deepcopy(X_0)
    for i in range(len(X_updated)):
        # Only update allowed decision variables
        if decision_variable_mask[i]:
            X_updated[i] = X_subset[k]
            k += 1

    X_updated = np.array([X_updated]) # the non-dim OE
    return X_updated


########################
## Vector Constraints ##
########################

def constraint_shooting(V_0, lpe, x_0, 
    decision_variable_mask, 
    constraint_variable_mask, 
    constraint_angle_wrap_mask, 
    rtol, 
    atol):
    
    x_i = update_state(x_0[:-1], V_0, decision_variable_mask).reshape((-1,))
    x_i = x_i.reshape((-1,))
    if decision_variable_mask[-1]:
        T = V_0[-1]
    else:
        T = x_0[-1]

    # Propagate the updated state
    pbar = ProgressBar(T, enable=False)
    sol = solve_ivp(dynamics_OE, 
            [0, T],
            x_i.reshape((-1,)), 
            args=(lpe, pbar),
            atol=atol, rtol=rtol
            )
    pbar.close()
    x_f = sol.y[:,-1]

    # Calculate the constraint vector
    C = x_f - x_i 
    C = np.hstack((C, [0.0])) # add the period to the constraint ()

    # Wrap angles depending on the element set
    for i in range(len(x_f)):
        if constraint_angle_wrap_mask[i]:
            C[i] = calc_angle_diff(x_i[i], x_f[i])
            C[i] = C[i] / lpe.theta_star # scale the angles to be 1 rather than 2*np.pi
    C = C[constraint_variable_mask] # remove masked variables
    return C

def constraint_shooting_jac(V_0, lpe, x_0, 
    decision_variable_mask, 
    constraint_variable_mask,
    constraint_angle_wrap_mask,
    rtol, 
    atol):
    N = len(x_0) - 1 # Remove from the state
    phi_0 = np.identity(N)

    # Only update allowed decision variables
    x_i = update_state(x_0[:-1], V_0, decision_variable_mask)
    x_i = x_i.reshape((-1,))

    if decision_variable_mask[-1]:
        T = V_0[-1]
    else:
        T = x_0[-1]

    # Propagate the corrected state
    z_i = np.hstack((
        x_i.reshape((-1,)),
        phi_0.reshape((-1)))
        )

    pbar = ProgressBar(T, enable=True)
    sol = solve_ivp(dynamics_OE_w_STM, 
                    [0, T],
                    z_i.reshape((-1,)),
                    args=(lpe,pbar),
                    atol=atol, rtol=rtol,
                    #method='LSODA'
                    )
    pbar.close()
    z_f = sol.y[:,-1]
    x_f = z_f[:N]
    phi_ti_t0 = np.reshape(z_f[N:], (N,N))
    x_dot_f = lpe.dOE_dt(x_f)

    # Evaluate the general jacobian
    D = np.hstack([phi_ti_t0 - np.eye(N), x_dot_f.reshape((N,-1))])

    # Append a time variable row
    D = np.vstack((D, np.zeros((N+1))))

    # Remove specified decision variables from jacobian
    D = D[:, decision_variable_mask] # remove columns in D

    # Scale angle constraints by 2*pi
    for i in range(len(x_f)):
        if constraint_angle_wrap_mask[i]:
            D[i] = D[i] / lpe.theta_star # scale the angles to be 1 rather than 2*np.pi 

    # # Remove constraint variables
    D = D[constraint_variable_mask, :] # remove rows in D

    return D

def constraint_instantaneous(V_0, lpe, x_0, 
    decision_variable_mask, 
    constraint_variable_mask, 
    constraint_angle_wrap_mask):
    
    x_i = update_state(x_0, V_0, decision_variable_mask)
    C = lpe.dOE_dt(x_i)
    C = C[constraint_variable_mask] # remove masked variables

    return C

def constraint_instantaneous_jac(V_0, lpe, x_0, 
    decision_variable_mask, 
    constraint_variable_mask,
    constraint_angle_wrap_mask):

    # Only update allowed decision variables
    x_i = update_state(x_0, V_0, decision_variable_mask)

    # Evaluate the general jacobian
    D = lpe.dOE_dt_dx(x_i)

    # Remove specified decision variables from jacobian
    D = D[:, decision_variable_mask] # remove columns in D
    
    D = D[constraint_variable_mask, :] # remove rows in D

    return D

########################
## Scalar Constraints ##
########################

def constraint_shooting_scalar(V_0, lpe, x_0, 
    decision_variable_mask, 
    constraint_variable_mask, 
    constraint_angle_wrap_mask):
    
    C = constraint_shooting(V_0, lpe, x_0, 
    decision_variable_mask, 
    constraint_variable_mask, 
    constraint_angle_wrap_mask)
    return np.linalg.norm(C)**2

def constraint_shooting_jac_scalar(V_0, lpe, x_0, 
    decision_variable_mask, 
    constraint_variable_mask,
    constraint_angle_wrap_mask):

    C = constraint_shooting(V_0, lpe, x_0, 
    decision_variable_mask, 
    constraint_variable_mask, 
    constraint_angle_wrap_mask)

    D = constraint_shooting_jac(V_0, lpe, x_0, 
    decision_variable_mask, 
    constraint_variable_mask, 
    constraint_angle_wrap_mask)

    D_scalar = np.zeros((len(D),))
    for i in range(len(D)):
        D_scalar += 2*C[i]*D[i,:]
    
    return D

def constraint_instantaneous_scalar(V_0, lpe, x_0, 
    decision_variable_mask, 
    constraint_variable_mask, 
    constraint_angle_wrap_mask):
    
    C = constraint_instantaneous(V_0, lpe, x_0, 
    decision_variable_mask, 
    constraint_variable_mask, 
    constraint_angle_wrap_mask)

    return np.linalg.norm(C)**2

def constraint_instantaneous_jac_scalar(V_0, lpe, x_0, 
    decision_variable_mask, 
    constraint_variable_mask,
    constraint_angle_wrap_mask):

    D = constraint_instantaneous_jac(V_0, lpe, x_0, 
    decision_variable_mask, 
    constraint_variable_mask, 
    constraint_angle_wrap_mask)

    D_scalar = np.zeros((len(D),))
    for i in range(len(D)):
        D_scalar += D[i,:]
    
    return D


def OE_constraint(x_f, phi_f, x_0, T_i, model, k, decision_variable_mask=None, constraint_variable_mask=None, constraint_angle_wrap_mask=None):

    if decision_variable_mask is None:
        decision_variable_mask = [True]*(len(x_f)+1) # N + 1
    if constraint_variable_mask is None:
        constraint_variable_mask = [True]*len(x_f) # N
    if constraint_angle_wrap_mask is None:
        constraint_angle_wrap_mask = [False]*len(x_f) # N

    N = len(x_0)
    x_dot_f = model.dOE_dt(x_f)

    # Nominal decision variables
    V = np.hstack((x_0, T_i))

    # Nominal constraint variables
    C = x_f - x_0 

    # This is necessary because the constraint will look heavily violated in M.
    for i in range(len(C)):
        if constraint_angle_wrap_mask[i]:
            C[i] = calc_angle_diff(x_0[i], x_f[i])

    # Nominal partials
    D = np.hstack([phi_f - np.eye(N), x_dot_f.reshape((N,-1))])

    # Remove decision variables 
    V = V[decision_variable_mask]
    D = D[:, decision_variable_mask] # remove columns in D

    # Remove constraint variables
    C = C[constraint_variable_mask]
    D = D[constraint_variable_mask, :] # remove rows in D

    # Compute decision variable update
    dV = np.transpose(D.T@np.linalg.pinv(D@D.T)@C).squeeze()

    # Weighted Least Squares # https://en.wikipedia.org/wiki/Weighted_least_squares#Parameter_errors_and_correlation
    # W = np.diag([1, 1, 1/np.pi, 1/(2*np.pi), 1/(2*np.pi), 1/(2*np.pi), 1])**2
    # dV_weighted = np.transpose((D@W).T@np.linalg.pinv(D@W@D.T)@C).squeeze()

    V_corrected = V - dV

    # Map V back onto the state
    x_corrected = copy.deepcopy(x_0)
    constraint_idx = 0
    for i in range(len(x_0)):
        if decision_variable_mask[i]:
            x_corrected[i] = V_corrected[constraint_idx]
            constraint_idx += 1
    
    # Check if period is a variable that can be updated
    if decision_variable_mask[-1]:
        T_corrected = V_corrected[-1]
    else: 
        T_corrected = T_i

    # print_update(x_0, x_corrected, T_i, T_corrected, k, C)
    return x_corrected, T_corrected