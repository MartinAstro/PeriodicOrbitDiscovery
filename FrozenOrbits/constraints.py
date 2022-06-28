import numpy as np
import time
import copy

from FrozenOrbits.utils import calc_angle_diff

def print_update(x_0, X_corrected, T_i, T_corrected, k, C):
    tol = np.linalg.norm(C)
    dx = np.linalg.norm((X_corrected - x_0))
    print(f"Iteration {k}: tol = {tol} \t dx_k = {dx} \t dT = {T_corrected - T_i}")
    print(f"Old Non-Dim State: {x_0}")
    print(f"New Non-Dim State: {X_corrected}")

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

    print_update(x_0, x_corrected, T_i, T_corrected, k, C)
    return x_corrected, T_corrected