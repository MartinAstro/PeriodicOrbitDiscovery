import numpy as np
import time

def OE(z_f, x_i, x_i_p1, T_i, start_time, model, k):
    N = len(x_i)
    x_f = z_f[:N]

    x_dot_f = model.dOE_dt(x_f)
    phi_t0_tf = z_f[N:].reshape((N,N))

    V_i = np.hstack((x_i, T_i))
    C = x_f - x_i # F(V) in pdf
    D = np.hstack([phi_t0_tf - np.eye(N), x_dot_f.reshape((N,-1))])

    dV = np.transpose(D.T@np.linalg.pinv(D@D.T)@C).squeeze()
    V_i_p1 = V_i - dV
    x_i_p1 = V_i_p1[0:N]
    T_i_p1 = V_i_p1[N]

    tol = np.linalg.norm(C)
    dx = np.linalg.norm((x_i_p1 - x_i))
    print(f"Iteration {k}: tol = {tol} \t dx_k = {dx} \t dT = {T_i_p1 - T_i} \t Time Elapsed: {time.time() - start_time}")
    print(f"Old Non-Dim State: {x_i}")
    print(f"{dV}")
    print(f"New Non-Dim State: {x_i_p1}")
    return x_i_p1, T_i_p1

def OE_wo_M(z_f, x_i, x_i_p1, T_i, start_time, model, k):
    N = len(x_i)
    x_f = z_f[:N]

    x_dot_f = model.dOE_dt(x_f)
    phi_t0_tf = z_f[N:].reshape((N,N))

    M = 5
    V_i = np.hstack((x_i[:M], T_i))
    C = C[:M]
    D_original = np.hstack([phi_t0_tf - np.eye(N), x_dot_f.reshape((N,-1))])
    D = np.hstack((D_original[:M, :M], D_original[:M,-1:]))

    dV = np.transpose(D.T@np.linalg.pinv(D@D.T)@C).squeeze()
    V_i_p1 = V_i - dV
    x_i_p1 = V_i_p1[:M]
    T_i_p1 = V_i_p1[-1]

    tol = np.linalg.norm(C)
    dx = np.linalg.norm((x_i_p1 - x_i))
    print(f"Iteration {k}: tol = {tol} \t dx_k = {dx} \t dT = {T_i_p1 - T_i} \t Time Elapsed: {time.time() - start_time}")
    print(f"Old Non-Dim State: {x_i}")
    print(f"{dV}")
    print(f"New Non-Dim State: {x_i_p1}")
    return x_i_p1, T_i_p1

def OE_wo_T(z_f, x_i, x_i_p1, T_i, start_time, model, k):
    N = len(x_i)
    x_f = z_f[:N]

    x_dot_f = model.dOE_dt(x_f)
    phi_t0_tf = z_f[N:].reshape((N,N))

    V_i = x_i
    C = x_f - x_i
    D = phi_t0_tf - np.eye(N)

    dV =  np.transpose(D.T@np.linalg.pinv(D@D.T)@C).squeeze()
    V_i_p1 = V_i - dV
    x_i_p1 = V_i_p1
    T_i_p1 = T_i

    tol = np.linalg.norm(C)
    dx = np.linalg.norm((x_i_p1 - x_i))
    print(f"Iteration {k}: tol = {tol} \t dx_k = {dx} \t dT = {T_i_p1 - T_i} \t Time Elapsed: {time.time() - start_time}")
    print(f"Old Non-Dim State: {x_i}")
    print(f"{dV}")
    print(f"New Non-Dim State: {x_i_p1}")
    return x_i_p1, T_i_p1

def OE_wo_M_T(z_f, x_i, x_i_p1, T_i, start_time, model, k):
    N = len(x_i)
    x_f = z_f[:N]

    x_dot_f = model.dOE_dt(x_f)
    phi_t0_tf = z_f[N:].reshape((N,N))

    M = 5
    V_i = x_i[:M]
    C = (x_f - x_i)[:M]
    D_original = phi_t0_tf - np.eye(N)
    D = D_original[:M, :M]

    dV = np.transpose(D.T@np.linalg.pinv(D@D.T)@C).squeeze()
    V_i_p1 = V_i - dV
    x_i_p1[:M] = V_i_p1
    T_i_p1 = T_i

    tol = np.linalg.norm(C)
    dx = np.linalg.norm((x_i_p1 - x_i))
    print(f"Iteration {k}: tol = {tol} \t dx_k = {dx} \t dT = {T_i_p1 - T_i} \t Time Elapsed: {time.time() - start_time}")
    print(f"Old Non-Dim State: {x_i}")
    print(f"{dV}")
    print(f"New Non-Dim State: {x_i_p1}")
    return x_i_p1, T_i_p1

def OE_wo_semimajor(z_f, x_i, x_i_p1, T_i, start_time, model, k):
    N = len(x_i)
    x_f = z_f[:N]

    x_dot_f = model.dOE_dt(x_f)
    phi_t0_tf = z_f[N:].reshape((N,N))

    V_i = x_i[1:]
    C = x_f - x_i
    D_original = phi_t0_tf - np.eye(N)
    D = D_original[:, 1:]

    dV = np.transpose(D.T@np.linalg.pinv(D@D.T)@C).squeeze()
    V_i_p1 = V_i - dV
    x_i_p1[1:] = V_i_p1
    T_i_p1 = T_i

    tol = np.linalg.norm(C)
    dx = np.linalg.norm((x_i_p1 - x_i))
    print(f"Iteration {k}: tol = {tol} \t dx_k = {dx} \t dT = {T_i_p1 - T_i} \t Time Elapsed: {time.time() - start_time}")
    print(f"Old Non-Dim State: {x_i}")
    print(f"{dV}")
    print(f"New Non-Dim State: {x_i_p1}")
    return x_i_p1, T_i_p1

def OE_wo_a_e_i(z_f, x_i, x_i_p1, T_i, start_time, model, k):
    N = len(x_i)
    x_f = z_f[:N]

    x_dot_f = model.dOE_dt(x_f)
    phi_t0_tf = z_f[N:].reshape((N,N))

    V_i = x_i[3:]
    C = x_f - x_i
    D_original = phi_t0_tf - np.eye(N)
    D = D_original[:, 3:]

    dV = np.transpose(D.T@np.linalg.pinv(D@D.T)@C).squeeze()
    V_i_p1 = V_i - dV
    x_i_p1[3:] = V_i_p1
    T_i_p1 = T_i

    tol = np.linalg.norm(C)
    dx = np.linalg.norm((x_i_p1 - x_i))
    print(f"Iteration {k}: tol = {tol} \t dx_k = {dx} \t dT = {T_i_p1 - T_i} \t Time Elapsed: {time.time() - start_time}")
    print(f"Old Non-Dim State: {x_i}")
    print(f"{dV}")
    print(f"New Non-Dim State: {x_i_p1}")
    return x_i_p1, T_i_p1

def OE_wo_a_e_i__w_T(z_f, x_i, x_i_p1, T_i, start_time, model, k):
    N = len(x_i)
    x_f = z_f[:N]

    # C = x_f - x_i # F(V) in pdf
    # D = np.hstack([phi_t0_tf - np.eye(N), x_dot_f.reshape((N,-1))])

    x_dot_f = model.dOE_dt(x_f)
    phi_t0_tf = z_f[N:].reshape((N,N))

    V_i = np.hstack((x_i[3:], T_i))
    C = x_f - x_i
    D_original = phi_t0_tf - np.eye(N)
    D = D_original[:, 3:]
    D = np.hstack([D, x_dot_f.reshape((N,-1))])


    dV = np.transpose(D.T@np.linalg.pinv(D@D.T)@C).squeeze()
    V_i_p1 = V_i - dV
    x_i_p1[3:] = V_i_p1[0:3]
    T_i_p1 = V_i_p1[-1]

    tol = np.linalg.norm(C)
    dx = np.linalg.norm((x_i_p1 - x_i))
    print(f"Iteration {k}: tol = {tol} \t dx_k = {dx} \t dT = {T_i_p1 - T_i} \t Time Elapsed: {time.time() - start_time}")
    print(f"Old Non-Dim State: {x_i}")
    print(f"{dV}")
    print(f"New Non-Dim State: {x_i_p1}")
    return x_i_p1, T_i_p1


def OE_wo_a_e_i__w_T_inv(z_f, x_i, x_i_p1, T_i, start_time, model, k):
    N = len(x_i)
    x_f = z_f[:N]

    x_dot_f = model.dOE_dt(x_f)
    phi_t0_tf = z_f[N:].reshape((N,N))

    V_i = np.hstack((x_i[3:], T_i))
    C = x_f - x_i
    C = np.hstack((C, 1E-8/T_i))
    D_original = phi_t0_tf - np.eye(N)
    D = D_original[:, 3:]
    D = np.hstack([D, x_dot_f.reshape((N,-1))])
    D = np.vstack((D, [0, 0, 0, -1E-8/T_i**2]))

    dV = np.transpose(D.T@np.linalg.pinv(D@D.T)@C).squeeze()
    V_i_p1 = V_i - dV
    x_i_p1[3:] = V_i_p1[0:3]
    T_i_p1 = V_i_p1[-1]

    tol = np.linalg.norm(C)
    dx = np.linalg.norm((x_i_p1 - x_i))
    print(f"Iteration {k}: tol = {tol} \t dx_k = {dx} \t dT = {T_i_p1 - T_i} \t Time Elapsed: {time.time() - start_time}")
    print(f"Old Non-Dim State: {x_i}")
    print(f"{dV}")
    print(f"New Non-Dim State: {x_i_p1}")
    return x_i_p1, T_i_p1
