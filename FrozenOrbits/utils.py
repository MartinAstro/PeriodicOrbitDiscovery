from tkinter.ttk import Progressbar
import numpy as np
from FrozenOrbits.coordinate_transforms import cart2oe_tf


from scipy.fft import fftn, ifftn
from FrozenOrbits.dynamics import dynamics_cart
import os
import trimesh
import GravNN
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import OrbitalElements.orbitalPlotting as op
from GravNN.CelestialBodies.Asteroids import Eros
from GravNN.Support.ProgressBar import ProgressBar
from scipy.integrate import solve_ivp
class Solution:
    def __init__(self, y, t):
        self.y = y
        self.t = t

def compute_period(mu, a):
    if a > 0.0:
        n = np.sqrt(mu/a**3)
    else:
        n = np.sqrt(mu/(-a**3))
    T = 2*np.pi/n 
    return T

def compute_period_from_state(mu, state):
    oe = cart2oe_tf(state, mu).numpy()
    T = compute_period(mu, oe[0,0])
    return T

def sample_safe_trad_OE(R_min, R_max):
    a = np.random.uniform(R_min, R_max)
    e_1 = 1. - R_min/a
    e_2 = R_max/a - 1.
    e = np.random.uniform(0, np.min([e_1, e_2]))

    trad_OE = np.array([[a, 
                        e, 
                        np.random.uniform(0.0, np.pi),
                        np.random.uniform(0.0, 2*np.pi),
                        np.random.uniform(0.0, 2*np.pi),
                        np.random.uniform(0.0, 2*np.pi)]]) 
    return trad_OE


def get_energy(state, lpe):
    R = np.array([state[0:3]])
    V = np.array([state[3:6]])
    KE = 1.0/2.0*1*np.linalg.norm(V,axis=1)**2
    U_NN = lpe.model.generate_potential(R).T
    return (KE + U_NN).squeeze()

def get_evolved_elements(T, initial_nodes, state, lpe, plot=False, closed_loop=True):
    t_mesh, step = np.linspace(0, T, initial_nodes-1, retstep=True)
    sol = solve_ivp_oe_problem(T, state, lpe, t_eval=t_mesh)
    y = sol.y
    return t_mesh, y


def get_initial_orbit_guess(T, initial_nodes, state, lpe, plot=False, closed_loop=True):
    t_mesh, step = np.linspace(0, T, initial_nodes-1, retstep=True)
    sol = solve_ivp_pos_problem(T, state, lpe, t_eval=t_mesh)
    y = sol.y

    y_fft = fftn(y,axes=[1])
    N_kept = 25
    for i in range(len(y_fft)):
        y_fft_sorted_idx = np.argsort(np.abs(y_fft[i]))[:len(y_fft[0]) - N_kept]
        y_fft[i, y_fft_sorted_idx] = 0.0

    y_new = ifftn(y_fft, axes=(1))
    y_new = np.hstack((y_new, y_new[:,0].reshape((6,1))))
    t_mesh_new = np.hstack((t_mesh, [t_mesh[-1] + step]))
   
    if plot:
        sol = solve_ivp_pos_problem(t_mesh_new[-1], y_new[:,0].real, lpe, t_eval=t_mesh_new)
        op.plot3d(y[0:3], obj_file=Eros().obj_8k) # Original Orbit
        op.plot3d(y_new[0:3].real, obj_file=Eros().obj_8k) # Periodic Terms
        op.plot3d(sol.y, obj_file=Eros().obj_8k) # Propogated Periodic Terms
        plt.show()
        
    E0 = get_energy(y_new[:,0].real, lpe)
    state = y
    t_mesh = t_mesh

    if closed_loop:
        state = y_new.real
        t_mesh = t_mesh_new
    
    return t_mesh, state, E0

def get_S_matrix(y_guess, lpe, option="None"):
    if option == "None":
        S = None
    elif option == "OE_trad":
        y0 = y_guess[:,0]

        a, e, i, o, O, M = y0

        # Need to account for the fact that ya shifts
        S = np.zeros((6,6))
        # y1*y2 - y2*y1

        S[0,:] = [e, -a, 0, 0, 0, 0]
        S[1,:] = [a*e, -a**2, 0, 0, 0, 0]
        S[2,:] = [a**1.01*e, -a**2.01, 0, 0, 0, 0]
        S[3,:] = [a**1.02*e, -a**2.02, 0, 0, 0, 0]
        S[4,:] = [a**1.03*e, -a**2.03, 0, 0, 0, 0]
        S[5,:] = [a**1.04*e, -a**2.04, 0, 0, 0, 0]
        
        # S[0:3,0:3] = np.diag(y0[3:6]) 
        # S[0:3,3:6] = -np.diag(y0[0:3])

        # # y1*y2**2 - y2*(y1*y2)
        # S[3:6,0:3] = np.diag(y0[3:6])**2
        # S[3:6,3:6] = -np.diag(y0[0:3])*np.diag(y0[3:6])
    
    elif option == 'milankovitch':
        S = np.zeros((7,7)) 
        y0 = y_guess[:,0]
        h1, h2, h3, e1, e2, e3, L = y0
        h_mag = np.linalg.norm(y0[0:3])

        S[0,:6] = np.array([h3, 0, -h1, 0, 0, 0])/h_mag
        S[1,:6] = np.array([h2, -h1, 0, 0, 0, 0])/h_mag
        S[2,:6] = np.array([0, h3, -h2, 0, 0, 0])/h_mag
        S[3,:6] = [L**2, 0, 0, 0, 0, -h1*L]
        # S[3,:6] = [e1, e2, e3, -h1, -h2, -h3]
        S[4,:] = [-L, 0, 0, 0, 0, 0, h1]
        S[5,:] = [0, -L, 0, 0, 0, 0, h2]
        S[6,:] = [0, 0, -L, 0, 0, 0, h3]
        S /= h_mag
        
    else:
        S = np.zeros((7,7))
        y0 = y_guess[:,0]
        R, V = y0[0:3].reshape((1,3)), y0[3:6].reshape((1,3))
        KE = 1.0/2.0*1*np.linalg.norm(V,axis=1)**2
        U_NN = lpe.model.generate_potential(R).numpy().T

        S[0:3,0:3] = np.diag(y0[3:6])
        S[0:3,3:6] = -np.diag(y0[0:3])

        S[3:6,0:3] = np.diag(y0[3:6])**2
        S[3:6,3:6] = -np.diag(y0[0:3])*np.diag(y0[3:6])

        S[6, 0] = U_NN.astype(np.float64)/y0[0]
        S[6, 3] = KE/y0[3]
        S[6, 6] = -1.0
    return S




def check_solution_validity(results, proximity, lpe, max_radius_in_km):
    # Check that the integrated solution doesn't violate any conditions
    # i.e. run into the surface of the asteroid. 
    sol = None
    valid = True
    success = results.success
    singular_jacobian = np.any(np.isnan(results.rms_residuals))

    if not success:
        print("BVP Solver terminated without converging")
    
    if singular_jacobian:
        print("BVP solver encountered singular jacobian and failed")
        valid = False

    if np.max(results.rms_residuals) > 1:
        print("Dynamics Residual too large!")
        valid = False
    
    if valid:
        t_mesh = np.linspace(0, results.x[-1], 100)
        sol = solve_ivp_pos_problem(t_mesh[-1], results.sol(0)[:6], lpe, t_eval=t_mesh)

        traj = sol.y[0:3, :]
        cart_pos_in_km = traj/1E3 
        distance = proximity.signed_distance(cart_pos_in_km.reshape((3,-1)).T) + 0.5

        if np.any(distance > 0):
            print("Integrated Solution Intersected Body!")
            valid = False
        elif np.any(distance < -max_radius_in_km):
            print("Unreliable gravity model, exiting")
            valid = False

    return sol, valid


def get_solution_metrics(results, sol):
    bc_pos_diff = np.linalg.norm(results.sol(results.x[-1])[0:3] - results.sol(0)[0:3])
    print("Boundary Position Difference \t %f [m]" % (bc_pos_diff))

    # if invalid solution doesn't exist, can only analyze the 
    if sol is not None:
        pos_diff = np.linalg.norm(sol.y[0:3,-1] - sol.y[0:3,0])
        vel_diff = np.linalg.norm(sol.y[3:6,-1] - sol.y[3:6,0])
        
        print("Integrated Position Difference \t %f [m]" % (pos_diff))
        print("Integrated Velocity Difference \t %f [m/s]" % (vel_diff))
    else:
        pos_diff = None
        vel_diff = None

    return bc_pos_diff, pos_diff, vel_diff

def propagate_orbit(T, x_0, model, tol=1E-12, t_eval=None):
    if t_eval is None:
        t_eval = np.linspace(0, T, 100)
    pbar = ProgressBar(T, enable=True)
    sol = solve_ivp(dynamics_cart, 
                    [0, T], 
                    x_0, 
                    args=(model, pbar),
                    atol=tol, rtol=tol, 
                    t_eval=t_eval)
    pbar.close()
    return sol

def calc_angle_diff(theta_0, theta_f):
    #https://stackoverflow.com/questions/1878907/how-can-i-find-the-difference-between-two-angles -- not the top answer
    d_theta = np.arctan2(np.sin(theta_f-theta_0), np.cos(theta_f-theta_0))
    return d_theta


if __name__ == "__main__":
    eps = 1E-6
    print(calc_angle_diff(0.0, 2*np.pi - eps))
    print(calc_angle_diff(0.0, 2*np.pi + eps))
    
    print(calc_angle_diff(np.pi, 2*np.pi - eps))
    print(calc_angle_diff(np.pi, 2*np.pi + eps))