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