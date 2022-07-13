import os
import copy
import time
import pandas as pd
from FrozenOrbits.analysis import check_for_intersection, print_OE_differences, print_state_differences
from FrozenOrbits.bvp import *

import GravNN
import matplotlib.pyplot as plt
import numpy as np
import FrozenOrbits
from FrozenOrbits.boundary_conditions import *
from FrozenOrbits.gravity_models import (pinnGravityModel,
                                         polyhedralGravityModel)
from FrozenOrbits.LPE import *
from FrozenOrbits.utils import propagate_orbit
from FrozenOrbits.visualization import *
from GravNN.CelestialBodies.Asteroids import Eros
import OrbitalElements.orbitalPlotting as op
from FrozenOrbits.constraints import *
from Scripts.BVP.initial_conditions import *

np.random.seed(15)

def sample_initial_conditions():
    planet = Eros()
    a = np.random.uniform(3*planet.radius, 7*planet.radius)
    e = np.random.uniform(0.1, 0.3)
    i = np.random.uniform(-np.pi, np.pi)
    omega = np.random.uniform(0.0, 2*np.pi)
    Omega = np.random.uniform(0.0, 2*np.pi)
    M = np.random.uniform(0.0, 2*np.pi)

    trad_OE = np.array([[a, e, i, omega, Omega, M]])
    X = trad2cart_tf(trad_OE,planet.mu).numpy()[0]
    T = 2*np.pi*np.sqrt(trad_OE[0,0]**3/planet.mu)
    return trad_OE, X, T, planet

def solve_cart(model, OE_0, X_0, T_0, planet, experiment):
        scale = 1.0 
        lpe = LPE_Cartesian(model.gravity_model, planet.mu, 
                                    l_star=1.0,#np.linalg.norm(OE_0[0,0:3])/scale, 
                                    t_star=T_0, 
                                    m_star=1.0)
        start_time = time.time()

        # Shooting solvers
        bounds = ([-np.inf, -np.inf,-np.inf,-np.inf,-np.inf,-np.inf, 0.9],
              [ np.inf,  np.inf, np.inf, np.inf, np.inf, np.inf, 1.1])
        decision_variable_mask = [True, True, True, True, True, True, True] # [OE, T] [N+1]
        constraint_variable_mask = [True, True, True, True, True, True, False] 
        constraint_angle_wrap = [False, False, False, True, True, True, False] 

        solver = CartesianShootingLsSolver(lpe, 
                                decision_variable_mask,
                                constraint_variable_mask,
                                constraint_angle_wrap,
                                max_nfev=10) 

        OE_0_sol, X_0_sol, T_0_sol, results = solver.solve(OE_0, T_0, bounds)
        elapsed_time = time.time() - start_time

        # propagate the initial and solution orbits
        init_sol = propagate_orbit(T_0, X_0, model, tol=1E-7) 
        bvp_sol = propagate_orbit(T_0_sol, X_0_sol, model, tol=1E-7) 
        
        valid = check_for_intersection(bvp_sol, planet.obj_8k)
        
        dX_0 = print_state_differences(init_sol)
        dX_sol = print_state_differences(bvp_sol)

        OE_trad_init = cart2trad_tf(init_sol.y.T, planet.mu).numpy()
        OE_trad_bvp = cart2trad_tf(bvp_sol.y.T, planet.mu).numpy()

        dOE_0, dOE_0_dimless = print_OE_differences(OE_trad_init, lpe, "IVP", constraint_angle_wrap)
        dOE_sol, dOE_sol_dimless = print_OE_differences(OE_trad_bvp, lpe, "BVP", constraint_angle_wrap)

        data = {
            "experiment" : [experiment],
            "T_0" : [T_0], "T_0_sol" : [T_0_sol],
            "OE_0" : [OE_0[0]], "OE_0_sol" : [OE_0_sol[0]],
            "X_0" : [X_0], "X_0_sol" : [X_0_sol],
            "dOE_0" : [dOE_0], "dOE_sol" : [dOE_sol],
            "dX_0" : [dX_0], "dX_sol" : [dX_sol],       
            "elapsed_time" : [elapsed_time],
            "result" : [results]
        }
        return data


def solve_OE(model, OE_0, X_0, T_0, planet, experiment):
        scale = 1.0 
        lpe = LPE_Traditional(model.gravity_model, planet.mu, 
                                    l_star=OE_0[0,0]/scale, 
                                    t_star=T_0, 
                                    m_star=1.0)
        start_time = time.time()

        # Shooting solvers
        
        
        if experiment == "OE_constrained":
            bounds = ([0.7*scale, 0.1, -np.pi, -2*np.pi, -2*np.pi, -2*np.pi, 0.9],
                    [1.1*scale, 0.5, np.pi, 2*np.pi, 2*np.pi, 2*np.pi, 2.0])
            decision_variable_mask = [False, True, False, True, True, True, True] # [OE, T] [N+1]
            constraint_variable_mask = [True, True, True, True, True, True, False] 
            constraint_angle_wrap = [False, False, False, True, True, True, False] 
        elif experiment == "OE_bounded":
            i = OE_0[0,2]
            bounds = ([0.9*scale, 0.1, i-np.pi/8, -2*np.pi, -2*np.pi, -2*np.pi, 0.9],
                    [1.1*scale, 0.5, i+np.pi/8, 2*np.pi, 2*np.pi, 2*np.pi, 2.0])
            decision_variable_mask = [True, True, True, True, True, True, True] # [OE, T] [N+1]
            constraint_variable_mask = [True, True, True, True, True, True, False] 
            constraint_angle_wrap = [False, False, False, True, True, True, False] 
        else:
            bounds = ([0.1*scale, 0.1, -np.pi, -2*np.pi, -2*np.pi, -2*np.pi, 0.9],
                    [np.inf*scale, 0.5, np.pi, 2*np.pi, 2*np.pi, 2*np.pi, 2.0])
            decision_variable_mask = [True, True, True, True, True, True, True] # [OE, T] [N+1]
            constraint_variable_mask = [True, True, True, True, True, True, False] 
            constraint_angle_wrap = [False, False, False, True, True, True, False] 

        solver = ShootingLsSolver(lpe, 
                                decision_variable_mask,
                                constraint_variable_mask,
                                constraint_angle_wrap,
                                max_nfev=10) 

        OE_0_sol, X_0_sol, T_0_sol, results = solver.solve(OE_0, T_0, bounds)
        elapsed_time = time.time() - start_time

        # propagate the initial and solution orbits
        init_sol = propagate_orbit(T_0, X_0, model, tol=1E-7) 
        bvp_sol = propagate_orbit(T_0_sol, X_0_sol, model, tol=1E-7) 
        
        valid = check_for_intersection(bvp_sol, planet.obj_8k)
        
        dX_0 = print_state_differences(init_sol)
        dX_sol = print_state_differences(bvp_sol)

        OE_trad_init = cart2trad_tf(init_sol.y.T, planet.mu).numpy()
        OE_trad_bvp = cart2trad_tf(bvp_sol.y.T, planet.mu).numpy()

        dOE_0, dOE_0_dimless = print_OE_differences(OE_trad_init, lpe, "IVP", constraint_angle_wrap)
        dOE_sol, dOE_sol_dimless = print_OE_differences(OE_trad_bvp, lpe, "BVP", constraint_angle_wrap)

        data = {
            "experiment" : [experiment],
            "T_0" : [T_0], "T_0_sol" : [T_0_sol],
            "OE_0" : [OE_0[0]], "OE_0_sol" : [OE_0_sol[0]],
            "X_0" : [X_0], "X_0_sol" : [X_0_sol],
            "dOE_0" : [dOE_0], "dOE_sol" : [dOE_sol],
            "dX_0" : [dX_0], "dX_sol" : [dX_sol],       
            "elapsed_time" : [elapsed_time],
            "result" : [results]
        }
        return data

def main():
    """Solve a BVP problem using the dynamics of the cartesian state vector"""

    model = pinnGravityModel(os.path.dirname(GravNN.__file__) + \
        "/../Data/Dataframes/eros_BVP_PINN_III.data")  

    planet = model.config['planet'][0]
    df = pd.DataFrame({
            "T_0" : [], "T_0_sol" : [],
            "OE_0" : [],  "OE_0_sol" : [],
            "X_0" : [], "X_0_sol" : [],
            "dOE_0" : [], "dOE_sol" : [],
            "dX_0" : [], "dX_sol" : [],     
            "result" : []  
        })

    experiment_list = ['cartesian', 'OE', 'OE_constrained', 'OE_bounded']
    OE_0, X_0, T_0, planet = sample_initial_conditions()

    for experiment in experiment_list:
        print(f"Experiment: {experiment}")
        if experiment == 'cartesian':
            data = solve_cart(model, np.array([X_0]), X_0, T_0, planet, experiment)
        else:
            data = solve_OE(model, OE_0, X_0, T_0, planet, experiment)
        df_k = pd.DataFrame().from_dict(data).set_index('experiment')
        df = pd.concat([df, df_k], axis=0)

    directory =  os.path.dirname(FrozenOrbits.__file__)+ "/Data/"
    os.makedirs(directory, exist_ok=True)
    pd.to_pickle(df, directory + "constrained_orbit_solutions.data")


if __name__ == "__main__":
    main()