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


def sample_initial_conditions(df, k):
    planet = Eros()
    OE_0 = np.array([df.iloc[k]["OE_0_sol"]])
    T_0 = df.iloc[k]["T_0_sol"]
    X_0 = np.array(df.iloc[k]["X_0_sol"])
    return OE_0, X_0, T_0, planet

def main():
    """Solve a BVP problem using the dynamics of the cartesian state vector"""
    np.random.seed(15)

    model = pinnGravityModel(os.path.dirname(GravNN.__file__) + \
        "/../Data/Dataframes/eros_BVP_PINN_III.data")  

    directory =  os.path.dirname(FrozenOrbits.__file__)+ "/Data/"
    coarse_df = pd.read_pickle(directory + "coarse_orbit_solutions.data")

    df = pd.DataFrame({
            "T_0" : [], "T_0_sol" : [],
            "OE_0" : [],  "OE_0_sol" : [],
            "X_0" : [], "X_0_sol" : [],
            "dOE_0" : [], "dOE_sol" : [],
            "dX_0" : [], "dX_sol" : [],   
            "result" : []    
        })

    for k in range(10):
        print(f"Iteration {k}")

        OE_0, X_0, T_0, planet = sample_initial_conditions(coarse_df, k)
        OE_original = np.array([coarse_df.iloc[k]["OE_0"]])
        T_original = np.array([coarse_df.iloc[k]["T_0"]])
        scale = 1.0 #/ OE_0[0,0]
        lpe = LPE_Traditional(model.gravity_model, planet.mu, 
                                    l_star=OE_original[0,0]/scale, 
                                    t_star=T_original, 
                                    m_star=1.0
                                    theta_star=2*np.pi)
        start_time = time.time()

        # Shooting solvers
        bounds = ([0.7*scale, 0.001, -np.pi, -2*np.pi, -2*np.pi, -2*np.pi, 0.9],
                [1.1*scale, 0.5, np.pi, 2*np.pi, 2*np.pi, 2*np.pi, 2.0])
        decision_variable_mask = [True, True, True, True, True, True, True] # [OE, T] [N+1]
        constraint_variable_mask = [True, True, True, True, True, True, False] 
        constraint_angle_wrap = [False, False, False, True, True, True, False] 
        solver = ShootingLsSolver(lpe, 
                                decision_variable_mask,
                                constraint_variable_mask,
                                constraint_angle_wrap,
                                max_nfev=50,
                                atol=1E-6,
                                rtol=1E-6) 

        OE_0_sol, X_0_sol, T_0_sol, results = solver.solve(OE_0, T_0, bounds)
        
        elapsed_time = time.time() - start_time
        print(f"Time Elapsed: {elapsed_time}")
        print(f"Initial OE: {OE_0} \t T: {T_0}")
        print(f"BVP OE: {OE_0_sol} \t T {T_0_sol}")

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
            "index" : k,
            "T_0" : [T_0], "T_0_sol" : [T_0_sol],
            "OE_0" : [OE_0[0]], "OE_0_sol" : [OE_0_sol[0]],
            "X_0" : [X_0], "X_0_sol" : [X_0_sol],
            "dOE_0" : [dOE_0], "dOE_sol" : [dOE_sol],
            "dX_0" : [dX_0], "dX_sol" : [dX_sol],       
            "elapsed_time" : [elapsed_time],
            "result" : [results]
        }
        df_k = pd.DataFrame().from_dict(data)
        df = pd.concat([df, df_k], axis=0)

    directory =  os.path.dirname(FrozenOrbits.__file__)+ "/Data/"
    os.makedirs(directory, exist_ok=True)
    pd.to_pickle(df, directory + "fine_orbit_solutions.data")


if __name__ == "__main__":
    main()
