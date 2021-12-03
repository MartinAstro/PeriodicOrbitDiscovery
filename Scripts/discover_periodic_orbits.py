import numpy as np
import pandas as pd
from GravNN.CelestialBodies.Asteroids import Eros
from GravNN.Networks.Model import load_config_and_model
from FrozenOrbits.gravity_models import polyhedralGravityModel, simpleGravityModel, noneGravityModel

from FrozenOrbits.bvp import solve_bvp_pos_problem
from FrozenOrbits.coordinate_transforms import *
from FrozenOrbits.LPE import LPE
from FrozenOrbits.visualization import plot_pos_results
from FrozenOrbits.utils import compute_period, sample_safe_trad_OE
import pickle
import os
import time
import pathlib

import FrozenOrbits
import GravNN

class Result:
    def __init__(self):
        self.time = None
        self.y = None
        pass

def main():
    """Solve a BVP problem using the dynamics of the cartesian state vector"""

    num_IC = 100 # 30 minutes
    np.random.seed(10)

    # Load in the gravity model
    planet = Eros()
    path = os.path.dirname(GravNN.__file__)
    df = pd.read_pickle(path + "/../Data/Dataframes/eros_grav_model_minus_pm.data")
    config, model  = load_config_and_model(df.iloc[-1]['id'], df)
    model_name = "PINN"

    # model = polyhedralGravityModel(planet, planet.obj_200k)
    # model_name = "Poly"

    # model = noneGravityModel(planet.mu)
    # model_name = "Simple"

    start_time = time.time()
    k = 0
    orbits_sol = []
    while k < num_IC:
        print("Num IC = " + str(k))
        trad_OE = sample_safe_trad_OE(planet.radius, planet.radius*10)
        T = compute_period(planet.mu, trad_OE[0,0])
        state = np.hstack(oe2cart_tf(trad_OE, planet.mu))
        lpe = LPE(model, config, planet.mu, element_set="traditional")

        if model_name == "Simple":
            results = Result()
            results.y = state.reshape((6,1))
            results.time = time.time() - start_time
        else:
            # Run the solver
            results = solve_bvp_pos_problem(T, state, lpe, max_nodes=10000, tol=1e-6, bc_tol=1e3)
            results.time = time.time() - start_time
        orbits_sol.append(results)
        k +=1
    
    path = os.path.dirname(FrozenOrbits.__file__)
    file = path +  "/../Data/BVP_Solutions/V2/" + model_name + "_bvp_solutions.data"
    os.makedirs(os.path.dirname(file), exist_ok=True)
    with open(file, 'wb') as f:
        pickle.dump(orbits_sol, f) 


if __name__ == "__main__":
    main()
