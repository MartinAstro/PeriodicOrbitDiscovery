import numpy as np
import pandas as pd
from GravNN.CelestialBodies.Asteroids import Eros
from GravNN.Networks.Model import load_config_and_model

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
def main():
    """Solve a BVP problem using the dynamics of the cartesian state vector"""

    max_time = 30*60 # 30 minutes
    np.random.seed(None)
    
    # file_path = pathlib.Path(__file__).parent.absolute()
    # file = file_path.as_posix() +  "/Data/BVP_Solutions/PINN_bvp.data"

    # Load in the gravity model
    planet = Eros()
    path = os.path.dirname(GravNN.__file__)
    df = pd.read_pickle(path + "/../Data/Dataframes/eros_grav_model_minus_pm.data")
    config, model  = load_config_and_model(df.iloc[-1]['id'], df)

    start_time = time.time()
    total_time = 0.0
    orbits_sol = []
    while total_time < max_time:
        trad_OE = sample_safe_trad_OE(planet.radius, planet.radius*10)
        T = compute_period(planet.mu, trad_OE[0,0])
        state = np.hstack(oe2cart_tf(trad_OE, planet.mu))
        lpe = LPE(model, config, planet.mu, element_set="traditional")

        # Run the solver
        results = solve_bvp_pos_problem(T, state, lpe, max_nodes=10000)
        results.time = time.time() - start_time
        orbits_sol.append(results)
        total_time = results.time

    
    path = os.path.dirname(FrozenOrbits.__file__)
    file = path +  "/../Data/BVP_Solutions/PINN_bvp.data"
    os.makedirs(os.path.dirname(file), exist_ok=True)
    with open(file, 'wb') as f:
        pickle.dump(orbits_sol, f) 


if __name__ == "__main__":
    main()
