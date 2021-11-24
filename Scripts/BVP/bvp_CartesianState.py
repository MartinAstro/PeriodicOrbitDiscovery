import numpy as np
import pandas as pd
from GravNN.CelestialBodies.Asteroids import Eros
from GravNN.Networks.Model import load_config_and_model

from FrozenOrbits.bvp import solve_bvp_pos_problem
from FrozenOrbits.coordinate_transforms import *
from FrozenOrbits.LPE import LPE
from FrozenOrbits.visualization import plot_pos_results
from FrozenOrbits.utils import compute_period, sample_safe_trad_OE
def main():
    """Solve a BVP problem using the dynamics of the cartesian state vector"""

    # Load in the gravity model
    planet = Eros()
    df = pd.read_pickle("Data/Dataframes/eros_grav_model_minus_pm.data")
    config, model  = load_config_and_model(df.iloc[-1]['id'], df)

    # Set the initial conditions and dynamics via OE
    #trad_OE = np.array([[planet.radius*2, 0.1, np.pi/3, np.pi/2, np.pi/2, 0.]]) 
    # np.random.seed(2)
    trad_OE = sample_safe_trad_OE(planet.radius, planet.radius*10)

    T = compute_period(planet.mu, trad_OE[0,0])
    state = np.hstack(oe2cart_tf(trad_OE, planet.mu))
    lpe = LPE(model, config, planet.mu, element_set="traditional")

    # Run the solver
    results = solve_bvp_pos_problem(T, state, lpe, max_nodes=10000)

    print(results.y[:,0], sep=',')
    plot_pos_results(results, T, lpe, obj_file=Eros().model_potatok)



if __name__ == "__main__":
    main()
