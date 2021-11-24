from GravNN.Networks.Model import load_config_and_model
from GravNN.CelestialBodies.Asteroids import Eros
from FrozenOrbits.coordinate_transforms import *
from FrozenOrbits.LPE import LPE
import numpy as np
import pandas as pd
import pandas as pd
from FrozenOrbits.bvp import solve_bvp_oe_problem
from FrozenOrbits.visualization import plot_OE_results
from FrozenOrbits.utils import compute_period
def main():
    """Solve a BVP problem using the dynamics of the orbit elements via LPE"""

    # Load gravity model
    planet = Eros()
    df = pd.read_pickle("Data/Dataframes/eros_grav_model_minus_pm.data")
    config, model = load_config_and_model(df.iloc[-1]['id'], df)

    # Pick Initial Conditions via OE and set dynamics
    trad_OE = np.array([[planet.radius*3, 0.4, np.pi/4, np.pi/3, np.pi/3, np.pi/3]]) 
    # OE = oe2equinoctial_tf(trad_OE).numpy()
    T = compute_period(planet.mu, trad_OE[0,0])
    lpe = LPE(model, config, planet.mu, element_set="traditional")
    
    # Run the solver
    results = solve_bvp_oe_problem(T, trad_OE, lpe)
    plot_OE_results(results, T, lpe)

if __name__ == "__main__":
    main()