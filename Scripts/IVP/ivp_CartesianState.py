from GravNN.Networks.utils import configure_tensorflow
from GravNN.Networks.Model import load_config_and_model
from GravNN.CelestialBodies.Planets import Earth
from GravNN.CelestialBodies.Asteroids import Eros,Toutatis
from GravNN.Support.ProgressBar import ProgressBar
from FrozenOrbits.coordinate_transforms import *
from FrozenOrbits.LPE import LPE
tf = configure_tensorflow()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pandas as pd
import panel as pn
pn.extension()

from scipy.integrate import solve_ivp
import OrbitalElements.orbitalPlotting as op
import OrbitalElements.oe as oe
from GravNN.Support.transformations import cart2sph, invert_projection
from FrozenOrbits.ivp import solve_ivp_pos_problem
from FrozenOrbits.visualization import plot_OE_suite_from_state_sol
from FrozenOrbits.utils import compute_period, compute_period_from_state
from FrozenOrbits.events import coarse_periodic
def main():
    planet = Eros()
    df = pd.read_pickle("Data/Dataframes/eros_grav_model_minus_pm.data")
    config, model  = load_config_and_model(df.iloc[-1]['id'], df)
    lpe = LPE(model, config, planet.mu, element_set='traditional')

    state = np.array([-6.36256532e+02, -4.58656092e+04,  1.31640352e+04,  3.17126984e-01, -1.12030801e+00, -3.38751010e+00])
    state = np.array([ 2.38751411e+04, 1.30276509e+04, 4.06603138e+04, 1.64428862e+00, 2.65557817e+00, -1.41360807e+00])
    state = np.array([-3.46263679e+04, 1.20752840e+04, -5.61351930e+04, -2.18777090e+00, 8.37627356e-01, 1.01480902e+00]) # Seed 2

    t_eval = None
    T = compute_period_from_state(planet.mu, state.reshape((1,6)))* 2
    sol = solve_ivp_pos_problem(T, state, lpe, t_eval, events=coarse_periodic, args=(state,))
    
    t_events = sol.t_events[0] # Grab all course periodic time stamps
    t_period = np.average(t_events[-2:])
    
    sol = solve_ivp_pos_problem(t_period, state, lpe, t_eval, events=coarse_periodic, args=(state,))


    plot_OE_suite_from_state_sol(sol, planet)
    op.plot_orbit_3d(sol.y[0:3,:])
    plt.show()


if __name__ == "__main__":
    main()