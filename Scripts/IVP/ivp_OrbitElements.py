from GravNN.CelestialBodies.Asteroids import Eros
from GravNN.Networks.Model import load_config_and_model
from GravNN.Networks.utils import configure_tensorflow

from FrozenOrbits.coordinate_transforms import *
from FrozenOrbits.LPE import LPE

tf = configure_tensorflow()
import matplotlib.pyplot as plt
import numpy as np
import OrbitalElements.orbitalPlotting as op
import pandas as pd
from scipy.integrate import solve_ivp
from FrozenOrbits.ivp import solve_ivp_oe_problem

def main():
    planet = Eros()
    df = pd.read_pickle("Data/Dataframes/eros_grav_model_minus_pm.data")
    config, model  = load_config_and_model(df.iloc[-1]['id'], df)

    OE = np.array([[planet.radius*2, 0.01, np.pi/4, 0.1, 0.1, 0.1]]) 

    n = np.sqrt(planet.mu/OE[0,0]**3)
    T = 2*np.pi/n
    t_eval = None
    lpe = LPE(model, config, planet.mu, element_set='traditional')
    sol = solve_ivp_oe_problem(T, OE, lpe)

    # lpe = LPE(model, config, planet.mu, OE, element_set='Delaunay')
    # DelaunayOE = oe2delaunay_tf(OE, planet.mu).numpy()
    # solve_ivp_problem(T, DelaunayOE, lpe) 

    # Milankovitch actually requires the original OE as input
    # lpe = LPE(model, config, planet.mu, OE, element_set='Milankovitch')
    # MilankovitchOE = oe2milankovitch_tf(OE, planet.mu).numpy()
    # solve_ivp_problem(T, MilankovitchOE, lpe) 

    # lpe = LPE(model, config, planet.mu, OE, element_set='Equinoctial')
    # equinoctialOE = oe2equinoctial_tf(OE).numpy()
    # solve_ivp_problem(T, equinoctialOE, lpe, t_eval) 

    op.plot_OE(sol.t, sol.y, lpe.element_set)
    plt.show()

if __name__ == "__main__":
    main()
