import numpy as np
import matplotlib.pyplot as plt
from GravNN.Networks.Model import load_config_and_model
from GravNN.Visualization.VisualizationBase import VisualizationBase
from GravNN.CelestialBodies.Asteroids import Eros
import pathlib
import os
import GravNN
import FrozenOrbits
import pickle
from FrozenOrbits.ivp import solve_ivp_pos_problem
import OrbitalElements.orbitalPlotting as op
import pandas as pd
from FrozenOrbits.LPE import LPE
from FrozenOrbits.coordinate_transforms import oe2cart_tf, cart2oe_tf
from FrozenOrbits.utils import compute_period, sample_safe_trad_OE
from FrozenOrbits.visualization import plot_OE_results, plot_OE_suite_from_state_sol, plot_OE_from_state_sol

def main():
    path = os.path.dirname(FrozenOrbits.__file__)
    np.random.seed(2)
    planet = Eros()
    df = pd.read_pickle(os.path.dirname(GravNN.__file__) + "/../Data/Dataframes/eros_grav_model_minus_pm.data")
    config, model  = load_config_and_model(df.iloc[-1]['id'], df)
    lpe = LPE(model, config, planet.mu, element_set="traditional")
    vis = VisualizationBase()

    trad_OE = sample_safe_trad_OE(planet.radius, planet.radius*10)  
    T = compute_period(planet.mu, trad_OE[0,0])*3
    state = np.hstack(oe2cart_tf(trad_OE, planet.mu))

    sol = solve_ivp_pos_problem(T, state, lpe, t_eval=None, events=None, args=(state,))

    plot_OE_from_state_sol(sol, planet, OE_set="traditional")
    plt.gcf().set_size_inches(9, 5)
    P = T/3
    for i in range(1,7):
        plt.subplot(3,2,i)
        for x in [P, 2*P, 3*P]:
            plt.axvline(x, linestyle='--', c='black')
            if i >=5:
                plt.xlabel("Time Elapsed [s]")
    plt.suptitle(None)
    vis.save(plt.gcf(), path +  "/../Plots/Random_Orbits_Elements.pdf")
    plt.show()


if __name__ == "__main__":
    main()
