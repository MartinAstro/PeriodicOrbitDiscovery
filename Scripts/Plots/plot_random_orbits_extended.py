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
from FrozenOrbits.coordinate_transforms import oe2cart_tf
from FrozenOrbits.utils import compute_period, sample_safe_trad_OE


def main():
    path = os.path.dirname(FrozenOrbits.__file__)
    np.random.seed(10)
    planet = Eros()
    df = pd.read_pickle(os.path.dirname(GravNN.__file__) + "/../Data/Dataframes/eros_grav_model_minus_pm.data")
    config, model  = load_config_and_model(df.iloc[-1]['id'], df)
    lpe = LPE(model, config, planet.mu, element_set="traditional")
    vis = VisualizationBase(formatting_style="AIAA")
    start = 0
    total_orbits = 1
    for i in range(start, total_orbits):
        print(i)
        if i == 0:
            new = True
            obj_file = Eros().model_potatok 
        else:
            new = False
            obj_file = None

        for _ in range(3):
            trad_OE = sample_safe_trad_OE(planet.radius, planet.radius*10)
        T = compute_period(planet.mu, trad_OE[0,0])
        state = np.hstack(oe2cart_tf(trad_OE, planet.mu))
        lpe = LPE(model, config, planet.mu, element_set="traditional")
        t_coarse_mesh = np.linspace(T-(T/3), T+(T/3), 5000)

        sol = solve_ivp_pos_problem(T*1.5, state, lpe, t_eval=t_coarse_mesh, events=None, args=(state,))

        dstate = np.linalg.norm(sol.y - state.reshape((6,1)), axis=0)
        time = sol.t[np.where(dstate == np.min(dstate))][0]
        t_fine_mesh = np.linspace(time-(time/5), time+(time/5), 5000)

        sol = solve_ivp_pos_problem(T*1.5, state, lpe, t_eval=t_fine_mesh, events=None, args=(state,))

        dstate = np.linalg.norm(sol.y - state.reshape((6,1)), axis=0)
        time = sol.t[np.where(dstate == np.min(dstate))]

        sol = solve_ivp_pos_problem(time[0]*5, state, lpe, t_eval=None, events=None, args=(state,))

        op.plot3d(None, sol.y, None, show=False, obj_file=obj_file, save=False, traj_cm=None, new_fig=new)
        plt.gca().view_init(65,45)

    
    vis.save(plt.gcf(), path +  "/../Plots/Random_Orbit_Extended.pdf")
    plt.show()


if __name__ == "__main__":
    main()
