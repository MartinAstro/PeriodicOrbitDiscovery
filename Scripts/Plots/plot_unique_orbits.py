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

def load_data(model_name):
    path = os.path.dirname(FrozenOrbits.__file__)
    file = path +  "/../Data/BVP_Solutions/V2/" + model_name + "_stats.data"
    with open(file, 'rb') as f:
        results = pickle.load(f)
    return results    

def main():
    path = os.path.dirname(FrozenOrbits.__file__)
    np.random.seed(10)
    planet = Eros()
    df = pd.read_pickle(os.path.dirname(GravNN.__file__) + "/../Data/Dataframes/eros_grav_model_minus_pm.data")
    config, model  = load_config_and_model(df.iloc[-1]['id'], df)
    lpe = LPE(model, config, planet.mu, element_set="traditional")
    vis = VisualizationBase(formatting_style="AIAA")

    results = load_data("PINN")
    total_orbits = 5
    k = 0
    orbits_plotted = 0
    while orbits_plotted < total_orbits:
        print(k)
        if orbits_plotted == 0:
            new = True
            obj_file = Eros().model_potatok 
        else:
            new = False
            obj_file = None

        criteria = (results[k]['failure'] == "None")
        if criteria: 
            criteria = criteria * (results[k]['closest_approach'] < 3000)
            criteria = criteria * (np.linalg.norm(results[k]['closest_state'][3:6] - results[k]['initial_condition'][3:6]) < 0.3)
        if not criteria:
            k += 1
            continue
        else:
            orbits_plotted +=1 
        state = results[k]['initial_condition']

        lpe = LPE(model, config, planet.mu, element_set="traditional")

        sol = solve_ivp_pos_problem(results[k]['t_closest_approach'], state, lpe, t_eval=None, events=None, args=(state,))

        op.plot3d(sol.y, show=False, obj_file=obj_file, save=False, traj_cm=None, new_fig=new)
        plt.gca().view_init(65,45)
        k +=1


    
    vis.save(plt.gcf(), path +  "/../Plots/Unique_Orbits.pdf")
    plt.show()


if __name__ == "__main__":
    main()
