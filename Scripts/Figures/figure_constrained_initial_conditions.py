from locale import D_FMT
import os
import copy
import time
from FrozenOrbits.analysis import check_for_intersection, print_OE_differences, print_state_differences
from FrozenOrbits.bvp import *

import FrozenOrbits
import GravNN
import matplotlib.pyplot as plt
import numpy as np
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
    T_0= df.iloc[k]["T_0"]
    X_0 = np.array(df.iloc[k]["X_0"])
    T_0_sol = df.iloc[k]["T_0_sol"]
    X_0_sol = np.array(df.iloc[k]["X_0_sol"])

    # Don't include the bounded OE scenario
    # replace with the initial conditions instead
    if df.index[k] == "OE_Bounded":
        T_0= df.iloc[k]["T_0"]
        X_0 = np.array(df.iloc[k]["X_0"])
        T_0_sol = df.iloc[k]["T_0"]
        X_0_sol = np.array(df.iloc[k]["X_0"])

    return T_0, X_0, T_0_sol, X_0_sol, planet

def plot_integrated_orbits(df, base_figure_name):
    vis = VisualizationBase()
    model = pinnGravityModel(os.path.dirname(GravNN.__file__) + \
        "/../Data/Dataframes/eros_BVP_PINN_III.data")  
    solid_color_list = ['blue', 'red', 'green']
    color_list = [plt.cm.Blues, plt.cm.Reds, plt.cm.Greens]
    
    OE_fig, OE_ax = vis.newFig() # for future use

    color_idx = 0
    for k in range(len(df)):
        T_0, X_0, T_0_sol, X_0_sol, planet = sample_initial_conditions(df, k)


        # The original IC should be in black
        experiment_name = df.index[k]
        if experiment_name == 'OE_bounded':
            color = 'black'
        else:
            color = solid_color_list[color_idx]
            color_idx += 1

        if experiment_name != "OE_bounded":
            # Original Orbit
            t_eval_original = np.linspace(0, T_0, 100)
            original_orbit_sol = propagate_orbit(
                T_0, 
                X_0, 
                model, 
                tol=1E-7,
                t_eval=t_eval_original) 
            plot_cartesian_state_3d(original_orbit_sol.y.T, planet.obj_8k, new_fig=True, solid_color='black', line_opacity=0.5)
            
            # Corrected Orbit
            t_eval_solution = np.linspace(0, T_0_sol, 100)
            solution_orbit_sol = propagate_orbit(
                T_0_sol, 
                X_0_sol, 
                model, 
                tol=1E-7,
                t_eval=t_eval_solution) 
            plot_cartesian_state_3d(solution_orbit_sol.y.T, None, new_fig=False, solid_color=color)
            vis.save(plt.gcf(), f"{base_figure_name}_{color_idx}_corrected.pdf")
        else:
            t_eval_solution = np.linspace(0, T_0_sol, 100)
            solution_orbit_sol = propagate_orbit(
                T_0_sol, 
                X_0_sol, 
                model, 
                tol=1E-7,
                t_eval=t_eval_solution) 
        plt.figure(OE_fig.number)
        OE = cart2oe_tf(solution_orbit_sol.y.T, planet.mu).numpy()
        plot_OE_1d(solution_orbit_sol.t, OE, "traditional", 
                    color=color, 
                    new_fig=False, 
                    label=experiment_name, 
                    horizontal=True)
    vis.save(plt.gcf(), f"{base_figure_name}_OE.pdf")



def main():
    directory =  os.path.dirname(FrozenOrbits.__file__)+ "/Data/"
    df = pd.read_pickle(directory + "constrained_orbit_solutions.data")
    base_figure_name = os.path.basename(__file__).split('.py')[0]
    plot_integrated_orbits(df, base_figure_name)
    plt.show()

if __name__ == "__main__":
    main()
