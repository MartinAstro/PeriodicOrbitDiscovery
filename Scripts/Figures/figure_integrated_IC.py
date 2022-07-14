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
    return T_0, X_0, T_0_sol, X_0_sol, planet


def plot_integrated_orbits(coarse_df, fine_df, base_figure_name):
    vis = VisualizationBase()
    model = pinnGravityModel(os.path.dirname(GravNN.__file__) + \
        "/../Data/Dataframes/eros_BVP_PINN_III.data")  
    solid_color_list = ['blue', 'red', 'green']
    color_list = [plt.cm.Blues, plt.cm.Reds, plt.cm.Greens]
    for k in range(3):
        T_0, X_0, _, _, planet = sample_initial_conditions(coarse_df, k)
        _, _, T_0_sol, X_0_sol, planet = sample_initial_conditions(fine_df, k)

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
        plot_cartesian_state_3d(solution_orbit_sol.y.T, None, new_fig=False, solid_color=solid_color_list[k])
        vis.save(plt.gcf(), f"{base_figure_name}_{k}_corrected.pdf")

        # Corrected Orbit Propagated for 10 periods
        period_multiplier = 10
        t_eval_solution = np.linspace(0, T_0_sol*period_multiplier, 100*period_multiplier)
        solution_orbit_sol = propagate_orbit(
            T_0_sol*period_multiplier, 
            X_0_sol, 
            model, 
            tol=1E-7,
            t_eval=t_eval_solution) 
        plot_cartesian_state_3d(solution_orbit_sol.y.T, planet.obj_8k, new_fig=True, cmap=color_list[k])
        vis.save(plt.gcf(), f"{base_figure_name}_{k}_corrected_x10.pdf")


def main():

    directory =  os.path.dirname(FrozenOrbits.__file__)+ "/Data/"
    coarse_df = pd.read_pickle(directory + "coarse_orbit_solutions.data")
    fine_df = pd.read_pickle(directory + "fine_orbit_solutions.data")
    base_figure_name = os.path.basename(__file__).split('.py')[0]
    plot_integrated_orbits(coarse_df, fine_df, base_figure_name)

    coarse_df = pd.read_pickle(directory + "cartesian_coarse_orbit_solutions.data")
    fine_df = pd.read_pickle(directory + "cartesian_fine_orbit_solutions.data")
    base_figure_name = os.path.basename(__file__).split('.py')[0] + "_cart"
    plot_integrated_orbits(coarse_df, fine_df, base_figure_name)

    # plt.show()

if __name__ == "__main__":
    main()
