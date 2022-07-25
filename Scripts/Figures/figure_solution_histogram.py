from asyncio import gather
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



def get_original_IC(df, k):
    T_0 = df.iloc[k]["T_0"]
    X_0 = np.array(df.iloc[k]["X_0"])
    return T_0, X_0

def get_corrected_IC(df, k):
    T_0 = df.iloc[k]["T_0_sol"]
    X_0 = np.array(df.iloc[k]["X_0_sol"])
    return T_0, X_0

def get_integrated_orbit(T_0, X_0, model, orbit_multiplier=10):
    t_eval = np.linspace(0, T_0*orbit_multiplier, int(100*orbit_multiplier))
    orbit_sol = propagate_orbit(
        T_0*orbit_multiplier, 
        X_0, 
        model, 
        tol=1E-7,
        t_eval=t_eval) 
    return orbit_sol

def gather_integrated_orbits(df):
    planet = Eros()
    model = pinnGravityModel(os.path.dirname(GravNN.__file__) + \
        "/../Data/Dataframes/eros_BVP_PINN_III.data")  
    
    # Generate the raw data for plotting
    cart_orbit_list = []
    for k in range(len(df)):
        T_0_sol, X_0_sol = get_corrected_IC(df, k)
        corrected_orbit_sol = get_integrated_orbit(T_0_sol, X_0_sol, model)
        cart_orbit_list.append(corrected_orbit_sol)
    return cart_orbit_list

def plot_solution_histogram(cart_orbit_list, **kwargs):
    percent_error_list = []
    position_error_list = []
    for orbit in cart_orbit_list:
        X_0 = orbit.y[:,0]
        X_f = orbit.y[:,-1]
        dX = X_f - X_0
        dX_percent = np.abs(dX / X_0) * 100
        dX_percent_mag = np.average(dX_percent)
        percent_error_list.append(dX_percent_mag)
        position_error_list.append(np.linalg.norm(dX[0:3]))

    # range = kwargs.get('range', [0, np.max(percent_error_list)])
    range = kwargs.get('range', [0, np.max(position_error_list)])
    kwargs.update({"range" : range})
    # range=None
    print(position_error_list)
    plt.hist(position_error_list,
             bins=np.logspace(np.log10(1),np.log10(10**6), 50),
            #  range=[np.log10(1E-2), np.log10(5E2)],
             **kwargs)
    return range

def main():

    directory =  os.path.dirname(FrozenOrbits.__file__)+ "/Data/"
    fine_df = pd.read_pickle(directory + "fine_orbit_solutions.data")
    cart_fine_df = pd.read_pickle(directory + "cartesian_fine_orbit_solutions.data")

    sol_OE = gather_integrated_orbits(fine_df)
    sol_cart = gather_integrated_orbits(cart_fine_df)

    vis = VisualizationBase()
    vis.newFig()
    
    range = plot_solution_histogram(sol_cart, color='orange', label='Cartesian Shooting Method', alpha=0.8)
    plot_solution_histogram(sol_OE, color='blue', label='OE Shooting Method', alpha=0.8)#, range=range)
    plt.xscale('log')
    plt.legend()
    plt.xlabel("Cartesian State Error after 10 Orbits [m]")
    plt.ylabel("\# of Solutions")
    base_figure_name = os.path.basename(__file__).split('.py')[0]
    vis.save(plt.gcf(), f"{base_figure_name}.pdf")
    plt.show()
    

if __name__ == "__main__":
    main()
