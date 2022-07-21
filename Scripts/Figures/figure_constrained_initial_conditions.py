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

vis = VisualizationBase()
vis.fig_size = (2.1, 2.1*1.25)

plt.rc('font', size=8)
plt.rc('axes', labelsize=8)
plt.rc('axes', linewidth=1)
plt.rc('lines', markersize=1)
plt.rc('xtick.major', size=4)
plt.rc('xtick.minor', size=4)
plt.rc('ytick', labelsize=8)
plt.rc('xtick', labelsize=8)
plt.rc('xtick', direction='in')
plt.rc('ytick', direction='in')
plt.rc('axes', labelpad=1)
plt.rc('lines', markersize=6)

def get_original_IC(df, k):
    T_0 = df.iloc[k]["T_0"]
    X_0 = np.array(df.iloc[k]["X_0"])
    return T_0, X_0

def get_corrected_IC(df, k):
    T_0 = df.iloc[k]["T_0_sol"]
    X_0 = np.array(df.iloc[k]["X_0_sol"])
    return T_0, X_0

def get_integrated_orbit(T_0, X_0, model):
    t_eval = np.linspace(0, T_0, 500)
    orbit_sol = propagate_orbit(
        T_0, 
        X_0, 
        model, 
        tol=1E-7,
        t_eval=t_eval) 
    return orbit_sol

def gather_integrated_orbits(df):
    planet = Eros()
    model = pinnGravityModel(os.path.dirname(GravNN.__file__) + \
        "/../Data/Dataframes/eros_BVP_PINN_III.data")  
    
    T_0, X_0 = get_original_IC(df, 0) # use 0 b/c they all share same IC
    original_orbit = get_integrated_orbit(T_0, X_0, model)

    # Generate the raw data for plotting
    experiment_name_list = []
    cart_orbit_list = []
    OE_orbit_list = []

    # Add the original orbit to the list
    original_orbit_OE = cart2oe_tf(original_orbit.y.T, planet.mu).numpy()
    cart_orbit_list.append(original_orbit)
    OE_orbit_list.append(original_orbit_OE)
    experiment_name_list.append("Original")

    for k in range(len(df)):
        T_0_sol, X_0_sol = get_corrected_IC(df, k)
        corrected_orbit_sol = get_integrated_orbit(T_0_sol, X_0_sol, model)
        corrected_orbit_sol_OE = cart2oe_tf(corrected_orbit_sol.y.T, planet.mu).numpy()
        cart_orbit_list.append(corrected_orbit_sol)
        OE_orbit_list.append(corrected_orbit_sol_OE)
        experiment_name = df.index[k].replace("_", " ")
        if experiment_name == "cartesian":
            experiment_name = experiment_name.capitalize()
        experiment_name_list.append(experiment_name)

    return experiment_name_list, cart_orbit_list, OE_orbit_list, original_orbit

def plot_cartesian_orbits(experiment_name_list, cart_orbit_list, original_orbit):
    base_figure_name = os.path.basename(__file__).split('.py')[0]
    color_list = ['black', 'blue', 'red', 'green', 'purple']
    planet = Eros()
    # Plot original orbit and corresponding solution
    for k in range(len(cart_orbit_list)):
        plot_cartesian_state_3d(original_orbit.y.T, planet.obj_8k, new_fig=True, solid_color='black', line_opacity=0.5, label="Original",fig_size=vis.fig_size)
        plot_cartesian_state_3d(cart_orbit_list[k].y.T, None, new_fig=False, solid_color=color_list[k], label=experiment_name_list[k],fig_size=vis.fig_size)
        plt.legend(loc='upper center')
        name = experiment_name_list[k].replace(" ", "_")
        vis.save(plt.gcf(), f"{base_figure_name}_{name}_corrected.pdf")

def plot_OE(experiment_name_list, cart_orbit_list, OE_orbit_list):
    base_figure_name = os.path.basename(__file__).split('.py')[0]
    color_list = ['black', 'blue', 'red', 'green', 'purple']
    # vis.fig_size = (2.1, 1.8)
    # Make single OE figure
    vis.newFig() 
    for k in range(len(cart_orbit_list)):
        t_vec = cart_orbit_list[k].t
        plot_OE_1d(t_vec, OE_orbit_list[k], "traditional", 
                    color=color_list[k], 
                    new_fig=False, 
                    label=experiment_name_list[k], 
                    horizontal=True,
                    individual_figures=False)

    # Divide single figure into the individual figures
    element_ylabels = ['$a$ [m]', '$e$', '$i$ [rad]', '$\omega$ [rad]', '$\Omega$ [rad]', '$M$ [rad]']
    all_axes = plt.gcf().get_axes()
    for i in range(len(all_axes)):
        vis.newFig()
        axis = all_axes[i]
        for k in range(len(axis.lines)):
            line = axis.lines[k]
            plt.plot(line.get_xdata(), line.get_ydata(), c=line.get_color(), label=line.get_label())
        plt.xlabel("Time [s]")
        plt.ylabel(element_ylabels[i])
        plt.gca().xaxis.set_major_formatter(MathTextSciFormatter("%1.1e"))
        if element_ylabels[i] == '$a$ [m]':
            plt.gca().yaxis.set_major_formatter(MathTextSciFormatter("%1.2e"))

        vis.save(plt.gcf(), f"{base_figure_name}_OE{i}.pdf")

def plot_OE_diff(experiment_name_list, cart_orbit_list, OE_orbit_list):
    base_figure_name = os.path.basename(__file__).split('.py')[0]
    color_list = ['black', 'blue', 'red', 'green', 'purple']
    # vis.fig_size = (2.1, 1.8)
    # Make single OE figure
    vis.fig_size = (6.3, 2.1)
    fig_da, ax = vis.newFig() # da
    fig_di, ax = vis.newFig() # di 
    fig_dX, ax = vis.newFig() # dX
    OE_original = OE_orbit_list[0]
    
    for k in range(len(OE_orbit_list)):
        OE = OE_orbit_list[k]
        T = cart_orbit_list[k].t
        X = cart_orbit_list[k].y.T
        da = np.abs(OE[:,0] - OE_original[0,0])/OE_original[0,0] * 100
        di = np.abs(OE[:,2] - OE_original[0,2])/OE_original[0,2] * 100
        dX =  np.linalg.norm(X - X[0,:], axis=1)/ np.linalg.norm(X[0,:]) * 100
        if k == 0:
            color = 'black'
        elif k == 1:
            color = 'blue'
        elif k == 2: 
            color = 'red' 
        else:
            color = 'green'

        plt.figure(fig_da.number)
        plt.plot(T, da, color=color)
        plt.xlabel("Time [s]")
        plt.ylabel("Semi-Major Axis Error [\%]")
        # plt.ylabel(r"$\frac{|a(t) - a^{\star}|}{a^{\star}}$")
        plt.yscale("log")
        plt.ylim([10**-3, 10**2])
        plt.gca().xaxis.set_major_formatter(MathTextSciFormatter("%1.1e"))
        vis.save(plt.gcf(), f"{base_figure_name}_da.pdf")


        plt.figure(fig_di.number)
        plt.plot(T, di, color=color)
        plt.xlabel("Time [s]")
        plt.ylabel("Inclination Error [\%]")
        # plt.ylabel(r"$\frac{|i(t) - i^{\star}|}{i^{\star}}$")
        plt.yscale("log")
        plt.ylim([5*10**-5, 10**2])
        plt.gca().xaxis.set_major_formatter(MathTextSciFormatter("%1.1e"))
        vis.save(plt.gcf(), f"{base_figure_name}_di.pdf")

        plt.figure(fig_dX.number)
        plt.plot(T, dX, color=color)
        plt.xlabel("Time [s]")
        plt.ylabel("State Error [\%]")
        # plt.ylabel(r"$\frac{|X(t) - X(0)|}{X(0)}$")
        # plt.ylabel(r"$|\delta X|$")
        plt.yscale("log")
        plt.ylim([5**-3, 2*10**2])
        plt.gca().xaxis.set_major_formatter(MathTextSciFormatter("%1.1e"))
        vis.save(plt.gcf(), f"{base_figure_name}_dX.pdf")

    plt.show()

    # Divide single figure into the individual figures
    element_ylabels = ['$a$ [m]', '$e$', '$i$ [rad]', '$\omega$ [rad]', '$\Omega$ [rad]', '$M$ [rad]']
    all_axes = plt.gcf().get_axes()
    for i in range(len(all_axes)):
        vis.newFig()
        axis = all_axes[i]
        for k in range(len(axis.lines)):
            line = axis.lines[k]
            plt.plot(line.get_xdata(), line.get_ydata(), c=line.get_color(), label=line.get_label())
        plt.xlabel("Time [s]")
        plt.ylabel(element_ylabels[i])
        plt.gca().xaxis.set_major_formatter(MathTextSciFormatter("%1.1e"))
        if element_ylabels[i] == '$a$ [m]':
            plt.gca().yaxis.set_major_formatter(MathTextSciFormatter("%1.2e"))

        vis.save(plt.gcf(), f"{base_figure_name}_OE{i}.pdf")



def main():
    directory =  os.path.dirname(FrozenOrbits.__file__)+ "/Data/"
    df = pd.read_pickle(directory + "constrained_orbit_solutions.data")
    df = df.drop(index="OE_bounded")
    experiment_name_list, cart_orbit_list, OE_orbit_list, original_orbit = gather_integrated_orbits(df)
    plot_cartesian_orbits(experiment_name_list, cart_orbit_list, original_orbit)
    plot_OE_diff(experiment_name_list, cart_orbit_list, OE_orbit_list)
    # plot_OE(experiment_name_list, cart_orbit_list, OE_orbit_list)
    plt.show()

if __name__ == "__main__":
    main()
