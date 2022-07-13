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
    OE_0 = np.array([df["OE_0_sol"][k]])
    T_0 = df["T_0_sol"][k]
    X_0 = np.array(df["X_0_sol"][k])

    # Don't include the bounded OE scenario
    # replace with the initial conditions instead
    if df.index[k] == "OE_Bounded":
        OE_0 = np.array([df["OE_0"][k]])
        T_0 = df["T_0"][k]
        X_0 = np.array(df["X_0"][k])

    return OE_0, X_0, T_0, planet


def main():
    model = pinnGravityModel(os.path.dirname(GravNN.__file__) + \
        "/../Data/Dataframes/eros_BVP_PINN_III.data")  

    directory =  os.path.dirname(FrozenOrbits.__file__)+ "/Data/"
    df = pd.read_pickle(directory + "constrained_orbit_solutions.data")
    vis = VisualizationBase()
    period_multiplier = 1
    for k in range(len(df)):
        new_fig = True if k == 0 else False
        OE_0, X_0, T, planet = sample_initial_conditions(df, k)           
        init_sol = propagate_orbit(T*period_multiplier, X_0, model, tol=1E-7, t_eval=np.linspace(0, T*period_multiplier, 100*period_multiplier)) 
        plot_cartesian_state_3d(init_sol.y.T, planet.obj_8k, new_fig=new_fig)
        plt.title("Initial Conditions")


    vis.save(plt.gcf(), os.path.basename(__file__).split('.py')[0] + '.pdf')
    plt.show()

if __name__ == "__main__":
    main()
