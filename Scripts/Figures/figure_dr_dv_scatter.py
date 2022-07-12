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


def main():
    model = pinnGravityModel(os.path.dirname(GravNN.__file__) + \
        "/../Data/Dataframes/eros_BVP_PINN_III.data")  
    planet = model.config['planet'][0]
    directory =  os.path.dirname(FrozenOrbits.__file__)+ "/Data/"
    coarse_df = pd.read_pickle(directory + "coarse_orbit_solutions.data")

    vis = VisualizationBase()
    vis.newFig()
    for k in range(len(coarse_df)):
        dX_0 = np.array(coarse_df["dX_0"][k])
        dX_0_sol = np.array(coarse_df["dX_sol"][k])

        r_0 = np.linalg.norm(dX_0[0:3])
        v_0 = np.linalg.norm(dX_0[3:6])
        r_0_sol = np.linalg.norm(dX_0_sol[0:3])
        v_0_sol = np.linalg.norm(dX_0_sol[3:6])

        dr = r_0_sol - r_0
        dv = v_0_sol - v_0

        plt.scatter(r_0, v_0, c='blue')
        plt.scatter(r_0_sol, v_0_sol, c='red')
        plt.arrow(r_0, v_0, dr, dv, head_length=np.linalg.norm([dr, dv])/10, length_includes_head=True,color='black')

    plt.xlabel("$\delta r$ [m]")
    plt.ylabel("$\delta v$ [m/s]")
    # plt.xscale('log')
    plt.show()

if __name__ == "__main__":
    main()
