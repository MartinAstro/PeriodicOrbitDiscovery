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


def main(df_filename, start_color, end_color):
    model = pinnGravityModel(os.path.dirname(GravNN.__file__) + \
        "/../Data/Dataframes/eros_BVP_PINN_III.data")  
    planet = model.config['planet'][0]
    directory =  os.path.dirname(FrozenOrbits.__file__)+ "/Data/"
    df = pd.read_pickle(directory + df_filename)

    for k in range(len(df)):
        dX_0 = np.array(df["dX_0"].iloc[k])
        dX_0_sol = np.array(df["dX_sol"].iloc[k])

        r_0 = np.linalg.norm(dX_0[0:3])
        v_0 = np.linalg.norm(dX_0[3:6])
        r_0_sol = np.linalg.norm(dX_0_sol[0:3])
        v_0_sol = np.linalg.norm(dX_0_sol[3:6])

        dr = r_0_sol - r_0
        dv = v_0_sol - v_0

        plt.scatter(r_0, v_0, c=start_color)
        plt.scatter(r_0_sol, v_0_sol, c=end_color)
        plt.arrow(r_0, v_0, dr, dv, head_length=np.linalg.norm([dr, dv])/10, length_includes_head=True,color='black')

    plt.xlabel("$\delta r$ [m]")
    plt.ylabel("$\delta v$ [m/s]")

if __name__ == "__main__":

    vis = VisualizationBase()
    vis.newFig()
    main("coarse_orbit_solutions.data", 'red', 'blue')
    main("fine_orbit_solutions.data", 'blue', 'green')
    vis.save(plt.gcf(), os.path.basename(__file__).split('.py')[0] + '.pdf')
    plt.show()

