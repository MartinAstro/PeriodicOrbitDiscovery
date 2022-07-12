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
    df = pd.read_pickle(directory + "propagation_time_error.data")

    vis = VisualizationBase(save_directory=os.path.dirname(FrozenOrbits.__file__) + "/../Plots/")
    vis.newFig()
    for k in range(len(df)):
        dt_pinn = df["dt_pinn"][k]
        dt_poly = df["dt_poly"][k]
        
        Xf_pinn = df["Xf_pinn"][k]
        Xf_poly = df["Xf_poly"][k]

        dX = Xf_pinn - Xf_poly
        
        dX_mag = np.linalg.norm(dX)
        dX_percent_error = np.linalg.norm(dX)/np.linalg.norm(Xf_poly)*100

        dr = np.linalg.norm(Xf_pinn[0:3] - Xf_poly[0:3])

        dt_percent_error = (1 - np.abs(dt_poly-dt_pinn)/dt_poly)*100      

        plt.scatter(dt_percent_error, dr, c='blue')

    plt.xlabel("Fraction of Propagation Time [\%]")
    plt.ylabel("Position Error [m]")
    # plt.ylabel("Percent State Error [\%]")
    # plt.xscale('log')
    
    # TODO: make save default to latest figure and the name of the file
    vis.save(plt.gcf(), os.path.basename(__file__).split('.py')[0] + '.pdf')
    plt.show()

if __name__ == "__main__":
    main()
