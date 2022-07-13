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

        OE_tilde_0 = df['OE_0'].iloc[k].reshape((-1,6))
        lpe = LPE_Traditional(model.gravity_model, planet.mu, 
                                    l_star=OE_tilde_0[0,0]/1.0, 
                                    t_star=df['T_0'].iloc[k], 
                                    m_star=1.0)

        dOE_tilde_0 = df["dOE_0"].iloc[k].reshape((-1,6))
        dOE_tilde_0_sol = df["dOE_sol"].iloc[k].reshape((-1,6))

        dOE_0 = lpe.non_dimensionalize_state(dOE_tilde_0)
        dOE_0_sol = lpe.non_dimensionalize_state(dOE_tilde_0_sol)


        # Get scalar magnitudes
        dOE_tilde_0_mag = np.linalg.norm(dOE_tilde_0)
        dOE_tilde_0_sol_mag = np.linalg.norm(dOE_tilde_0_sol)
        
        dOE_0_mag = np.linalg.norm(dOE_0)
        dOE_0_sol_mag = np.linalg.norm(dOE_0_sol)
        
        # Difference these results for the arrow direction

        d_dOE_tilde = dOE_tilde_0_sol_mag - dOE_tilde_0_mag 
        d_dOE = dOE_0_sol_mag - dOE_0_mag 

        plt.scatter(dOE_tilde_0_mag, dOE_0_mag, c=start_color)
        plt.scatter(dOE_tilde_0_sol_mag, dOE_0_sol_mag, c=end_color)
        plt.arrow(dOE_tilde_0_mag, dOE_0_mag, d_dOE_tilde, d_dOE, head_length=np.linalg.norm([d_dOE_tilde, d_dOE])/10, length_includes_head=True,color='black')

    plt.xlabel(r"$\delta \tilde{\oe}$")
    plt.ylabel(r"$\delta \oe$")
    # plt.xscale('log')

if __name__ == "__main__":
    vis = VisualizationBase()
    vis.newFig()
    main("coarse_orbit_solutions.data", 'red', 'blue')
    main("fine_orbit_solutions.data", 'blue', 'green')
    plt.xscale('log')
    plt.yscale('log')
    vis.save(plt.gcf(), os.path.basename(__file__).split('.py')[0] + '.pdf')
    plt.show()