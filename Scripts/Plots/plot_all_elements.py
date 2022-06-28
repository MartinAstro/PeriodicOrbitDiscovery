import os
from FrozenOrbits.analysis import check_for_intersection, print_state_differences
from FrozenOrbits.bvp import *

import GravNN
import matplotlib.pyplot as plt
import numpy as np
from FrozenOrbits.boundary_conditions import *
from FrozenOrbits.gravity_models import pinnGravityModel
from FrozenOrbits.LPE import *
from FrozenOrbits.utils import propagate_orbit
from FrozenOrbits.visualization import *
from GravNN.CelestialBodies.Asteroids import Eros
from FrozenOrbits.constraints import *
from Scripts.BVP.initial_conditions import near_periodic_IC


def main():
    """Solve a BVP problem using the dynamics of the cartesian state vector"""
    OE_trad, X_0, T, planet = near_periodic_IC()
    
    model = pinnGravityModel(os.path.dirname(GravNN.__file__) + \
        "/../Data/Dataframes/eros_BVP_PINN_III.data")  


    X_0 = oe2cart_tf(OE_trad, planet.mu, 'traditional')[0]
    init_sol = propagate_orbit(T, X_0, model, tol=1E-7)     
    print_state_differences(init_sol)

    plot_cartesian_state_3d(init_sol.y.T, planet.obj_8k)
    plt.title("Initial Guess")

    # OE_trad = cart2trad_tf(init_sol.y.T, planet.mu).numpy()
    # OE_mil = cart2milankovitch_tf(init_sol.y.T, planet.mu).numpy()
    # OE_del = cart2delaunay_tf(init_sol.y.T, planet.mu).numpy()
    # OE_equi = cart2equinoctial_tf(init_sol.y.T, planet.mu).numpy()
    # plot_OE_1d(init_sol.t, OE_trad, 'traditional')
    # plot_OE_1d(init_sol.t, OE_mil, 'milankovitch')
    # plot_OE_1d(init_sol.t, OE_del, 'delaunay')
    # plot_OE_1d(init_sol.t, OE_equi, 'equinoctial')

    plt.show()


if __name__ == "__main__":
    main()
