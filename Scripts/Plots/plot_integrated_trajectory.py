import os
import copy
import time
from FrozenOrbits.analysis import check_for_intersection, print_OE_differences, print_state_differences
from FrozenOrbits.bvp import *

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
    """Solve a BVP problem using the dynamics of the cartesian state vector"""
    np.random.seed(15)
    # tf.config.run_functions_eagerly(True)

    model = pinnGravityModel(os.path.dirname(GravNN.__file__) + \
        "/../Data/Dataframes/eros_BVP_PINN_III.data")  

    OE_0, X_0, T, planet = crazy_IC()

    # Run the solver
    # scale = 2*np.pi
    scale = 1.0
    lpe = LPE_Traditional(model.gravity_model, planet.mu, 
                                l_star=OE_0[0,0]/scale, 
                                t_star=T, 
                                m_star=1.0)#planet.mu/(6.67430*1E-11))

    start_time = time.time()

    # Shooting solvers
    bounds = ([0.7*scale, 0.01, -np.pi, -2*np.pi, -2*np.pi, -2*np.pi, 0.9],
              [1.0*scale, 0.5, np.pi, 2*np.pi, 2*np.pi, 2*np.pi, 1.1])

    
    print(f"Time Elapsed: {time.time()-start_time}")
    print(f"Initial OE: {OE_0} \t T: {T}")

    # propagate the initial and solution orbits
    init_sol = propagate_orbit(T, X_0, model, tol=1E-7) 

    print_state_differences(init_sol)
    plot_cartesian_state_3d(init_sol.y.T, planet.obj_8k)
    plt.title("Initial Guess")


    OE_trad_init = cart2trad_tf(init_sol.y.T, planet.mu).numpy()
    # print_OE_differences(OE_trad_init, lpe, "IVP", constraint_angle_wrap)
    plot_OE_1d(init_sol.t, OE_trad_init, 'traditional', y0_hline=True)

    plt.show()

if __name__ == "__main__":
    main()
