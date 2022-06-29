import os
import copy
from FrozenOrbits.analysis import check_for_intersection, print_state_differences
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

    # OE_0, X_0, T, planet = not_periodic_IC()
    # OE_0, X_0, T, planet = long_near_periodic_IC()
    OE_0, X_0, T, planet = near_periodic_IC_2()

    # Run the solver
    lpe = LPE_Traditional(model.gravity_model, planet.mu, 
                                l_star=OE_0[0,0], 
                                t_star=T, 
                                m_star=1.0)#planet.mu/(6.67430*1E-11))
    element_set = 'traditional'

    # normalized coordinates for semi-major axis
    bounds = ([0.7, 0.1, -np.pi, -np.inf, -np.inf, -np.inf],
              [1.0, 0.5, np.pi, np.inf, np.inf, np.inf])
    bounds = ([-np.inf, -np.inf, -np.pi, -np.inf, -np.inf, -np.inf, 0.9],
              [np.inf, np.inf, np.pi, np.inf, np.inf, np.inf, 1.1])

    decision_variable_mask = [False, False, True, True, True, True, True] # [OE, T] [N+1]
    constraint_angle_wrap_mask = [False, False, False, True, True, True] # wrap w, Omega, M # 

    OE_0_sol, X_0_sol, T_sol = scipy_periodic_orbit_algorithm_v2(T, OE_0, lpe, 
                                            bounds, element_set, decision_variable_mask, 
                                            constraint_angle_wrap_mask=constraint_angle_wrap_mask)
    # OE_0_sol, X_0_sol, T_sol = scipy_periodic_orbit_algorithm(T, OE_0, lpe, bounds, element_set)

    print(f"Initial OE: {OE_0} \t T: {T}")
    print(f"BVP OE: {OE_0_sol} \t T {T_sol}")

    # propagate the initial and solution orbits
    init_sol = propagate_orbit(T, X_0, model, tol=1E-7) 
    bvp_sol = propagate_orbit(T_sol, X_0_sol, model, tol=1E-7) 
    
    check_for_intersection(bvp_sol, planet.obj_8k)
    
    print_state_differences(init_sol)
    print_state_differences(bvp_sol)

    plot_cartesian_state_3d(init_sol.y.T, planet.obj_8k)
    plt.title("Initial Guess")

    plot_cartesian_state_3d(bvp_sol.y.T, planet.obj_8k)
    plt.title("BVP Solution")
    plt.show()

if __name__ == "__main__":
    main()
