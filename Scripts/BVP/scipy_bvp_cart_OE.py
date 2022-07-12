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

    OE_0, X_0, T, planet = near_periodic_IC()
    # OE_0, X_0, T, planet = near_periodic_IC_2()
    # OE_0, X_0, T, planet = long_near_periodic_IC()

    # Run the solver
    # scale = 2*np.pi
    scale = 1.0
    lpe = LPE_Cartesian(model.gravity_model, planet.mu, 
                                l_star=np.linalg.norm(X_0[0:3])/scale, 
                                t_star=T, 
                                m_star=1.0)#planet.mu/(6.67430*1E-11))

    start_time = time.time()

    # Shooting solvers
    bounds = ([-np.inf, -np.inf,-np.inf,-np.inf,-np.inf,-np.inf, 0.9],
              [ np.inf,  np.inf, np.inf, np.inf, np.inf, np.inf, 1.1])
    decision_variable_mask = [True, True, True, True, True, True, True] # [OE, T] [N+1]
    constraint_variable_mask = [True, True, True, True, True, True, False] #Don't use period as part of the constraint, just as part of jacobian
    # solver = CartesianShootingRootSolver(lpe, decision_variable_mask, constraint_variable_mask) 
    solver = CartesianShootingLsSolver(lpe, decision_variable_mask, constraint_variable_mask) 
    OE_0_sol, X_0_sol, T_sol, results = solver.solve(np.array([X_0]), T, bounds)
    
    print(f"Time Elapsed: {time.time()-start_time}")
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

    OE_trad_init = cart2trad_tf(init_sol.y.T, planet.mu).numpy()
    OE_trad_bvp = cart2trad_tf(bvp_sol.y.T, planet.mu).numpy()

    print_OE_differences(OE_trad_init, lpe, "IVP", constraint_variable_mask)
    print_OE_differences(OE_trad_bvp, lpe, "BVP", constraint_variable_mask)

    plot_OE_1d(init_sol.t, OE_trad_init, 'traditional', y0_hline=True)
    plot_OE_1d(bvp_sol.t, OE_trad_bvp, 'traditional', y0_hline=True)

    plt.show()

if __name__ == "__main__":
    main()
