import os
from FrozenOrbits.analysis import check_for_intersection, print_OE_differences, print_state_differences
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
from initial_conditions import * 



def main():
    """Solve a BVP problem using the dynamics of the cartesian state vector"""
    planet = Eros()
    np.random.seed(15)
    tf.config.run_functions_eagerly(True)

    OE_0, X_0, T, planet = near_periodic_IC()
    OE_0 = oe2equinoctial_tf(OE_0, planet.mu).numpy()
    # OE_0 = oe2milankovitch_tf(OE_0, planet.mu).numpy()

    model = pinnGravityModel(os.path.dirname(GravNN.__file__) + \
        "/../Data/Dataframes/eros_BVP_PINN_III.data")  

    # Desire |H_0| = 1
    # Acknowledging H = (t_star/l_star**2)*H_tilde 
    # Solve for l_star -> l_star = sqrt(t_star*H_tilde_mag/H_mag) 

    lpe = LPE_Equinoctial(model.gravity_model, planet.mu, 
                            l_star=OE_0[0,0],  # H = t/l**2 * H_tilde
                            t_star=T, 
                            m_star=1.0)


    # normalized coordinates for semi-major axis
    bounds = ([0.7, -1.0, -1.0, -np.inf, -np.inf, -2*np.pi, 0.9],
              [ 1.3, 1.0, 1.0, np.inf, np.inf,  2*np.pi, 1.1])
              

    # Shooting solvers
    start_time = time.time()
    decision_variable_mask = [True, True, True, True, True, True, True ] # [OE, T] [N+1]
    constraint_variable_mask = [True, True, True, True, True, True, False] #Don't use period as part of the constraint, just as part of jacobian
    constraint_angle_wrap_mask = [False, False, False, False, False, True, False]
    solver = ShootingLsSolver(lpe, decision_variable_mask, constraint_variable_mask, constraint_angle_wrap_mask) # Finds a local optimum, step size gets too small

    OE_0_sol, X_0_sol, T_sol, results = solver.solve(OE_0, T, bounds)
    
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

    OE_trad_init = cart2oe_tf(init_sol.y.T, planet.mu, lpe.element_set).numpy()
    OE_trad_bvp = cart2oe_tf(bvp_sol.y.T, planet.mu, lpe.element_set).numpy()

    print_OE_differences(OE_trad_init, lpe, "IVP", constraint_angle_wrap_mask)
    print_OE_differences(OE_trad_bvp, lpe, "BVP", constraint_angle_wrap_mask)

    plot_OE_1d(init_sol.t, OE_trad_init, lpe.element_set, y0_hline=True)
    plot_OE_1d(bvp_sol.t, OE_trad_bvp, lpe.element_set, y0_hline=True)
    plt.show()


if __name__ == "__main__":
    main()
