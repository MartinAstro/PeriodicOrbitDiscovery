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

def get_initial_conditions(mu, element_set):
    # OE_trad = np.array([[7.93E+04/2, 1.00E-02, 1.53E+00, -7.21E-01, -1.61E-01, 2.09e+00]])
    OE_trad = np.array([[7.93E+04/2, 1.00E-01, 1.53E+00/2, -7.21E-01, -1.61E-01, 2.09e+00]])
    T = 2*np.pi*np.sqrt(OE_trad[0,0]**3/mu)
    OE_0 = oe2milankovitch_tf(OE_trad, mu).numpy()
    X_0 = oe2cart_tf(OE_0, mu, element_set)[0]
    return OE_0, X_0, T

def main():
    """Solve a BVP problem using the dynamics of the cartesian state vector"""
    planet = Eros()
    np.random.seed(15)
    # tf.config.run_functions_eagerly(True)

    element_set = 'milankovitch'
    OE_0, X_0, T = get_initial_conditions(planet.mu, element_set)
    
    model = pinnGravityModel(os.path.dirname(GravNN.__file__) + \
        "/../Data/Dataframes/eros_BVP_PINN_III.data")  

    # Desire |H_0| = 1
    # Acknowledging H = (t_star/l_star**2)*H_tilde 
    # Solve for l_star -> l_star = sqrt(t_star*H_tilde_mag/H_mag) 
    lpe = LPE_Milankovitch(model.gravity_model, planet.mu, 
                            l_star=np.sqrt(T*np.linalg.norm(OE_0[0,0:3])/1.0),  # H = t/l**2 * H_tilde
                            t_star=T, 
                            m_star=1.0)


    # normalized coordinates for semi-major axis
    bounds = ([-2*1/np.sqrt(3), -2*1/np.sqrt(3), -2*1/np.sqrt(3), -1.1, -1.1, -1.1, -np.inf],
              [ 2*1/np.sqrt(3),  2*1/np.sqrt(3),  2*1/np.sqrt(3),  1.1,  1.1,  1.1,  np.inf])
              
    bounds = ([ -np.inf,  -np.inf,  -np.inf,  -np.inf,  -np.inf,  -np.inf, -np.inf],
              [  np.inf,   np.inf,   np.inf,   np.inf,   np.inf,   np.inf,  np.inf])

    OE_0_sol, X_0_sol, T_sol = scipy_periodic_orbit_algorithm(T, OE_0, lpe, bounds, element_set)

    print(f"Initial OE: {OE_0}")
    print(f"BVP OE: {OE_0_sol}")

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
