import os

import GravNN
import matplotlib.pyplot as plt
import numpy as np

from FrozenOrbits.analysis import check_for_intersection, print_state_differences
from FrozenOrbits.boundary_conditions import *
from FrozenOrbits.bvp import general_variable_time_bvp_trad_OE
from FrozenOrbits.gravity_models import pinnGravityModel
from FrozenOrbits.LPE import LPE_Traditional
from FrozenOrbits.utils import propagate_orbit
from FrozenOrbits.visualization import plot_cartesian_state_3d
from Scripts.BVP.initial_conditions import not_periodic_IC

# def get_initial_conditions(R, mu):
#     OE = np.array([[5*R, 0.3, np.pi/4, np.pi/4, np.pi/4, np.pi/4]]) # Good example
#     # OE = np.array([[R/2, 0.1, np.pi/4, np.pi/4, np.pi/4, np.pi/4]])
#     T = 2*np.pi*np.sqrt(OE[0,0]**3/mu)
#     x = trad2cart_tf(OE,mu).numpy()[0]
#     return OE, x, T



def main():
    """Solve a BVP problem using the dynamics of the cartesian state vector"""
    np.random.seed(15)

    model = pinnGravityModel(os.path.dirname(GravNN.__file__) + \
    "/../Data/Dataframes/eros_BVP_PINN_III.data")  
    # model = polyhedralGravityModel(planet, planet.obj_8k)
    OE_0, x_0, T, planet = not_periodic_IC()

    lpe = LPE_Traditional(model.gravity_model, planet.mu, 
                                l_star=OE_0[0,0], 
                                t_star=T, 
                                m_star=1.0)#planet.mu/(6.67430*1E-11))
                                

    # Run the solver
    decision_variable_mask = [False, False, True, True, True, True, True] # [OE, T] [N+1]
    constraint_angle_wrap_mask = [False, False, False, True, True, True] # wrap w, Omega, M # [xf-x0]
    OE_0_sol, x_0_sol, T_sol = general_variable_time_bvp_trad_OE(T, OE_0, lpe, 
                                element_set='traditional',
                                decision_variable_mask=decision_variable_mask,
                                constraint_angle_wrap_mask=constraint_angle_wrap_mask)

    # propagate the initial and solution orbits
    init_sol = propagate_orbit(T, x_0, model, tol=1E-8) 
    bvp_sol = propagate_orbit(T_sol, x_0_sol, model, tol=1E-8) 

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
