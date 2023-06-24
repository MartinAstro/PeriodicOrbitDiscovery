import os
from FrozenOrbits.analysis import check_for_intersection, print_state_differences
from FrozenOrbits.bvp import variable_time_mirror_bvp

import GravNN
import matplotlib.pyplot as plt
import numpy as np
from FrozenOrbits.gravity_models import (pinnGravityModel,
                                         polyhedralGravityModel)
from FrozenOrbits.LPE import LPE
from FrozenOrbits.utils import propagate_orbit
from FrozenOrbits.visualization import plot_3d_trajectory
from GravNN.CelestialBodies.Asteroids import Eros


def sample_mirror_orbit(R, mu):
    v_0 = np.sqrt(2*mu/R)
    init_state = np.array([R, 0, 0, 0, v_0, 0])
    T = 2*np.pi*np.sqrt(R**3/mu)
    return init_state, T



def main():
    """Solve a BVP problem using the dynamics of the cartesian state vector"""
    planet = Eros()
    np.random.seed(15)

    model = pinnGravityModel(os.path.dirname(GravNN.__file__) + \
        "/../Data/Dataframes/eros_pinn_III_031222_500R.data")  
    # model = polyhedralGravityModel(planet, planet.obj_8k)

    x_0, T = sample_mirror_orbit(planet.radius*5, planet.mu)

    # Run the solver
    x_0_sol, T_sol = variable_time_mirror_bvp(T, x_0, model)

    # propagate the initial and solution orbits
    init_sol = propagate_orbit(T, x_0, model, tol=1E-8) 
    bvp_sol = propagate_orbit(T_sol, x_0_sol, model, tol=1E-8) 

    check_for_intersection(bvp_sol, planet.obj_8k)
    print_state_differences(bvp_sol)

    plot_3d_trajectory(init_sol, planet.obj_8k)
    plt.title("Initial Guess")
    plot_3d_trajectory(bvp_sol, planet.obj_8k)
    plt.title("BVP Solution")

    plt.show()


if __name__ == "__main__":
    main()
