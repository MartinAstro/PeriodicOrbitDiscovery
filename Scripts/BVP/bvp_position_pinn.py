import os

from numpy.lib.polynomial import poly

import GravNN
import matplotlib.pyplot as plt
import numpy as np
import trimesh
from FrozenOrbits.boundary_conditions import *
from FrozenOrbits.bvp import solve_bvp_pos_problem
from FrozenOrbits.coordinate_transforms import *
from FrozenOrbits.gravity_models import pinnGravityModel, polyhedralGravityModel
from FrozenOrbits.ivp import solve_ivp_pos_problem
from FrozenOrbits.LPE import LPE
from FrozenOrbits.utils import (
    Solution,
    check_solution_validity,
    compute_period,
    get_energy,
    get_initial_orbit_guess,
    get_S_matrix,
    get_solution_metrics,
    sample_safe_trad_OE,
)
from FrozenOrbits.visualization import plot_1d_solutions, plot_3d_solutions
from GravNN.CelestialBodies.Asteroids import Eros
from scipy.integrate import solve_bvp


def solve_bvp_pos_problem(
    t_mesh, y_guess, lpe, true_jac=False, tol=1e-4, bc_tol=1e0, max_nodes=300000
):
    fun, bc, fun_jac, bc_jac = get_pos_bc_fcns(y_guess, lpe, true_jac=true_jac)
    S = get_S_matrix(y_guess, lpe, option="None")
    results = solve_bvp(
        fun,
        bc,
        t_mesh,
        y_guess,
        S=S,
        p=None,
        fun_jac=fun_jac,
        bc_jac=bc_jac,
        verbose=2,
        max_nodes=max_nodes,
        tol=tol,
        bc_tol=bc_tol,
    )
    return results


def main():
    """Solve a BVP problem using the dynamics of the cartesian state vector"""
    planet = Eros()
    obj_file = planet.obj_8k
    init_nodes = 300
    max_nodes = 500
    dynamics_tol = 1e-3
    bc_tol = 1e0
    true_jac = True

    min_radius = planet.radius * 99.9999
    max_radius = planet.radius * 100
    np.random.seed(15)

    # Load in the gravity model
    model = pinnGravityModel(
        # os.path.dirname(GravNN.__file__) + "/../Data/Dataframes/eros_pinn_4.data"
        # os.path.dirname(GravNN.__file__) + "/../Data/Dataframes/eros_pinn_III_030822.data"
        os.path.dirname(GravNN.__file__) + "/../Data/Dataframes/eros_pinn_III_031222_500R.data"
    )  
    # model = polyhedralGravityModel(planet, planet.obj_8k)

    lpe = LPE(model, planet.mu, element_set="traditional")

    # Set the initial conditions and dynamics via OE
    trad_OE = sample_safe_trad_OE(min_radius, max_radius)
    T = compute_period(planet.mu, trad_OE[0, 0])
    init_state = np.hstack(oe2cart_tf(trad_OE, planet.mu))

    # Integrate state to produce guess for initial orbit
    t_mesh, y_guess, E = get_initial_orbit_guess(
        T, init_nodes, init_state, lpe, closed_loop=False, plot=True
    )

    # Run the solver
    results = solve_bvp_pos_problem(
        t_mesh,
        y_guess,
        lpe,
        max_nodes=max_nodes,
        true_jac=true_jac,
        tol=dynamics_tol,
        bc_tol=bc_tol,
    )

    mesh = trimesh.load_mesh(obj_file)
    proximity = trimesh.proximity.ProximityQuery(mesh)
    sol, valid = check_solution_validity(
        results, proximity, lpe, max_radius_in_km=planet.radius * 10 / 1e3
    )
    get_solution_metrics(results, sol)

    # Determine how bad first guess was
    t_mesh = np.linspace(0, T, 100)
    bvp_sol = Solution(results.sol(t_mesh), t_mesh)
    init_sol = solve_ivp_pos_problem(t_mesh[-1], init_state, lpe, t_eval=t_mesh)
    init_pos_diff = np.linalg.norm(init_sol.y[0:3, -1] - init_sol.y[0:3, 0])
    init_vel_diff = np.linalg.norm(init_sol.y[3:6, -1] - init_sol.y[3:6, 0])

    print("Initial Integrated Position Difference %f [m]" % (init_pos_diff))
    print("Initial Integrated Velocity Difference %f [m/s]" % (init_vel_diff))

    plot_3d_solutions(t_mesh, init_sol, results, sol, obj_file, y_scale=planet.radius)
    plot_1d_solutions(t_mesh, sol, y_scale=planet.radius)
    plot_1d_solutions(t_mesh, bvp_sol, new_fig=False, y_scale=planet.radius)
    plt.suptitle("Boundary vs Integrate Solutions")

    print(bvp_sol.y[:,0])
    plt.show()

if __name__ == "__main__":
    main()
