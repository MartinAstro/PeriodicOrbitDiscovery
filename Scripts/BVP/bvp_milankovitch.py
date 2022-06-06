import copy
import os

import GravNN
import matplotlib.pyplot as plt
import numpy as np
import OrbitalElements.orbitalPlotting as op
import trimesh
from FrozenOrbits.boundary_conditions import *
from FrozenOrbits.bvp import solve_bvp_oe_problem
from FrozenOrbits.coordinate_transforms import *
from FrozenOrbits.gravity_models import pinnGravityModel
from FrozenOrbits.ivp import solve_ivp_pos_problem
from FrozenOrbits.LPE import LPE
from FrozenOrbits.utils import (Solution, compute_period,
                                get_evolved_elements, get_initial_orbit_guess,
                                get_S_matrix, sample_safe_trad_OE, get_solution_metrics,
                                check_solution_validity)
from FrozenOrbits.visualization import (plot_1d_solutions, plot_3d_solutions,
                                        plot_OE_suite_from_state_sol)
from GravNN.CelestialBodies.Asteroids import Eros
from scipy.integrate import solve_bvp, solve_ivp

def solve_bvp_oe_problem(t_mesh, y_guess, lpe, true_jac=False, tol=1e-4, bc_tol=1e0, max_nodes=300000, element_set=None):
    fun, bc, fun_jac, bc_jac = get_milankovitch_bc_fcns(y_guess, lpe, true_jac=true_jac)
    S = get_S_matrix(y_guess, lpe, option=element_set)
    results = solve_bvp(fun, bc, t_mesh, y_guess, S=S, p=None, fun_jac=fun_jac, bc_jac=bc_jac, verbose=2, max_nodes=max_nodes, tol=tol, bc_tol=bc_tol)
    return results
    
def main():
    """Solve a BVP problem using the dynamics of the cartesian state vector"""
    planet = Eros()
    init_nodes = 500
    max_nodes = 100000
    dynamics_tol = 1e-3
    bc_tol = 1e0
    element_set = 'milankovitch'
    total_elements = 7
    true_jac = True

    min_radius = planet.radius*3
    max_radius = planet.radius*8
    np.random.seed(16)

    # Load in the gravity model
    model = pinnGravityModel(os.path.dirname(GravNN.__file__) + "/../Data/Dataframes/eros_pinn_4.data") # only goes up to 3R
    mesh = trimesh.load_mesh(planet.obj_8k)
    proximity = trimesh.proximity.ProximityQuery(mesh)

    # Set the initial conditions and dynamics via OE
    trad_OE = sample_safe_trad_OE(min_radius, max_radius)
    trad_OE[0,5] = 0.0
    T = compute_period(planet.mu, trad_OE[0,0])
    init_state = np.hstack(oe2cart_tf(trad_OE, planet.mu))


    lpe = LPE(model, planet.mu, element_set=element_set)
    
    state = copy.deepcopy(init_state)

    t_mesh, state_guess, E = get_initial_orbit_guess(T, init_nodes, state, lpe, closed_loop=False)
    y_guess = np.zeros((7, len(state_guess[0])))
    # TODO: Clean this up
    for i in range(len(state_guess.T)):
        y_guess[:,i] = cart2oe_tf(state_guess[:,i].reshape((1,-1)), planet.mu, element_set=element_set).numpy()
    
    sol = Solution(state_guess, t_mesh)

    # plot_OE_from_state_sol(sol, planet)
    plot_OE_suite_from_state_sol(sol, planet)
    plt.show()

    # Drop the anomaly term
    y_guess = y_guess[0:total_elements, :]

    # Integrate state to produce first loop
    # t_mesh, y_guess = get_evolved_elements(T, init_nodes, state, lpe, closed_loop=False)

    # Run the solver
    results = solve_bvp_oe_problem(t_mesh, y_guess, lpe, max_nodes=max_nodes, true_jac=true_jac, tol=dynamics_tol, bc_tol=bc_tol, element_set=element_set) #1e-4 is default, 1e-6 is different, 1e-12 is diff
    sol, valid = check_solution_validity(results, proximity, lpe, max_radius_in_km=planet.radius*10/1E3)

    get_solution_metrics(results, sol)
    state = sol.y[:,0] if valid else None



    # Determine how bad first guess was
    t_mesh = np.linspace(0, T, 100)
    init_sol = solve_ivp_pos_problem(t_mesh[-1], init_state, lpe, t_eval=t_mesh)
    init_pos_diff = np.linalg.norm(init_sol.y[0:3,-1] - init_sol.y[0:3,0])
    init_vel_diff = np.linalg.norm(init_sol.y[3:6,-1] - init_sol.y[3:6,0])

    print("Initial Integrated Position Difference %f [m]" % (init_pos_diff))
    print("Initial Integrated Velocity Difference %f [m/s]" % (init_vel_diff))

    plot_3d_solutions(t_mesh, init_sol, results, sol, Eros().obj_8k)

    plot_1d_solutions(t_mesh, sol)
    plot_1d_solutions(t_mesh, results.sol(t_mesh),new_fig=False)
    plt.suptitle("Boundary vs Integrate Solutions")
    
    plt.show()

if __name__ == "__main__":
    main()
