""" 
Adding in the jacobians for the boundary conditions

"""
import os
from FrozenOrbits.ivp import solve_ivp_pos_problem
import GravNN
import time
import copy
import trimesh
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt

from GravNN.CelestialBodies.Asteroids import Eros
from GravNN.Networks.Model import load_config_and_model
from GravNN.Support.transformations import cart2sph, invert_projection

from FrozenOrbits.bvp import solve_bvp_pos_problem
from FrozenOrbits.coordinate_transforms import *
from FrozenOrbits.LPE import LPE
from FrozenOrbits.visualization import plot_pos_results, plot_1d_solutions, plot_3d_solutions
from FrozenOrbits.utils import compute_period, sample_safe_trad_OE
from FrozenOrbits.boundary_conditions import *
from FrozenOrbits.coordinate_transforms import *
from FrozenOrbits.LPE import LPE
from FrozenOrbits.gravity_models import pinnGravityModel, polyhedralGravityModel

import OrbitalElements.orbitalPlotting as op

from scipy.integrate import solve_bvp, solve_ivp
from FrozenOrbits.utils import get_energy, get_initial_orbit_guess, get_S_matrix, get_solution_metrics, check_solution_validity



def solve_bvp_pos_problem(t_mesh, y_guess, lpe, true_jac=False, tol=1e-4, bc_tol=1e0, max_nodes=300000):
    fun, bc, fun_jac, bc_jac = get_pos_bc_fcns(y_guess, lpe, true_jac=True)
    S = get_S_matrix(y_guess, lpe, option="None")
    results = solve_bvp(fun, bc, t_mesh, y_guess, S=S, p=None, fun_jac=fun_jac, bc_jac=bc_jac, verbose=2, max_nodes=max_nodes, tol=tol, bc_tol=bc_tol)
    return results
    
def get_closest_approach(state_list, period_list, lpe, title):
        closest_approach_list = []
        for i in range(len(state_list)):
            state = state_list[i]
            T = period_list[i]
            t_mesh = np.linspace(0, T, 100)
            t_fine_mesh = np.linspace(T - T/5, T + T/5, 1000)
            t_mesh = np.append(t_mesh, t_fine_mesh)
            t_mesh.sort()
            t_mesh = np.unique(t_mesh)
            sol = solve_ivp_pos_problem(T + T/5 + 1, state, lpe, t_eval=t_mesh)

            # op.plot3d(sol.y[0:3],obj_file=planet.obj_8k, new_fig=True)
            # plt.title(title + str(i))
            closest_approach = np.min(np.linalg.norm((sol.y[:,50:] - sol.y[:,0].reshape((6,1)))[0:3], axis=0))
            closest_approach_list.append(closest_approach)
        return closest_approach_list

class Solution:
    def __init__(self, y, t):
        self.y = y
        self.t = t


def main():
    """
    Sample a random orbit and attempt to find a periodic orbit in its vicinity,
    repeat until time runs out. 
    """
    planet = Eros()
    iterations = 1
    init_nodes = 10000 # does increasing nodes improve performance
    max_nodes = init_nodes + 20000
    dynamics_tol = 5e-6

    # init_nodes = 10000 # does increasing nodes improve performance
    # max_nodes = init_nodes + 20000
    # dynamics_tol = 1e-6

    bc_tol = 1e3 # does setting lower tolerances improve performance
    true_jac = False
    time_limit = 60*3
    min_radius = planet.radius*1
    max_radius = planet.radius*10
    np.random.seed(16)

    # Load in the gravity model
    output_file = "bvp_pos_poly.data"
    model = polyhedralGravityModel(planet, planet.obj_200k) # only goes up to 3R

    # output_file = "bvp_pos_pinn.data"
    # model = pinnGravityModel(os.path.dirname(GravNN.__file__) + "/../Data/Dataframes/eros_pinn_III_030822.data") # only goes up to 3R

    mesh = trimesh.load_mesh(planet.obj_200k)
    proximity = trimesh.proximity.ProximityQuery(mesh)

    lpe = LPE(model, None, planet.mu, element_set="traditional")
    k = 0

    init_idx_list = []
    sol_idx_list = []

    init_state_list = []
    sol_state_list = []

    init_period_list = []
    sol_period_list = []

    time_list = []
    start_time = time.time()
    time_elapsed = 0
    while time_elapsed < time_limit:
        
        print("Iteration \t %d" % (k))
        trad_OE = sample_safe_trad_OE(min_radius, max_radius)
        init_state = np.hstack(oe2cart_tf(trad_OE, planet.mu))
        T = compute_period(planet.mu, trad_OE[0,0])

        state = copy.deepcopy(init_state)
        init_state_list.append(state)
        init_period_list.append(T)
        init_idx_list.append(k)

        # Check if IC already satisfy BC
        closest_approach = get_closest_approach([init_state], [T], lpe, None)
        if np.max(closest_approach) < bc_tol:
            time_elapsed = time.time() - start_time
            time_list.append(time_elapsed)
            sol_state_list.append(init_state)
            sol_period_list.append(T)
            sol_idx_list.append(k)
            k+=1
            continue

        # Integrate state to produce first loop
        t_mesh, y_guess, E = get_initial_orbit_guess(T, init_nodes, state, lpe, closed_loop=False)

        # Run the solver
        results = solve_bvp_pos_problem(t_mesh, y_guess, lpe, max_nodes=max_nodes, true_jac=true_jac, tol=dynamics_tol, bc_tol=bc_tol) #1e-4 is default, 1e-6 is different, 1e-12 is diff
        #sol, valid = check_solution_validity(results, proximity, lpe, max_radius_in_km=planet.radius*10/1E3)
        #bc_pos_diff, pos_diff, vel_diff = get_solution_metrics(results, sol)

        # if valid and pos_diff < bc_tol:
        time_elapsed = time.time() - start_time
        time_list.append(time_elapsed)

        if results.success:
            sol_state_list.append(results.y[:,0])
            sol_period_list.append(T)
            sol_idx_list.append(k)

        k += 1

    print("BVP Loop Ended!")

    initial_closest_approach = get_closest_approach(init_state_list, init_period_list, lpe, "INIT")
    solution_closest_approach = get_closest_approach(sol_state_list, sol_period_list, lpe, "BVP")

    num_sol_from_initial_guess = np.count_nonzero(np.array(initial_closest_approach) < bc_tol)
    num_sol_from_solutions = np.count_nonzero(np.array(solution_closest_approach) < bc_tol)
    
    trials = len(init_state_list)
    print("Initial guess which satisfy constraints: " + str(num_sol_from_initial_guess) + "/" + str(trials))
    print("BVP Successes: " + str(len(sol_state_list)) + "/" + str(trials))
    print("Integrated BVP solutions which satisfy constraints: " + str(num_sol_from_solutions) + "/" + str(trials))

    print(initial_closest_approach)
    print(solution_closest_approach)

    data = {
        'init_idx_list' : init_idx_list,
         'sol_idx_list' : sol_idx_list,

         'init_state_list' : init_state_list,
         'sol_state_list' : sol_state_list,

         'init_period_list' : init_period_list,
         'sol_period_list' : sol_period_list,

         'time_list' : time_list,
    }
    with open(output_file, 'wb') as f:
        pickle.dump(data, f)

    # plot_3d_solutions(t_mesh, init_sol, results, sol, Eros().obj_8k)
    # plot_1d_solutions(t_mesh, sol)
    # plot_1d_solutions(t_mesh, results.sol(t_mesh),new_fig=False)
    # plt.suptitle("Boundary vs Integrate Solutions")
    plt.show()

if __name__ == "__main__":
    main()
