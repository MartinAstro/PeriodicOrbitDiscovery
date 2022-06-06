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
import OrbitalElements.orbitalPlotting as op
from FrozenOrbits.utils import (
    check_solution_validity,
    compute_period,
    get_energy,
    get_initial_orbit_guess,
    get_S_matrix,
    get_solution_metrics,
    sample_safe_trad_OE,
    Solution
)
from FrozenOrbits.visualization import plot_1d_solutions, plot_3d_solutions
from GravNN.CelestialBodies.Asteroids import Eros
from scipy.integrate import solve_bvp

from OrbitalElements.orbitalPlotting import plot1d


def main():
    """
    Visualization of how discrepant the potential and accelerations are 
    between the polyhedral model and the PINN model. 
    """
    planet = Eros()
    init_nodes = 100

    # min_radius = planet.radius * 3
    # max_radius = planet.radius * 4
    min_radius = planet.radius * 5
    max_radius = planet.radius * 7
    np.random.seed(15)

    # Load in the gravity model
    pinn_model = pinnGravityModel(
        # os.path.dirname(GravNN.__file__) + "/../Data/Dataframes/eros_8k_100000.data"
        # os.path.dirname(GravNN.__file__) + "/../Data/Dataframes/eros_pinn_III.data"
        os.path.dirname(GravNN.__file__) + "/../Data/Dataframes/eros_pinn_III_030822.data"
    )  
    poly_model = polyhedralGravityModel(planet, planet.obj_8k)

    lpe_pinn = LPE(pinn_model, planet.mu, element_set="traditional")
    lpe_poly = LPE(poly_model, planet.mu, element_set="traditional")

    # Set the initial conditions and dynamics via OE
    trad_OE = sample_safe_trad_OE(min_radius, max_radius)
    T = compute_period(planet.mu, trad_OE[0, 0])
    init_state = np.hstack(oe2cart_tf(trad_OE, planet.mu))
    t_mesh, step = np.linspace(0, T, init_nodes, retstep=True)
    
    init_state = np.array([
        4.61513747e+04, 
        8.12741755e+04, 
        -1.00860719e+04, 
        8.49819800e-01,
        -1.49764060e+00,
        2.47435298e+00
        ])


    pinn_sol = solve_ivp_pos_problem(T, init_state, lpe_pinn, t_eval=t_mesh)
    poly_sol = solve_ivp_pos_problem(T, init_state, lpe_poly, t_eval=t_mesh)
   
    # Plot trajectories
    plot_1d_solutions(t_mesh, pinn_sol, new_fig=True)
    plot_1d_solutions(t_mesh, poly_sol, new_fig=False)
    plt.suptitle("PINN vs Polyhedral Solution")

    # Plot difference
    diff_sol = Solution(pinn_sol.y - poly_sol.y, t_mesh)
    plot_1d_solutions(t_mesh, diff_sol, new_fig=True)
    plt.suptitle("Trajectory Differences")

    poly_acc = lpe_poly.model.generate_acceleration(poly_sol.y[0:3,:])
    pinn_acc = lpe_pinn.model.generate_acceleration(poly_sol.y[0:3,:])

    positions = poly_sol.y[0:3,:]
    radii = np.linalg.norm(positions, axis=0)
    diff_acc = poly_acc - pinn_acc
    diff_acc_mag = np.linalg.norm(diff_acc,axis=1) 
    diff_acc_mag_percent = np.clip(diff_acc_mag / np.linalg.norm(poly_acc, axis=1)*100, 0, 100)

    scale = np.max(diff_acc_mag_percent) - np.min(diff_acc_mag_percent)
    colors = plt.cm.RdYlGn(1 - ((diff_acc_mag_percent  - np.min(diff_acc_mag_percent)) / scale))   
    op.plot3d(pinn_sol.y[0:3], cVec=colors, obj_file=Eros().obj_8k, plot_type='scatter')


    plt.figure()
    plt.scatter(radii, diff_acc_mag)
    plt.title("Acceleration Difference Magnitude")


    poly_pot = lpe_poly.model.generate_potential(poly_sol.y[0:3,:])
    pinn_pot = lpe_pinn.model.generate_potential(poly_sol.y[0:3,:])
    diff_pot = poly_pot - pinn_pot

    plt.figure()
    plt.scatter(radii, poly_pot, label='Poly')
    plt.scatter(radii, pinn_pot, label='PINN')
    plt.legend()
    plt.title("Potentials")

    plt.figure()
    plt.scatter(radii, diff_pot)
    plt.title("Potential Differences")

    plt.show()





if __name__ == "__main__":
    main()
