import matplotlib.pyplot as plt
import numpy as np
from GravNN.CelestialBodies.Asteroids import Eros

from FrozenOrbits.analysis import (
    check_for_intersection,
    print_OE_differences,
    print_state_differences,
)
from FrozenOrbits.bvp import *
from FrozenOrbits.constraints import *
from FrozenOrbits.LPE import *
from FrozenOrbits.utils import propagate_orbit
from FrozenOrbits.visualization import *
from Scripts_Orbits.BVP.initial_conditions import *

np.random.seed(15)


def bvp_mil_OE(OE_0, X_0, T_0, planet, model, tol=1e-9, show=False):
    """Solve a BVP problem using the dynamics of the cartesian state vector"""
    planet = Eros()
    # tf.config.run_functions_eagerly(True)

    OE_0 = oe2milankovitch_tf(OE_0, planet.mu).numpy()

    # If manual coordinate needed, rewrap OE
    # OE_0 = np.array(
    #     [[-1.49e04, -9.01e04, 9.56e04, 5.95e-02, -7.18e-02, -4.56e-02, 1.21e00]],
    # )
    # T = 73697.6367377903
    # X_0 = oe2cart_tf(OE_0, planet.mu, "milankovitch").numpy()
    # OE_0 = cart2oe_tf(X_0, planet.mu, "milankovitch").numpy()
    # X_0 = X_0[0]

    # Desire |H_0| = 1
    # Acknowledging H = (t_star/l_star**2)*H_tilde
    # Solve for l_star -> l_star = sqrt(t_star*H_tilde_mag/H_mag)

    H_mag_desired = 1.0
    lpe = LPE_Milankovitch(
        model.gravity_model,
        planet.mu,
        l_star=np.sqrt(
            T_0 * np.linalg.norm(OE_0[0, 0:3]) / H_mag_desired,
        ),  # H = t/l**2 * H_tilde
        t_star=T_0,
        m_star=1.0,
    )

    # normalized coordinates for semi-major axis
    # bounds = ([-2*1/np.sqrt(3), -2*1/np.sqrt(3), -2*1/np.sqrt(3), -0.3, -0.3, -0.3, -np.inf],
    #           [ 2*1/np.sqrt(3),  2*1/np.sqrt(3),  2*1/np.sqrt(3),  0.3,  0.3,  0.3,  np.inf])

    H_m = H_mag_desired
    bounds = (
        [-1.0 * H_m, -1.0 * H_m, -1.0 * H_m, -0.7, -0.7, -0.7, -np.inf, 0.9],
        [1.0 * H_m, 1.0 * H_m, 1.0 * H_m, 0.7, 0.7, 0.7, np.inf, 1.1],
    )

    bounds = (
        [-np.inf, -np.inf, -np.inf, -1.0, -1.0, -1.0, -np.inf, 0.5],
        [np.inf, np.inf, np.inf, 1.0, 1.0, 1.0, np.inf, 2.0],
    )

    # Shooting solvers
    start_time = time.time()
    decision_variable_mask = [
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
    ]  # [OE, T] [N+1]
    constraint_variable_mask = [
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        False,
    ]  # Don't use period as part of the constraint, just as part of jacobian
    constraint_angle_wrap_mask = [False, False, False, False, False, False, True, False]
    solver = ShootingLsSolver(
        lpe,
        decision_variable_mask,
        constraint_variable_mask,
        constraint_angle_wrap_mask,
        max_nfev=50,
        atol=tol,
        rtol=tol,
    )  # Finds a local optimum, step size gets too small

    OE_0_sol, X_0_sol, T_0_sol, results = solver.solve(OE_0, T_0, bounds)
    elapsed_time = time.time() - start_time

    # propagate the initial and solution orbits
    init_sol = propagate_orbit(T_0, X_0, model, tol=tol)
    bvp_sol = propagate_orbit(T_0_sol, X_0_sol, model, tol=tol)

    valid = check_for_intersection(bvp_sol, planet.obj_8k)

    dX_0 = print_state_differences(init_sol)
    dX_0_sol = print_state_differences(bvp_sol)

    angle_wrap = constraint_angle_wrap_mask
    dOE_0, dOE_0_dimless = print_OE_differences(OE_0, lpe, "IVP", angle_wrap)
    dOE_sol, dOE_sol_dimless = print_OE_differences(OE_0_sol, lpe, "BVP", angle_wrap)

    bvp_sol_10 = propagate_orbit(T_0_sol * 10, X_0_sol, model, tol=tol)
    dX_0_sol_10 = print_state_differences(bvp_sol_10)

    if show:
        plot_cartesian_state_3d(init_sol.y.T, planet.obj_8k)
        plt.title("Initial Guess")

        plot_cartesian_state_3d(bvp_sol.y.T, planet.obj_8k)
        plt.title("BVP Solution")

        plot_OE_1d(init_sol.t, OE_0, "milankovitch", y0_hline=True)
        plot_OE_1d(bvp_sol.t, OE_0_sol, "milankovitch", y0_hline=True)
        plt.show()

    print(f"Time Elapsed: {time.time()-start_time}")
    print(f"Initial OE: {OE_0} \t T: {T_0}")
    print(f"BVP OE: {OE_0_sol} \t T {T_0_sol}")

    data = {
        "T_0": [T_0],
        "T_0_sol": [T_0_sol],
        "OE_0": [OE_0[0]],
        "OE_0_sol": [OE_0_sol[0]],
        "X_0": [X_0],
        "X_0_sol": [X_0_sol],
        "dOE_0": [dOE_0],
        "dOE_sol": [dOE_sol],
        "dX_0": [dX_0],
        "dX_0_sol": [dX_0_sol],
        "dX_0_sol_10": [dX_0_sol_10],
        # "lpe": [lpe],
        "elapsed_time": [elapsed_time],
        "result": [results],
        "valid": [valid],
    }

    return data


if __name__ == "__main__":
    bvp_mil_OE()
