import time

import GravNN
import matplotlib.pyplot as plt
import numpy as np

from FrozenOrbits.analysis import (
    check_for_intersection,
    print_OE_differences,
    print_state_differences,
)
from FrozenOrbits.bvp import *
from FrozenOrbits.constraints import *
from FrozenOrbits.gravity_models import pinnGravityModel
from FrozenOrbits.LPE import *
from FrozenOrbits.utils import propagate_orbit
from FrozenOrbits.visualization import *
from Scripts_Orbits.BVP.initial_conditions import *

np.random.seed(15)


def bvp_trad_OE(
    OE_0,
    X_0,
    T_0,
    planet,
    model,
    tol=1e-9,
    use_bounds=True,
    show=False,
    bounds=None,
):
    """Solve a BVP problem using the dynamics of the cartesian state vector"""
    lpe = LPE_Traditional(
        model.gravity_model,
        planet.mu,
        l_star=OE_0[0, 0],
        t_star=T_0,
        m_star=1.0,
        theta_star=2 * np.pi,
    )

    start_time = time.time()

    # Shooting solvers
    if bounds is None:
        bounds = (
            [0, 0.01, -np.inf, -np.inf, -np.inf, -np.inf, 0.5],
            [np.inf, 1.0, np.inf, np.inf, np.inf, np.inf, 2.0],
        )

    decision_variable_mask = [True, True, True, True, True, True, True]  # [OE, T] [N+1]
    constraint_variable_mask = [
        True,
        True,
        True,
        True,
        True,
        True,
        False,
    ]  # Don't use period as part of the constraint, just as part of jacobian
    constraint_angle_wrap = [
        False,
        False,
        False,
        True,
        True,
        True,
        False,
    ]  # Don't use period as part of the constraint, just as part of jacobian
    solver = ShootingLsSolver(
        lpe,
        decision_variable_mask,
        constraint_variable_mask,
        constraint_angle_wrap,
        rtol=tol,
        atol=tol,
        max_nfev=10,
    )  # Finds a local optimum, step size gets too small

    OE_0_sol, X_0_sol, T_0_sol, results = solver.solve(OE_0, T_0, bounds)
    print("OE")
    print(OE_0_sol[i] for i in range(len(OE_0_sol)))
    print("Cart")
    print(X_0_sol[i] for i in range(len(X_0_sol)))
    print("Time")
    print(T_0_sol)
    elapsed_time = time.time() - start_time

    # propagate the initial and solution orbits
    init_sol = propagate_orbit(T_0, X_0, model, tol=tol)
    bvp_sol = propagate_orbit(T_0_sol, X_0_sol, model, tol=tol)

    valid = check_for_intersection(bvp_sol, planet.obj_8k)

    dX_0 = print_state_differences(init_sol)
    dX_0_sol = print_state_differences(bvp_sol)

    angle_wrap = constraint_angle_wrap
    dOE_0, dOE_0_dimless = print_OE_differences(OE_0, lpe, "IVP", angle_wrap)
    dOE_sol, dOE_sol_dimless = print_OE_differences(OE_0_sol, lpe, "BVP", angle_wrap)

    bvp_sol_10 = propagate_orbit(T_0_sol * 10, X_0_sol, model, tol=tol)
    dX_0_sol_10 = print_state_differences(bvp_sol_10)

    if show:
        plot_cartesian_state_3d(init_sol.y.T, planet.obj_8k)
        plt.title("Initial Guess")

        plot_cartesian_state_3d(bvp_sol.y.T, planet.obj_8k)
        plt.title("BVP Solution")

        # plot_OE_1d(init_sol.t, OE_0, "traditional", y0_hline=True)
        # plot_OE_1d(bvp_sol.t, OE_0_sol.y, "traditional", y0_hline=True)
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
    model = pinnGravityModel(
        os.path.dirname(GravNN.__file__)
        + "/../Data/Dataframes/eros_poly_061523_64.data",
    )

    def sample_initial_conditions():
        planet = Eros()
        a = np.random.uniform(1.3 * planet.radius, 1.5 * planet.radius)
        e = np.random.uniform(0.1, 0.4)
        i = np.random.uniform(np.pi / 4 - 0.25, np.pi / 4 + 0.25)
        omega = np.random.uniform(0.0, 2 * np.pi)
        Omega = np.random.uniform(0.0, 2 * np.pi)
        M = np.random.uniform(0.0, 2 * np.pi)
        trad_OE = np.array([[a, e, i, omega, Omega, M]])
        X = trad2cart_tf(trad_OE, planet.mu).numpy()[0]
        T = 2 * np.pi * np.sqrt(trad_OE[0, 0] ** 3 / planet.mu)

        # Force custom IC for debugging
        # trad_OE = np.array(
        #     [[1.11e00 * a, 1.44e-02, np.pi / 4, 8.57e00, 2.98e00, -9.95e-01]],
        # )
        # X = trad2cart_tf(trad_OE, planet.mu).numpy()[0]
        # T = 1.16e00 * T

        return trad_OE, X, T, planet

    OE_0, X_0, T_0, planet = sample_initial_conditions()
    bounds = (
        [0.9, 0.01, np.pi / 4 - 0.25, -np.inf, -np.inf, -np.inf, 0.5],
        [1.1, 0.9, np.pi / 4 + 0.25, np.inf, np.inf, np.inf, np.inf],
    )

    # OE_0, X_0, T_0, planet = near_periodic_IC()
    data = bvp_trad_OE(
        OE_0,
        X_0,
        T_0,
        planet,
        model,
        tol=1e-7,
        show=True,
        bounds=bounds,
    )
