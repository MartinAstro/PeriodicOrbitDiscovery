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

# tf.config.run_functions_eagerly(True)
np.random.seed(15)


def bvp_cart_OE(OE_0, X_0, T_0, planet, model, tol=1e-9, show=False):
    """Solve a BVP problem using the dynamics of the cartesian state vector"""

    l_star = np.linalg.norm(X_0[0:3])
    t_star = l_star / np.linalg.norm(X_0[3:6])

    lpe = LPE_Cartesian(
        model.gravity_model,
        planet.mu,
        l_star=l_star,
        t_star=t_star,
        m_star=1.0,
    )
    start_time = time.time()

    # Shooting solvers
    bounds = (
        [-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, 0.9 * T_0 / t_star],
        [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, 1.1 * T_0 / t_star],
    )

    bounds = (
        [-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, 0.5 * T_0 / t_star],
        [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, 2.0 * T_0 / t_star],
    )
    # [OE, T] [N+1]
    decision_variable_mask = [True, True, True, True, True, True, True]
    constraint_variable_mask = [True, True, True, True, True, True, False]
    constraint_angle_wrap = [False, False, False, False, False, False, False]

    solver = CartesianShootingLsSolver(
        lpe,
        decision_variable_mask,
        constraint_variable_mask,
        constraint_angle_wrap,
        max_nfev=50,
        atol=tol,
        rtol=tol,
    )

    OE_0_sol, X_0_sol, T_0_sol, results = solver.solve(np.array([X_0]), T_0, bounds)
    elapsed_time = time.time() - start_time

    # propagate the initial and solution orbits
    init_sol = propagate_orbit(T_0, X_0, model, tol=tol)
    bvp_sol = propagate_orbit(T_0_sol, X_0_sol, model, tol=tol)

    valid = check_for_intersection(bvp_sol, planet.obj_8k)

    dX_0 = print_state_differences(init_sol)
    dX_0_sol = print_state_differences(bvp_sol)

    # convert solution to traditional orbital elements
    OE_trad_init = cart2trad_tf(init_sol.y.T, planet.mu).numpy()
    OE_trad_bvp = cart2trad_tf(bvp_sol.y.T, planet.mu).numpy()

    angle_wrap = constraint_angle_wrap
    dOE_0, dOE_0_dimless = print_OE_differences(OE_trad_init, lpe, "IVP", angle_wrap)
    dOE_sol, dOE_sol_dimless = print_OE_differences(OE_trad_bvp, lpe, "BVP", angle_wrap)

    bvp_sol_10 = propagate_orbit(T_0_sol * 10, X_0_sol, model, tol=tol)
    dX_0_sol_10 = print_state_differences(bvp_sol_10)

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

    if show:
        plot_cartesian_state_3d(init_sol.y.T, planet.obj_8k)
        plt.title("Initial Guess")

        plot_cartesian_state_3d(bvp_sol.y.T, planet.obj_8k)
        plt.title("BVP Solution")

        plot_OE_1d(init_sol.t, OE_trad_init, "traditional", y0_hline=True)
        plot_OE_1d(bvp_sol.t, OE_trad_bvp, "traditional", y0_hline=True)
        plt.show()

    return data


if __name__ == "__main__":
    model = pinnGravityModel(
        os.path.dirname(GravNN.__file__)
        + "/../Data/Dataframes/eros_poly_061523_64.data",
    )
    OE_0, X_0, T_0, planet = near_periodic_IC()
    data = bvp_cart_OE(OE_0, X_0, T_0, planet, model, tol=1e-7, show=False)
    print(data)

    from pprint import pprint

    pprint(data)
