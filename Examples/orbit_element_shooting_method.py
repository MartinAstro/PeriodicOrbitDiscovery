import os
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

"""Solve a BVP problem using the dynamics of the cartesian state vector"""
np.random.seed(15)
# tf.config.run_functions_eagerly(True)

model = pinnGravityModel(
    os.path.dirname(GravNN.__file__) + "/../Data/Dataframes/eros_BVP_PINN_III.data"
)

OE_0, X_0, T, planet = near_periodic_IC()

lpe = LPE_Traditional(
    model.gravity_model,
    planet.mu,
    l_star=OE_0[0, 0],
    t_star=T,
    m_star=1.0,
    theta_star=2 * np.pi,
)

start_time = time.time()


# Shooting solvers
bounds = (
    [0.7, 0.1, -np.pi, -2 * np.pi, -2 * np.pi, -2 * np.pi, 0.9],
    [1.0, 0.5, np.pi, 2 * np.pi, 2 * np.pi, 2 * np.pi, 2.0],
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
    rtol=1e-6,
    atol=1e-6,
)  # Finds a local optimum, step size gets too small


OE_0_sol, X_0_sol, T_sol, results = solver.solve(OE_0, T, bounds)

print(f"Time Elapsed: {time.time()-start_time}")
print(f"Initial OE: {OE_0} \t T: {T}")
print(f"BVP OE: {OE_0_sol} \t T {T_sol}")

# propagate the initial and solution orbits
init_sol = propagate_orbit(T, X_0, model, tol=1e-7)
bvp_sol = propagate_orbit(T_sol, X_0_sol, model, tol=1e-7)

check_for_intersection(bvp_sol, planet.obj_8k)

print_state_differences(init_sol)
print_state_differences(bvp_sol)

plot_cartesian_state_3d(init_sol.y.T, planet.obj_8k)
plt.title("Initial Guess")

plot_cartesian_state_3d(bvp_sol.y.T, planet.obj_8k)
plt.title("BVP Solution")

OE_trad_init = cart2trad_tf(init_sol.y.T, planet.mu).numpy()
OE_trad_bvp = cart2trad_tf(bvp_sol.y.T, planet.mu).numpy()

print_OE_differences(OE_trad_init, lpe, "IVP", constraint_angle_wrap)
print_OE_differences(OE_trad_bvp, lpe, "BVP", constraint_angle_wrap)

plot_OE_1d(init_sol.t, OE_trad_init, "traditional", y0_hline=True)
plot_OE_1d(bvp_sol.t, OE_trad_bvp, "traditional", y0_hline=True)

plt.show()
