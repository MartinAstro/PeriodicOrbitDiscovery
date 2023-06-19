"""
Periodic Orbit Discovery Using PINN-GM Enabled Cartesian Shooting Method
=========================================================================

"""

import os
import time
from FrozenOrbits.analysis import check_for_intersection, print_OE_differences, print_state_differences
from FrozenOrbits.bvp import *

import GravNN
import matplotlib.pyplot as plt
import numpy as np
from FrozenOrbits.gravity_models import (pinnGravityModel)
from FrozenOrbits.LPE import *
from FrozenOrbits.utils import propagate_orbit
from FrozenOrbits.visualization import *
from FrozenOrbits.constraints import *
from Scripts.BVP.initial_conditions import *
np.random.seed(15)

#%% 
# Uncomment if debugging

# tf.config.run_functions_eagerly(True)

#%% 
# Load the trained PINN Gravity model from the dataframe in which it is stored
model = pinnGravityModel(os.path.dirname(GravNN.__file__) + \
    "/../Data/Dataframes/eros_BVP_PINN_III.data")  


#%% 
# Select and store an initial guess including cartesian state, 
# orbital element (OE), the keplerian orbit period, and the desired 
# celestial body

OE_0, X_0, T, planet = near_periodic_IC()

#%% 
# Initialize the equations of motion for the system.
# This includes choosing non-dimensionalization parameters for
# the cartesian state. 
# ..Warning:: 
#   Choice of non-dimensionalization can have considerable effects 
#   on the performance of the shooting method algorithm. Be careful 
#   to ensure that the position and velocity components of the state 
#   are of comparable magnitude!

scale = 1.0
lpe = LPE_Cartesian(model.gravity_model, planet.mu, 
                            l_star=np.linalg.norm(X_0[0:3])/scale, 
                            t_star=T, 
                            m_star=1.0)

#%% 
# Initialize the solver. This includes specifying the allowed bounds of the
# non-dimensionalized solution, as well as determining which variables should be used
# in the optimization (`decision_variable_mask`) and which should be included in the 
# minimization (`constraint_variable_mask`). 

bounds = ([-np.inf, -np.inf,-np.inf,-np.inf,-np.inf,-np.inf, 0.9],
            [ np.inf,  np.inf, np.inf, np.inf, np.inf, np.inf, 1.1])
decision_variable_mask = [True, True, True, True, True, True, True] # [OE, T] [N+1]
constraint_variable_mask = [True, True, True, True, True, True, False] #Don't use period as part of the constraint, just as part of jacobian
solver = CartesianShootingLsSolver(lpe, decision_variable_mask, constraint_variable_mask) 

#%% 
# Run the solver and store the solution initial conditions as well as the optimization Result object. 
start_time = time.time()
OE_0_sol, X_0_sol, T_sol, results = solver.solve(np.array([X_0]), T, bounds)

print(f"Time Elapsed: {time.time()-start_time}")
print(f"Initial OE: {OE_0} \t T: {T}")
print(f"BVP OE: {OE_0_sol} \t T {T_sol}")

#%% 
# Take the solution produced by the minimization and propagate it to determine how 
# close the solution's final state is to it's initial state. (Repeat for the initial guess
# for comparison). 

init_sol = propagate_orbit(T, X_0, model, tol=1E-7) 
bvp_sol = propagate_orbit(T_sol, X_0_sol, model, tol=1E-7) 

#%% 
# Make sure that the trajectory doesn't intersect with the body. 
check_for_intersection(bvp_sol, planet.obj_8k)

#%% 
# Print various metrics and plot results. 
print_state_differences(init_sol)
print_state_differences(bvp_sol)

plot_cartesian_state_3d(init_sol.y.T, planet.obj_8k)
plt.title("Initial Guess")

plot_cartesian_state_3d(bvp_sol.y.T, planet.obj_8k)
plt.title("BVP Solution")

OE_trad_init = cart2trad_tf(init_sol.y.T, planet.mu).numpy()
OE_trad_bvp = cart2trad_tf(bvp_sol.y.T, planet.mu).numpy()

print_OE_differences(OE_trad_init, lpe, "IVP", constraint_variable_mask)
print_OE_differences(OE_trad_bvp, lpe, "BVP", constraint_variable_mask)

plot_OE_1d(init_sol.t, OE_trad_init, 'traditional', y0_hline=True)
plot_OE_1d(bvp_sol.t, OE_trad_bvp, 'traditional', y0_hline=True)

plt.show()

