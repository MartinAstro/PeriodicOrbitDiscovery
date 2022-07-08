
from unittest import TestResult
from FrozenOrbits.boundary_conditions import *
from FrozenOrbits.coordinate_transforms import *
from FrozenOrbits.constraints import *
from FrozenOrbits.LPE import LPE

import numpy as np
import pandas as pd
import copy
import time
from FrozenOrbits.dynamics import *
from FrozenOrbits.utils import calc_angle_diff

from GravNN.Support.transformations import cart2sph, invert_projection
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares, root, minimize, basinhopping
from GravNN.Support.ProgressBar import ProgressBar
from abc import ABC,  abstractmethod
np.set_printoptions(formatter={'float': "{0:0.2e}".format})

######################
## Helper Functions ##
######################

def debug_plot(sol, lpe, element_set):
    from GravNN.CelestialBodies.Asteroids import Eros
    import matplotlib.pyplot as plt
    from FrozenOrbits.visualization import plot_cartesian_state_3d, plot_OE_1d
    
    plot_OE_1d(sol.t, sol.y.T, element_set)

    OE_dim = lpe.dimensionalize_state(sol.y.T).numpy()
    cart = oe2cart_tf(OE_dim, Eros().mu, element_set).numpy()

    sol_cart = copy.deepcopy(sol)
    sol_cart.y = cart.T
    plot_cartesian_state_3d(sol_cart.t, sol_cart.y.T, Eros().obj_8k)
    plt.show()
    return

def evolve_state_w_STM(t, x_0, F, model, tol=1E-6):
    N = len(x_0)
    phi_0 = np.identity(N)
    z_i = np.hstack((x_0, phi_0.reshape((-1))))
    pbar = ProgressBar(t, enable=True)
    sol = solve_ivp(F, [0, t], z_i, args=(model,pbar),atol=tol, rtol=tol)
    pbar.close()
    if not sol.success:
        print(f"{sol.success} \t {sol.message}")
    x_f = sol.y[:N,-1]
    phi_f = sol.y[N:,-1].reshape((N,N))
    return x_f, phi_f

def print_result_info(result):
    print(f"""
    Success? ({result.success} \t Status: {result.status} 
    Message: {result.message}
    OE_0 = {result.x} 
    Number of Function Evals: {result.nfev} \t Number of Jacobian Evals: {result.njev}")
    """)

def update_state(X_0, X_subset, decision_variable_mask):
    k = 0
    X_updated = copy.deepcopy(X_0)
    for i in range(len(X_updated)):
        # Only update allowed decision variables
        if decision_variable_mask[i]:
            X_updated[i] = X_subset[k]
            k += 1

    X_updated = np.array([X_updated]) # the non-dim OE
    return X_updated

def non_dimensionalize(OE_dim, T_dim, lpe):
    OE = lpe.non_dimensionalize_state(OE_dim).numpy()
    T = lpe.non_dimensionalize_time(T_dim).numpy()
    return OE, T

def dimensionalize(OE, T, lpe):
    OE_dim = lpe.dimensionalize_state(OE).numpy()
    T_dim = lpe.dimensionalize_time(T).numpy()
    return OE_dim, T_dim

def populate_masks(OE, decision_variable_mask, constraint_variable_mask, constraint_angle_wrap_mask):
    if decision_variable_mask is None:
        decision_variable_mask = [True]*(len(OE[0])+1) # N + 1
    if constraint_variable_mask is None:
        constraint_variable_mask = [True]*len(OE[0]+1) # N
    if constraint_angle_wrap_mask is None:
        constraint_angle_wrap_mask = [False]*len(OE[0]+1) # N

#######################
## Custom algorithms ##
#######################

def variable_time_mirror_bvp(T, x0, model):
    T_i_p1 = copy.deepcopy(T) / 2
    x_i_p1 = copy.deepcopy(x0)

    k = 0
    tol = 1
    while tol > 1E-6 and k < 10: 
        start_time = time.time()
        T_i = copy.deepcopy(T_i_p1)
        x_i = copy.deepcopy(x_i_p1)
        x_f, phi_f = evolve_state_w_STM(T_i, x_i, dynamics_cart_w_STM, model, tol=1E-6)

        x_dot_f = np.hstack((x_f[3:6], model.generate_acceleration(x_f[0:3])))

        V_i = np.array([x_i[0], x_i[4], T_i])
        C = np.array([
            [x_f[1]],
            [x_f[3]]
            ])
        D = np.array([
            [phi_f[1,0], phi_f[1,4], x_dot_f[1]],
            [phi_f[3,0], phi_f[3,4], x_dot_f[3]]
            ])
        
        V_i_p1 = V_i - np.transpose(D.T@np.linalg.pinv(D@D.T)@C).squeeze()
        x_i_p1[0] = V_i_p1[0]
        x_i_p1[4] = V_i_p1[1]
        T_i_p1 = V_i_p1[2]
        tol = np.linalg.norm(C)
        dx = np.linalg.norm((x_i_p1 - x_i)[0:6])
        print(f"Iteration {k}: tol = {tol} \t dx_k = {dx} \t dT = {T_i_p1 - T_i} \t Time Elapsed: {time.time() - start_time}")
        k += 1

    return x_i_p1, T_i_p1*2

def general_variable_time_cartesian_bvp(T, x0, model):
    T_i_p1 = copy.deepcopy(T) 
    x_i_p1 = copy.deepcopy(x0)

    k = 0
    tol = 1
    while tol > 1E-6 and k < 10: 
        start_time = time.time()
        T_i = copy.deepcopy(T_i_p1)
        x_i = copy.deepcopy(x_i_p1)
        N = len(x_i)

        # Propagate dynamics 
        x_f, phi_f = evolve_state_w_STM(T_i, x_i, dynamics_cart_w_STM, model, tol=1E-6)

        # Get X_dot @ t_f
        r_f = x_f[0:3]
        v_f = x_f[3:6]
        a_f = model.generate_acceleration(r_f)
        x_dot_f = np.hstack((v_f, a_f)).reshape((6,1))

        # Build BVP "state" vector
        V_i = np.hstack((x_i, T_i))

        # Define constraint vector and corresponding partials w.r.t. BVP state vector (dC/dV)
        C = x_f - x_i
        D = np.block([phi_f - np.eye(len(x0)), x_dot_f])

        # Compute correction term and apply 
        dV = np.transpose(D.T@np.linalg.pinv(D@D.T)@C).squeeze()
        V_i_p1 = V_i - dV

        # Map back to state
        x_i_p1 = V_i_p1[0:len(x0)]
        T_i_p1 = V_i_p1[len(x0)]
        
        tol = np.linalg.norm(C)
        dx = np.linalg.norm((x_i_p1 - x_i)[0:len(x0)])
        print(f"Iteration {k}: tol = {tol} \t dx_k = {dx} \t dT = {T_i_p1 - T_i} \t Time Elapsed: {time.time() - start_time}")
        k += 1

    return x_i_p1, T_i_p1

def general_variable_time_bvp_trad_OE(T_dim, OE_0_dim, model, element_set, decision_variable_mask=None, constraint_variable_mask=None, constraint_angle_wrap_mask=None):

    OE_0 = model.non_dimensionalize_state(OE_0_dim).numpy()
    T = model.non_dimensionalize_time(T_dim).numpy()
    print(f"Total Time {T} \n Dim State {OE_0_dim} \n Non Dim State {OE_0}")
    T_i_p1 = copy.deepcopy(T) 
    OE_i_p1 = copy.deepcopy(OE_0.reshape((-1,)))

    k = 0
    tol = 1
    while tol > 1E-6 and k < 10: 
        T_i = copy.deepcopy(T_i_p1) 
        x_i = copy.deepcopy(OE_i_p1)
        x_f, phi_f = evolve_state_w_STM(T_i, x_i, dynamics_OE_w_STM, model, tol=1E-6)
        OE_i_p1, T_i_p1 = OE_constraint(x_f, phi_f, x_i, T_i, model, k, 
                            decision_variable_mask=decision_variable_mask,
                            constraint_variable_mask=constraint_variable_mask,
                            constraint_angle_wrap_mask=constraint_angle_wrap_mask)
        k += 1

    OE_i_p1 = model.dimensionalize_state(np.array([OE_i_p1])).numpy()
    T_i_p1 = model.dimensionalize_time(T_i_p1).numpy()
    x_i_p1 = oe2cart_tf(OE_i_p1, model.mu_tilde, element_set).numpy()[0]

    return OE_i_p1, x_i_p1, T_i_p1

########################
## Vector Constraints ##
########################

def constraint_shooting(V_0, lpe, x_0, 
    decision_variable_mask, 
    constraint_variable_mask, 
    constraint_angle_wrap_mask):
    
    x_i = update_state(x_0[:-1], V_0, decision_variable_mask).reshape((-1,))
    x_i = x_i.reshape((-1,))
    if decision_variable_mask[-1]:
        T = V_0[-1]
    else:
        T = x_0[-1]

    # Propagate the updated state
    pbar = ProgressBar(T, enable=False)
    sol = solve_ivp(dynamics_OE, 
            [0, T],
            x_i.reshape((-1,)), 
            args=(lpe, pbar),
            atol=1E-7, rtol=1E-7)
    pbar.close()
    x_f = sol.y[:,-1]

    # Calculate the constraint vector
    C = x_f - x_i 
    C = np.hstack((C, [0.0])) # add the period to the constraint ()

    # Wrap angles depending on the element set
    for i in range(len(x_f)):
        if constraint_angle_wrap_mask[i]:
            C[i] = calc_angle_diff(x_0[i], x_f[i])

    C = C[constraint_variable_mask] # remove masked variables
    return C

def constraint_shooting_jac(V_0, lpe, x_0, 
    decision_variable_mask, 
    constraint_variable_mask,
    constraint_angle_wrap_mask):
    N = len(x_0) - 1 # Remove from the state
    phi_0 = np.identity(N)

    # Only update allowed decision variables
    x_i = update_state(x_0[:-1], V_0, decision_variable_mask)
    x_i = x_i.reshape((-1,))

    if decision_variable_mask[-1]:
        T = V_0[-1]
    else:
        T = x_0[-1]

    # Propagate the corrected state
    z_i = np.hstack((
        x_i.reshape((-1,)),
        phi_0.reshape((-1)))
        )

    pbar = ProgressBar(T, enable=True)
    sol = solve_ivp(dynamics_OE_w_STM, 
                    [0, T],
                    z_i.reshape((-1,)),
                    args=(lpe,pbar),
                    atol=1E-7, rtol=1E-7)
    pbar.close()
    z_f = sol.y[:,-1]
    x_f = z_f[:N]
    phi_ti_t0 = np.reshape(z_f[N:], (N,N))
    x_dot_f = lpe.dOE_dt(x_f)

    # Evaluate the general jacobian
    D = np.hstack([phi_ti_t0 - np.eye(N), x_dot_f.reshape((N,-1))])

    # Append a time variable row
    D = np.vstack((D, np.zeros((N+1))))

    # Remove specified decision variables from jacobian
    D = D[:, decision_variable_mask] # remove columns in D

    # # Remove constraint variables
    D = D[constraint_variable_mask, :] # remove rows in D

    return D

def constraint_instantaneous(V_0, lpe, x_0, 
    decision_variable_mask, 
    constraint_variable_mask, 
    constraint_angle_wrap_mask):
    
    x_i = update_state(x_0, V_0, decision_variable_mask)
    C = lpe.dOE_dt(x_i)
    C = C[constraint_variable_mask] # remove masked variables

    return C

def constraint_instantaneous_jac(V_0, lpe, x_0, 
    decision_variable_mask, 
    constraint_variable_mask,
    constraint_angle_wrap_mask):

    # Only update allowed decision variables
    x_i = update_state(x_0, V_0, decision_variable_mask)

    # Evaluate the general jacobian
    D = lpe.dOE_dt_dx(x_i)

    # Remove specified decision variables from jacobian
    D = D[:, decision_variable_mask] # remove columns in D
    
    D = D[constraint_variable_mask, :] # remove rows in D

    return D

########################
## Scalar Constraints ##
########################

def constraint_shooting_scalar(V_0, lpe, x_0, 
    decision_variable_mask, 
    constraint_variable_mask, 
    constraint_angle_wrap_mask):
    
    C = constraint_shooting(V_0, lpe, x_0, 
    decision_variable_mask, 
    constraint_variable_mask, 
    constraint_angle_wrap_mask)
    return np.linalg.norm(C)**2

def constraint_shooting_jac_scalar(V_0, lpe, x_0, 
    decision_variable_mask, 
    constraint_variable_mask,
    constraint_angle_wrap_mask):

    C = constraint_shooting(V_0, lpe, x_0, 
    decision_variable_mask, 
    constraint_variable_mask, 
    constraint_angle_wrap_mask)

    D = constraint_shooting_jac(V_0, lpe, x_0, 
    decision_variable_mask, 
    constraint_variable_mask, 
    constraint_angle_wrap_mask)

    D_scalar = np.zeros((len(D),))
    for i in range(len(D)):
        D_scalar += 2*C[i]*D[i,:]
    
    return D

def constraint_instantaneous_scalar(V_0, lpe, x_0, 
    decision_variable_mask, 
    constraint_variable_mask, 
    constraint_angle_wrap_mask):
    
    C = constraint_instantaneous(V_0, lpe, x_0, 
    decision_variable_mask, 
    constraint_variable_mask, 
    constraint_angle_wrap_mask)

    return np.linalg.norm(C)**2

def constraint_instantaneous_jac_scalar(V_0, lpe, x_0, 
    decision_variable_mask, 
    constraint_variable_mask,
    constraint_angle_wrap_mask):

    D = constraint_instantaneous_jac(V_0, lpe, x_0, 
    decision_variable_mask, 
    constraint_variable_mask, 
    constraint_angle_wrap_mask)

    D_scalar = np.zeros((len(D),))
    for i in range(len(D)):
        D_scalar += D[i,:]
    
    return D



##############################
## Solver Interface Classes ##
##############################

class InstantaneousSolver(ABC):
    def __init__(self, lpe, decision_variable_mask=None, constraint_variable_mask=None, constraint_angle_wrap_mask=None):
        self.lpe = lpe
        self.element_set = lpe.element_set

        if decision_variable_mask is None:
            decision_variable_mask = [True]*(lpe.num_elements+1) # N + 1
        self.decision_variable_mask = decision_variable_mask
        if constraint_variable_mask is None:
            constraint_variable_mask = [True]*(lpe.num_elements) # N + 1
        self.constraint_variable_mask = constraint_variable_mask
        if constraint_angle_wrap_mask is None:
            constraint_angle_wrap_mask = [True]*(lpe.num_elements+1) # N + 1
        self.constraint_angle_wrap_mask = constraint_angle_wrap_mask

        pass
    
    def initialize_solver_args(self, OE_0_dim, T_dim, solution_bounds):
        OE_0, T = non_dimensionalize(OE_0_dim, T_dim, self.lpe)
        X_0 = OE_0.reshape((-1)) # Decision variables that can be updated
        V_0 = X_0[self.decision_variable_mask]
        V_solution_bounds = np.array(solution_bounds)[:,self.decision_variable_mask]
        return X_0, V_0, V_solution_bounds, T
    
    @abstractmethod
    def solve_subroutine(self, X_0, V_0, V_solution_bounds):
        pass
    
    def prepare_outputs(self, X_0, T, result):
        print_result_info(result)

        OE_f = update_state(X_0, result.x, self.decision_variable_mask)
        T_f = T

        OE_0_sol, T_sol = dimensionalize(OE_f, T_f, self.lpe)
        X_0_sol = oe2cart_tf(OE_0_sol, self.lpe.mu_tilde, self.element_set).numpy()[0]
        return OE_0_sol, X_0_sol, T_sol, result

    def solve(self, OE_0_dim, T_dim, solution_bounds):
        X_0, V_0, V_solution_bounds, T = self.initialize_solver_args(OE_0_dim, T_dim, solution_bounds)
        result = self.solve_subroutine(X_0, V_0, V_solution_bounds)
        return self.prepare_outputs(X_0, T, result)

class ShootingSolver(ABC):
    def __init__(self, lpe, decision_variable_mask=None, constraint_variable_mask=None, constraint_angle_wrap_mask=None):
        self.lpe = lpe
        self.element_set = lpe.element_set

        if decision_variable_mask is None:
            decision_variable_mask = [True]*(lpe.num_elements+1) # N + 1
        self.decision_variable_mask = decision_variable_mask
        if constraint_variable_mask is None:
            constraint_variable_mask = [True]*(lpe.num_elements+1) # N + 1
        self.constraint_variable_mask = constraint_variable_mask
        if constraint_angle_wrap_mask is None:
            constraint_angle_wrap_mask = [True]*(lpe.num_elements+1) # N + 1
        self.constraint_angle_wrap_mask = constraint_angle_wrap_mask

        pass
    
    def initialize_solver_args(self, OE_0_dim, T_dim, solution_bounds):
        OE_0, T = non_dimensionalize(OE_0_dim, T_dim, self.lpe)
        X_0 = np.hstack((OE_0.reshape((-1)), T)) # Decision variables that can be updated
        V_0 = X_0[self.decision_variable_mask]
        V_solution_bounds = np.array(solution_bounds)[:,self.decision_variable_mask]

        return X_0, V_0, V_solution_bounds, T
    
    @abstractmethod
    def solve_subroutine(self, X_0, V_0, V_solution_bounds):
        pass
    
    def prepare_outputs(self, X_0, T, result):
        print_result_info(result)

        OE_f = update_state(X_0, result.x, self.decision_variable_mask) #remove time
        T_f = OE_f[0,-1]# The non-dim time
        OE_f = np.array([OE_f[0,:-1]]) # the non-dim OE

        OE_0_sol, T_sol = dimensionalize(OE_f, T_f, self.lpe)
        X_0_sol = oe2cart_tf(OE_0_sol, self.lpe.mu_tilde, self.element_set).numpy()[0]
        return OE_0_sol, X_0_sol, T_sol, result

    def solve(self, OE_0_dim, T_dim, solution_bounds):
        X_0, V_0, V_solution_bounds, T = self.initialize_solver_args(OE_0_dim, T_dim, solution_bounds)
        result = self.solve_subroutine(X_0, V_0, V_solution_bounds)
        return self.prepare_outputs(X_0, T, result)

class InstantaneousLsSolver(InstantaneousSolver):
    def __init__(self, *args):
        super().__init__(*args)
        pass

    def initialize_solver_args(self, OE_0_dim, T_dim, solution_bounds):
        return super().initialize_solver_args(OE_0_dim, T_dim, solution_bounds)

    def prepare_outputs(self, X_0, T, result):
        return super().prepare_outputs(X_0, T, result)

    def solve_subroutine(self, X_0, V_0, V_solution_bounds):
        result = least_squares(constraint_instantaneous, V_0, constraint_instantaneous_jac, 
                                    args=(
                                        self.lpe, 
                                        X_0,
                                        self.decision_variable_mask,
                                        self.constraint_variable_mask,
                                        self.constraint_angle_wrap_mask),
                                    bounds=V_solution_bounds,
                                    verbose=2,
                                    xtol=None,
                                    ftol=None,
                                    # method='dogbox'
                                    )
        return result

class InstantaneousRootSolver(InstantaneousSolver):
    def __init__(self, *args):
        super().__init__(*args)

    def initialize_solver_args(self, OE_0_dim, T_dim, solution_bounds):
        return super().initialize_solver_args(OE_0_dim, T_dim, solution_bounds)

    def prepare_outputs(self, X_0, T, result):
        return super().prepare_outputs(X_0, T, result)

    def solve_subroutine(self, X_0, V_0, V_solution_bounds):
        V_bounds_tuple = []
        for i in range(len(V_solution_bounds[0])):
            V_bounds_tuple.append(tuple(V_solution_bounds[:,i]))

        result = root(constraint_instantaneous, V_0, jac=constraint_instantaneous_jac, 
                            args=(
                                self.lpe, 
                                X_0,
                                self.decision_variable_mask,
                                self.constraint_variable_mask,
                                self.constraint_angle_wrap_mask),
                            #bounds=V_bounds_tuple,
                            )
        return result

class InstantaneousMinimizeSolver(InstantaneousSolver):
    def __init__(self, *args):
        super().__init__(*args)

    def initialize_solver_args(self, OE_0_dim, T_dim, solution_bounds):
        return super().initialize_solver_args(OE_0_dim, T_dim, solution_bounds)

    def prepare_outputs(self, X_0, T, result):
        return super().prepare_outputs(X_0, T, result)

    def solve_subroutine(self, X_0, V_0, V_solution_bounds):
        V_bounds_tuple = []
        for i in range(len(V_solution_bounds[0])):
            V_bounds_tuple.append(tuple(V_solution_bounds[:,i]))

        result = minimize(constraint_instantaneous_scalar, V_0, 
                            args=(
                                self.lpe, 
                                X_0,
                                self.decision_variable_mask,
                                self.constraint_variable_mask,
                                self.constraint_angle_wrap_mask),
                            tol=1E-48,
                            bounds=V_bounds_tuple,
                            )
        return result

class InstantaneousBasinHoppingSolver(InstantaneousSolver):
    def __init__(self, *args):
        super().__init__(*args)

    def initialize_solver_args(self, OE_0_dim, T_dim, solution_bounds):
        return super().initialize_solver_args(OE_0_dim, T_dim, solution_bounds)

    def prepare_outputs(self, X_0, T, result):
        return super().prepare_outputs(X_0, T, result)

    def solve_subroutine(self, X_0, V_0, V_solution_bounds):
        V_bounds_tuple = []
        for i in range(len(V_solution_bounds[0])):
            V_bounds_tuple.append(tuple(V_solution_bounds[:,i]))

        result = basinhopping(constraint_instantaneous_scalar, V_0, 
                            minimizer_kwargs={'args' : (
                                self.lpe, 
                                X_0,
                                self.decision_variable_mask,
                                self.constraint_variable_mask,
                                self.constraint_angle_wrap_mask)},
                            disp=True
                            )
        return result

class ShootingLsSolver(ShootingSolver):
    def __init__(self, *args):
        super().__init__(*args)    
    def solve_subroutine(self, X_0, V_0, V_solution_bounds):
        result = least_squares(constraint_shooting, V_0, jac=constraint_shooting_jac, 
                                    args=(
                                        self.lpe, 
                                        X_0,
                                        self.decision_variable_mask,
                                        self.constraint_variable_mask,
                                        self.constraint_angle_wrap_mask),
                                    bounds=V_solution_bounds,
                                    verbose=2,
                                    # xtol=1
                                    # xtol=None,
                                    # ftol=None,
                                    # method='dogbox'
                                    )
        return result

class ShootingRootSolver(ShootingSolver):
    def __init__(self, *args):
        super().__init__(*args)
    
    def solve_subroutine(self, X_0, V_0, V_solution_bounds):
        V_bounds_tuple = []
        for i in range(len(V_solution_bounds[0])):
            V_bounds_tuple.append(tuple(V_solution_bounds[:,i]))

        result = root(constraint_shooting, V_0, jac=constraint_shooting_jac, 
                            args=(
                                self.lpe, 
                                X_0,
                                self.decision_variable_mask,
                                self.constraint_variable_mask,
                                self.constraint_angle_wrap_mask),
                            # bounds=V_bounds_tuple,
                            )
        return result

class ShootingMinimizeSolver(ShootingSolver):
    def __init__(self, *args):
        super().__init__(*args)
    
    def solve_subroutine(self, X_0, V_0, V_solution_bounds):
        V_bounds_tuple = []
        for i in range(len(V_solution_bounds[0])):
            V_bounds_tuple.append(tuple(V_solution_bounds[:,i]))

        result = minimize(constraint_shooting_scalar, V_0, jac=constraint_shooting_jac_scalar,
                            args=(
                                self.lpe, 
                                X_0,
                                self.decision_variable_mask,
                                self.constraint_variable_mask,
                                self.constraint_angle_wrap_mask),
                            bounds=V_bounds_tuple,
                            )
        return result
    


#######################
## Cartesian Solvers ##
#######################
class CartesianShootingSolver(ABC):
    def __init__(self, lpe, decision_variable_mask=None, constraint_variable_mask=None, constraint_angle_wrap_mask=None):
        self.lpe = lpe
        self.element_set = lpe.element_set

        if decision_variable_mask is None:
            decision_variable_mask = [True]*(lpe.num_elements+1) # N + 1
        self.decision_variable_mask = decision_variable_mask
        if constraint_variable_mask is None:
            constraint_variable_mask = [True]*(lpe.num_elements+1) # N + 1
        self.constraint_variable_mask = constraint_variable_mask
        if constraint_angle_wrap_mask is None:
            constraint_angle_wrap_mask = [True]*(lpe.num_elements+1) # N + 1
        self.constraint_angle_wrap_mask = constraint_angle_wrap_mask

        pass
    
    def initialize_solver_args(self, OE_0_dim, T_dim, solution_bounds):
        OE_0, T = non_dimensionalize(OE_0_dim, T_dim, self.lpe)
        X_0 = np.hstack((OE_0.reshape((-1)), T)) # Decision variables that can be updated
        V_0 = X_0[self.decision_variable_mask]
        V_solution_bounds = np.array(solution_bounds)[:,self.decision_variable_mask]

        return X_0, V_0, V_solution_bounds, T
    
    def prepare_outputs(self, X_0, T, result):
        print_result_info(result)

        OE_f = update_state(X_0, result.x, self.decision_variable_mask) #remove time
        T_f = OE_f[0,-1]# The non-dim time
        OE_f = np.array([OE_f[0,:-1]]) # the non-dim OE

        # The OE set is actually cartesian state (henceforth referred to as X)
        X_0_sol, T_sol = dimensionalize(OE_f, T_f, self.lpe)
        OE_0_sol = cart2oe_tf(X_0_sol, self.lpe.mu_tilde, self.element_set).numpy()[0]
        X_0_sol = X_0_sol[0]

        return OE_0_sol, X_0_sol, T_sol, result

    def solve(self, OE_0_dim, T_dim, solution_bounds):
        X_0, V_0, V_solution_bounds, T = self.initialize_solver_args(OE_0_dim, T_dim, solution_bounds)
        result = self.solve_subroutine(X_0, V_0, V_solution_bounds)
        return self.prepare_outputs(X_0, T, result)
    
    @abstractmethod
    def solve_subroutine(self, X_0, V_0, V_solution_bounds):
        pass

class CartesianShootingLsSolver(CartesianShootingSolver):
    def solve_subroutine(self, X_0, V_0, V_solution_bounds):
        result = least_squares(constraint_shooting, V_0, jac=constraint_shooting_jac, 
                                    args=(
                                        self.lpe, 
                                        X_0,
                                        self.decision_variable_mask,
                                        self.constraint_variable_mask,
                                        self.constraint_angle_wrap_mask),
                                    bounds=V_solution_bounds,
                                    verbose=2,
                                    # xtol=1
                                    # xtol=None,
                                    # ftol=None,
                                    # method='dogbox'
                                    )
        return result

class CartesianShootingRootSolver(CartesianShootingSolver):
    def solve_subroutine(self, X_0, V_0, V_solution_bounds):
        result = root(constraint_shooting, V_0, jac=constraint_shooting_jac, 
                            args=(
                                self.lpe, 
                                X_0,
                                self.decision_variable_mask,
                                self.constraint_variable_mask,
                                self.constraint_angle_wrap_mask),
                                # tol=1E-20
                            # bounds=V_bounds_tuple,
                            )
        return result

