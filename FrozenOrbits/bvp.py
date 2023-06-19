import copy
import time
from abc import ABC, abstractmethod

import numpy as np
from GravNN.Support.ProgressBar import ProgressBar
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares, minimize, root

from FrozenOrbits.constraints import *
from FrozenOrbits.coordinate_transforms import *
from FrozenOrbits.dynamics import *

np.set_printoptions(formatter={"float": "{0:0.2e}".format})

######################
## Helper Functions ##
######################


def debug_plot(sol, lpe, element_set):
    import matplotlib.pyplot as plt
    from GravNN.CelestialBodies.Asteroids import Eros

    from FrozenOrbits.visualization import plot_cartesian_state_3d, plot_OE_1d

    plot_OE_1d(sol.t, sol.y.T, element_set)

    OE_dim = lpe.dimensionalize_state(sol.y.T).numpy()
    cart = oe2cart_tf(OE_dim, Eros().mu, element_set).numpy()

    sol_cart = copy.deepcopy(sol)
    sol_cart.y = cart.T
    plot_cartesian_state_3d(sol_cart.t, sol_cart.y.T, Eros().obj_8k)
    plt.show()
    return


def evolve_state_w_STM(t, x_0, F, model, tol=1e-6):
    N = len(x_0)
    phi_0 = np.identity(N)
    z_i = np.hstack((x_0, phi_0.reshape((-1))))
    pbar = ProgressBar(t, enable=True)
    sol = solve_ivp(F, [0, t], z_i, args=(model, pbar), atol=tol, rtol=tol)
    pbar.close()
    if not sol.success:
        print(f"{sol.success} \t {sol.message}")
    x_f = sol.y[:N, -1]
    phi_f = sol.y[N:, -1].reshape((N, N))
    return x_f, phi_f


def print_result_info(result):
    print(
        f"""
    Success? ({result.success} \t Status: {result.status} 
    Message: {result.message}
    OE_0 = {result.x} 
    Number of Function Evals: {result.nfev} \t Number of Jacobian Evals: {result.njev}")
    """,
    )


def non_dimensionalize(OE_dim, T_dim, lpe):
    OE = lpe.non_dimensionalize_state(OE_dim).numpy()
    T = lpe.non_dimensionalize_time(T_dim).numpy()
    return OE, T


def dimensionalize(OE, T, lpe):
    OE_dim = lpe.dimensionalize_state(OE).numpy()
    T_dim = lpe.dimensionalize_time(T).numpy()
    return OE_dim, T_dim


def populate_masks(
    OE,
    decision_variable_mask,
    constraint_variable_mask,
    constraint_angle_wrap_mask,
):
    if decision_variable_mask is None:
        decision_variable_mask = [True] * (len(OE[0]) + 1)  # N + 1
    if constraint_variable_mask is None:
        constraint_variable_mask = [True] * len(OE[0] + 1)  # N
    if constraint_angle_wrap_mask is None:
        constraint_angle_wrap_mask = [False] * len(OE[0] + 1)  # N


#######################
## Custom algorithms ##
#######################


def variable_time_mirror_bvp(T, x0, model):
    T_i_p1 = copy.deepcopy(T) / 2
    x_i_p1 = copy.deepcopy(x0)

    k = 0
    tol = 1
    while tol > 1e-6 and k < 10:
        start_time = time.time()
        T_i = copy.deepcopy(T_i_p1)
        x_i = copy.deepcopy(x_i_p1)
        x_f, phi_f = evolve_state_w_STM(T_i, x_i, dynamics_cart_w_STM, model, tol=1e-6)

        x_dot_f = np.hstack((x_f[3:6], model.generate_acceleration(x_f[0:3])))

        V_i = np.array([x_i[0], x_i[4], T_i])
        C = np.array(
            [
                [x_f[1]],
                [x_f[3]],
            ],
        )
        D = np.array(
            [
                [phi_f[1, 0], phi_f[1, 4], x_dot_f[1]],
                [phi_f[3, 0], phi_f[3, 4], x_dot_f[3]],
            ],
        )

        V_i_p1 = V_i - np.transpose(D.T @ np.linalg.pinv(D @ D.T) @ C).squeeze()
        x_i_p1[0] = V_i_p1[0]
        x_i_p1[4] = V_i_p1[1]
        T_i_p1 = V_i_p1[2]
        tol = np.linalg.norm(C)
        dx = np.linalg.norm((x_i_p1 - x_i)[0:6])
        print(
            f"Iteration {k}: tol = {tol} \t dx_k = {dx} \t dT = {T_i_p1 - T_i} \t Time Elapsed: {time.time() - start_time}",
        )
        k += 1

    return x_i_p1, T_i_p1 * 2


def general_variable_time_cartesian_bvp(T, x0, model):
    T_i_p1 = copy.deepcopy(T)
    x_i_p1 = copy.deepcopy(x0)

    k = 0
    tol = 1
    while tol > 1e-6 and k < 10:
        start_time = time.time()
        T_i = copy.deepcopy(T_i_p1)
        x_i = copy.deepcopy(x_i_p1)
        len(x_i)

        # Propagate dynamics
        x_f, phi_f = evolve_state_w_STM(T_i, x_i, dynamics_cart_w_STM, model, tol=1e-6)

        # Get X_dot @ t_f
        r_f = x_f[0:3]
        v_f = x_f[3:6]
        a_f = model.generate_acceleration(r_f)
        x_dot_f = np.hstack((v_f, a_f)).reshape((6, 1))

        # Build BVP "state" vector
        V_i = np.hstack((x_i, T_i))

        # Define constraint vector and corresponding partials w.r.t. BVP state vector (dC/dV)
        C = x_f - x_i
        D = np.block([phi_f - np.eye(len(x0)), x_dot_f])

        # Compute correction term and apply
        dV = np.transpose(D.T @ np.linalg.pinv(D @ D.T) @ C).squeeze()
        V_i_p1 = V_i - dV

        # Map back to state
        x_i_p1 = V_i_p1[0 : len(x0)]
        T_i_p1 = V_i_p1[len(x0)]

        tol = np.linalg.norm(C)
        dx = np.linalg.norm((x_i_p1 - x_i)[0 : len(x0)])
        print(
            f"Iteration {k}: tol = {tol} \t dx_k = {dx} \t dT = {T_i_p1 - T_i} \t Time Elapsed: {time.time() - start_time}",
        )
        k += 1

    return x_i_p1, T_i_p1


def general_variable_time_bvp_trad_OE(
    T_dim,
    OE_0_dim,
    model,
    element_set,
    decision_variable_mask=None,
    constraint_variable_mask=None,
    constraint_angle_wrap_mask=None,
):
    OE_0 = model.non_dimensionalize_state(OE_0_dim).numpy()
    T = model.non_dimensionalize_time(T_dim).numpy()
    print(f"Total Time {T} \n Dim State {OE_0_dim} \n Non Dim State {OE_0}")
    T_i_p1 = copy.deepcopy(T)
    OE_i_p1 = copy.deepcopy(OE_0.reshape((-1,)))

    k = 0
    tol = 1
    while tol > 1e-6 and k < 10:
        T_i = copy.deepcopy(T_i_p1)
        x_i = copy.deepcopy(OE_i_p1)
        x_f, phi_f = evolve_state_w_STM(T_i, x_i, dynamics_OE_w_STM, model, tol=1e-6)
        OE_i_p1, T_i_p1 = OE_constraint(
            x_f,
            phi_f,
            x_i,
            T_i,
            model,
            k,
            decision_variable_mask=decision_variable_mask,
            constraint_variable_mask=constraint_variable_mask,
            constraint_angle_wrap_mask=constraint_angle_wrap_mask,
        )
        k += 1

    OE_i_p1 = model.dimensionalize_state(np.array([OE_i_p1])).numpy()
    T_i_p1 = model.dimensionalize_time(T_i_p1).numpy()
    x_i_p1 = oe2cart_tf(OE_i_p1, model.mu_tilde, element_set).numpy()[0]

    return OE_i_p1, x_i_p1, T_i_p1


##############################
## Solver Interface Classes ##
##############################


class ShootingSolver(ABC):
    def __init__(
        self,
        lpe,
        decision_variable_mask=None,
        constraint_variable_mask=None,
        constraint_angle_wrap_mask=None,
        rtol=1e-3,
        atol=1e-6,
        max_nfev=None,
    ):
        self.lpe = lpe
        self.element_set = lpe.element_set

        if decision_variable_mask is None:
            decision_variable_mask = [True] * (lpe.num_elements + 1)  # N + 1
        self.decision_variable_mask = decision_variable_mask
        if constraint_variable_mask is None:
            constraint_variable_mask = [True] * (lpe.num_elements + 1)  # N + 1
        self.constraint_variable_mask = constraint_variable_mask
        if constraint_angle_wrap_mask is None:
            constraint_angle_wrap_mask = [True] * (lpe.num_elements + 1)  # N + 1
        self.constraint_angle_wrap_mask = constraint_angle_wrap_mask
        self.rtol = rtol
        self.atol = atol
        self.max_nfev = max_nfev
        pass

    def initialize_solver_args(self, OE_0_dim, T_dim, solution_bounds):
        OE_0, T = non_dimensionalize(OE_0_dim, T_dim, self.lpe)
        X_0 = np.hstack(
            (OE_0.reshape((-1)), T),
        )  # Decision variables that can be updated
        V_0 = X_0[self.decision_variable_mask]
        V_solution_bounds = np.array(solution_bounds)[:, self.decision_variable_mask]

        print(f"OE_0 tilde : {OE_0_dim}")
        print(f"OE_0 : {OE_0}")

        return X_0, V_0, V_solution_bounds, T

    @abstractmethod
    def solve_subroutine(self, X_0, V_0, V_solution_bounds):
        pass

    def prepare_outputs(self, X_0, T, result):
        print_result_info(result)

        OE_f = update_state(X_0, result.x, self.decision_variable_mask)  # remove time
        T_f = OE_f[0, -1]  # The non-dim time
        OE_f = np.array([OE_f[0, :-1]])  # the non-dim OE

        OE_0_sol, T_sol = dimensionalize(OE_f, T_f, self.lpe)
        X_0_sol = oe2cart_tf(OE_0_sol, self.lpe.mu_tilde, self.element_set).numpy()[0]
        return OE_0_sol, X_0_sol, T_sol, result

    def solve(self, OE_0_dim, T_dim, solution_bounds):
        X_0, V_0, V_solution_bounds, T = self.initialize_solver_args(
            OE_0_dim,
            T_dim,
            solution_bounds,
        )
        result = self.solve_subroutine(X_0, V_0, V_solution_bounds)
        return self.prepare_outputs(X_0, T, result)


class ShootingLsSolver(ShootingSolver):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def solve_subroutine(self, X_0, V_0, V_solution_bounds):
        result = least_squares(
            constraint_shooting,
            V_0,
            jac=constraint_shooting_jac,
            args=(
                self.lpe,
                X_0,
                self.decision_variable_mask,
                self.constraint_variable_mask,
                self.constraint_angle_wrap_mask,
                self.rtol,
                self.atol,
            ),
            bounds=V_solution_bounds,
            verbose=2,
            max_nfev=self.max_nfev,
            # x_scale=[1.0, 1.0 ,np.pi, 2*np.pi, 2*np.pi, 2*np.pi, 1.0]
            # x_scale=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2*np.pi, 1.0]
            # xtol=1
            # xtol=None,
            # ftol=1E-4,
            # loss='soft_l1'
            # method='dogbox'
        )
        return result


class ShootingRootSolver(ShootingSolver):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def solve_subroutine(self, X_0, V_0, V_solution_bounds):
        V_bounds_tuple = []
        for i in range(len(V_solution_bounds[0])):
            V_bounds_tuple.append(tuple(V_solution_bounds[:, i]))

        result = root(
            constraint_shooting,
            V_0,
            jac=constraint_shooting_jac,
            args=(
                self.lpe,
                X_0,
                self.decision_variable_mask,
                self.constraint_variable_mask,
                self.constraint_angle_wrap_mask,
            ),
            # bounds=V_bounds_tuple,
        )
        return result


class ShootingMinimizeSolver(ShootingSolver):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def solve_subroutine(self, X_0, V_0, V_solution_bounds):
        V_bounds_tuple = []
        for i in range(len(V_solution_bounds[0])):
            V_bounds_tuple.append(tuple(V_solution_bounds[:, i]))

        result = minimize(
            constraint_shooting_scalar,
            V_0,  # jac=constraint_shooting_jac_scalar,
            args=(
                self.lpe,
                X_0,
                self.decision_variable_mask,
                self.constraint_variable_mask,
                self.constraint_angle_wrap_mask,
            ),
            bounds=V_bounds_tuple,
        )
        return result


#######################
## Cartesian Solvers ##
#######################
class CartesianShootingSolver(ABC):
    def __init__(
        self,
        lpe,
        decision_variable_mask=None,
        constraint_variable_mask=None,
        constraint_angle_wrap_mask=None,
        rtol=1e-3,
        atol=1e-6,
        max_nfev=None,
    ):
        self.lpe = lpe
        self.element_set = lpe.element_set

        if decision_variable_mask is None:
            decision_variable_mask = [True] * (lpe.num_elements + 1)  # N + 1
        self.decision_variable_mask = decision_variable_mask
        if constraint_variable_mask is None:
            constraint_variable_mask = [True] * (lpe.num_elements + 1)  # N + 1
        self.constraint_variable_mask = constraint_variable_mask
        if constraint_angle_wrap_mask is None:
            constraint_angle_wrap_mask = [True] * (lpe.num_elements + 1)  # N + 1
        self.constraint_angle_wrap_mask = constraint_angle_wrap_mask
        self.rtol = rtol
        self.atol = atol
        self.max_nfev = max_nfev
        pass

    def initialize_solver_args(self, OE_0_dim, T_dim, solution_bounds):
        OE_0, T = non_dimensionalize(OE_0_dim, T_dim, self.lpe)
        X_0 = np.hstack(
            (OE_0.reshape((-1)), T),
        )  # Decision variables that can be updated
        V_0 = X_0[self.decision_variable_mask]
        V_solution_bounds = np.array(solution_bounds)[:, self.decision_variable_mask]

        print(f"OE_0 tilde : {OE_0_dim}")
        print(f"OE_0 : {OE_0}")

        return X_0, V_0, V_solution_bounds, T

    def prepare_outputs(self, X_0, T, result):
        print_result_info(result)

        OE_f = update_state(X_0, result.x, self.decision_variable_mask)  # remove time
        T_f = OE_f[0, -1]  # The non-dim time
        OE_f = np.array([OE_f[0, :-1]])  # the non-dim OE

        # The OE set is actually cartesian state (henceforth referred to as X)
        X_0_sol, T_sol = dimensionalize(OE_f, T_f, self.lpe)
        OE_0_sol = cart2oe_tf(X_0_sol, self.lpe.mu_tilde, self.element_set).numpy()
        X_0_sol = X_0_sol[0]

        return OE_0_sol, X_0_sol, T_sol, result

    def solve(self, OE_0_dim, T_dim, solution_bounds):
        X_0, V_0, V_solution_bounds, T = self.initialize_solver_args(
            OE_0_dim,
            T_dim,
            solution_bounds,
        )
        result = self.solve_subroutine(X_0, V_0, V_solution_bounds)
        return self.prepare_outputs(X_0, T, result)

    @abstractmethod
    def solve_subroutine(self, X_0, V_0, V_solution_bounds):
        pass


class CartesianShootingLsSolver(CartesianShootingSolver):
    def solve_subroutine(self, X_0, V_0, V_solution_bounds):
        result = least_squares(
            constraint_shooting,
            V_0,
            jac=constraint_shooting_jac,
            args=(
                self.lpe,
                X_0,
                self.decision_variable_mask,
                self.constraint_variable_mask,
                self.constraint_angle_wrap_mask,
                self.rtol,
                self.atol,
            ),
            bounds=V_solution_bounds,
            verbose=2,
            max_nfev=self.max_nfev,
            # xtol=1
            # xtol=None,
            # ftol=None,
            # method='dogbox'
        )
        return result


class CartesianShootingRootSolver(CartesianShootingSolver):
    def solve_subroutine(self, X_0, V_0, V_solution_bounds):
        result = root(
            constraint_shooting,
            V_0,
            jac=constraint_shooting_jac,
            args=(
                self.lpe,
                X_0,
                self.decision_variable_mask,
                self.constraint_variable_mask,
                self.constraint_angle_wrap_mask,
            ),
            # tol=1E-20
            # bounds=V_bounds_tuple,
        )
        return result
