from scipy.integrate import solve_bvp, solve_ivp
import numpy as np
from GravNN.Support.transformations import invert_projection, cart2sph

def solve_ivp_oe_problem(T, OE, lpe, t_eval=None):
    def fun(x,y,p=None):
        "Return the first-order system"
        print(x)
        results = lpe(y.reshape((1,-1)))
        dxdt = np.array([v.numpy() for v in results.values()])
        return dxdt.reshape((-1,))
    
    sol = solve_ivp(fun, [0, T], OE.reshape((-1,)), t_eval=t_eval, atol=1e-8, rtol=1e-10)
    return sol
    

def solve_ivp_pos_problem(T, state, lpe, t_eval=None, events=None, args=None, atol=1e-8, rtol=1e-10):
    def fun(x,y,IC=None):
        "Return the first-order system"
        R = np.array([y[0:3]])
        V = np.array([y[3:6]])
        a = lpe.model.generate_acceleration(R)
        dxdt = np.hstack((V, a)).squeeze()
        return dxdt
    
    sol = solve_ivp(fun, [0, T], state.reshape((-1,)), t_eval=t_eval, events=events, atol=atol, rtol=rtol, args=args)
    return sol


def backprop_error_vec(T, state_extended, lpe, t_eval=None, events=None, args=None):
    def fun(x,y,IC=None):
        "Return the first-order system"

        R, V = y[0:3], y[3:6]
        # dR, dV = y[6:9], y[9:12]

        r = np.linalg.norm(R)
        a_pm_sph = np.array([[-lpe.mu/r**2, 0.0, 0.0]])
        r_sph = cart2sph(R.reshape((1,3)))
        a_pm_xyz = invert_projection(r_sph, a_pm_sph).reshape((3,))
        a = lpe.model.generate_acceleration(R.reshape((1,3)))
        dxdt = np.hstack((V, a_pm_xyz - a.reshape((3,))))
        dxdt = np.hstack((dxdt, dxdt))
        return dxdt.reshape((12,))
    
    sol = solve_ivp(fun, [0, T], state_extended.reshape((-1,)), t_eval=t_eval, events=events, atol=1e-8, rtol=1e-10, args=None)
    return sol