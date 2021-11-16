from scipy.integrate import solve_bvp, solve_ivp
import numpy as np
from GravNN.Support.transformations import invert_projection, cart2sph

def solve_ivp_problem(T, OE, lpe, t_eval=None):
    def fun(x,y,p=None):
        "Return the first-order system"
        print(x)
        results = lpe(y.reshape((1,-1)))
        dxdt = np.array([v.numpy() for v in results.values()])
        return dxdt.reshape((-1,))
    
    sol = solve_ivp(fun, [0, T], OE.reshape((-1,)), t_eval=t_eval, atol=1e-8, rtol=1e-10) #atol=1e-6, rtol=1e-8)
    y = sol.y
    y = np.array(y).squeeze()
    return y

def solve_ivp_pos_problem(T, state, lpe, t_eval=None):
    def fun(x,y,p=None):
        "Return the first-order system"
        print(x)
        R = y[0:3]
        V = y[3:6]

        r = np.linalg.norm(R)
        a_pm_sph = np.array([[-lpe.mu/r**2, 0.0, 0.0]])
        r_sph = cart2sph(R.reshape((1,3)))

        a_pm_xyz = invert_projection(r_sph, a_pm_sph).reshape((3,))

        a = lpe.model.generate_acceleration(R.reshape((1,3))).numpy()
        dxdt = np.hstack((V, a_pm_xyz - a.reshape((3,))))
        return dxdt.reshape((6,))
    

    sol = solve_ivp(fun, [0, T], state.reshape((-1,)), t_eval=t_eval, atol=1e-8, rtol=1e-10)

    t_plot = sol.t
    y_plot = sol.y # [6, N]

    return y_plot