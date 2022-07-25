import os
from FrozenOrbits.bvp import *

import GravNN
import matplotlib.pyplot as plt
import numpy as np
from FrozenOrbits.boundary_conditions import *
from FrozenOrbits.gravity_models import pinnGravityModel
from FrozenOrbits.LPE import *
from FrozenOrbits.utils import propagate_orbit
from FrozenOrbits.visualization import *
from GravNN.CelestialBodies.Asteroids import Eros

def sample_mirror_orbit(R, mu):
    # OE_trad = np.array([[7.93E+04/2, 1.00E-02, 1.53E+00, -7.21E-01, -1.61E-01, 2.09e+00]])
    # Recall traditional LPE break down for e ~ 0 b/c e in demonimator 
    OE_trad = np.array([[7.93E+04/2, 1.00E-01, 1.53E+00, -7.21E-01, -1.61E-01, 2.09e+00]])
    T = 2*np.pi*np.sqrt(OE_trad[0,0]**3/mu)
    return OE_trad, T

def compute_energy(sol, model):
    x = sol.y
    U = model.generate_potential(x[0:3,:].reshape((3,-1)).T).squeeze()
    T = (np.linalg.norm(x[3:6,:], axis=0)**2 / 2.0).squeeze()
    E = U + T
    return E

def main():
    """Solve a BVP problem using the dynamics of the cartesian state vector"""
    # tf.config.run_functions_eagerly(True)
    planet = Eros()
    np.random.seed(15)

    model = pinnGravityModel(os.path.dirname(GravNN.__file__) + \
        "/../Data/Dataframes/eros_BVP_PINN_III.data")  

    OE_0_dim, T_dim = sample_mirror_orbit(planet.radius, planet.mu)

    # Run the solver
    element_set = 'traditional'
    lpe = LPE_Traditional(model.gravity_model, planet.mu, 
                                l_star=OE_0_dim[0,0], 
                                t_star=T_dim, 
                                m_star=1.0)

    # Propagate OE using LPE
    OE_0 = lpe.non_dimensionalize_state(OE_0_dim).numpy().squeeze()
    T = lpe.non_dimensionalize_time(T_dim).numpy().squeeze()

    pbar = ProgressBar(T, enable=True)
    OE_sol = solve_ivp(dynamics_OE, 
            [0, T],
            OE_0,
            args=(lpe, pbar),
            atol=1E-7, rtol=1E-7,
            t_eval=np.linspace(0,T,100)
            )
    pbar.close()

    OE_sol.y = lpe.dimensionalize_state(OE_sol.y.T).numpy().T
    OE_sol.t = lpe.dimensionalize_time(OE_sol.t).numpy()

    # Convert to cartesian
    OE_sol.y = oe2cart_tf(OE_sol.y.T, planet.mu, element_set).numpy().T

    # Propagate cartesian state using cartesian accelerations
    x_0 = oe2cart_tf(OE_0_dim, planet.mu, element_set)[0]
    cart_sol = propagate_orbit(T_dim, x_0, model, tol=1E-7) 

    E_cart = compute_energy(cart_sol, model)
    E_OE = compute_energy(OE_sol, model)

    dE_cart = np.abs(E_cart - E_cart[0])
    dE_OE = np.abs(E_OE - E_OE[0])
    
    dE_cart_max = np.max(dE_cart)
    dE_OE_max = np.max(dE_OE)

    dE_cart_tol = 1E-5
    dE_OE_tol = 1E-4
    assert dE_cart_max < dE_cart_tol, f"Conservation of energy in violation of tolerance. Max of dE {dE_cart_max} > {dE_cart_tol}"
    assert dE_OE_max < dE_OE_tol, f"Conservation of energy in violation of tolerance. Max of dE {dE_OE_max} > {dE_OE_tol}"

    # plot_energy(cart_sol, model)
    # plot_energy(OE_sol, model)

    # plot_3d_trajectory(cart_sol, Eros().obj_8k)
    # plot_3d_trajectory(OE_sol, Eros().obj_8k)

    plt.show()


if __name__ == "__main__":
    main()