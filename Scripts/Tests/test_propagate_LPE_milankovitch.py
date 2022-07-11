import os
from FrozenOrbits.analysis import print_OE_differences
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
from Scripts.BVP.initial_conditions import near_periodic_IC

def main():
    """Solve a BVP problem using the dynamics of the cartesian state vector"""
    # tf.config.run_functions_eagerly(True)
    planet = Eros()
    np.random.seed(15)

    model = pinnGravityModel(os.path.dirname(GravNN.__file__) + \
        "/../Data/Dataframes/eros_BVP_PINN_III.data")  


    # Configuration 1
    OE_0_dim = np.array([[-1.4698e+04, -9.0480e+04, 9.5481e+04, 6.6627e-02, -5.8877e-02, -4.6041e-02 , 1.2080e+00]])
    X_0_dim = oe2cart_tf(OE_0_dim, planet.mu, element_set="milankovitch").numpy()
    T_dim = 74396.96080201163

    lpe = LPE_Milankovitch(model.gravity_model, planet.mu, 
                        l_star=np.sqrt(T_dim*np.linalg.norm(OE_0_dim[0,0:3])/1.0),  # H = t/l**2 * H_tilde
                        t_star=T_dim, 
                        m_star=1.0)


    # Configuration 2
    OE_0_dim = np.array([[-1.51e+04, -9.02e+04, 9.56e+04, 5.41e-02, -8.54e-02, -8.92e-02, 1.21e+00]])
    X_0_dim = oe2cart_tf(OE_0_dim, planet.mu, element_set="milankovitch").numpy()
    T_dim =74608.01941682874

    lpe = LPE_Milankovitch(model.gravity_model, planet.mu, 
                            l_star=np.sqrt(T_dim*np.linalg.norm(OE_0_dim[0,0:3])/10.0),  # H = t/l**2 * H_tilde
                            t_star=T_dim, 
                            m_star=1.0)


    # Propagate OE using LPE
    OE_0 = lpe.non_dimensionalize_state(OE_0_dim.reshape((1,-1))).numpy().squeeze()
    T = lpe.non_dimensionalize_time(T_dim).numpy().squeeze()

    pbar = ProgressBar(T, enable=True)
    OE_sol = solve_ivp(dynamics_OE, 
            [0, T],
            OE_0,
            args=(lpe, pbar),
            atol=1E-10, rtol=1E-10,
            t_eval=np.linspace(0,T,100)
            )
    pbar.close()

    OE_sol.y = lpe.dimensionalize_state(OE_sol.y.T).numpy().T
    OE_sol.t = lpe.dimensionalize_time(OE_sol.t).numpy()

    # Propagate the cartesian state using gravity model + convert to OE
    init_sol = propagate_orbit(T_dim, X_0_dim[0], model, tol=1E-10)
    init_sol.y = cart2oe_tf(init_sol.y.T, planet.mu, lpe.element_set).numpy().T    
    
    # Print the X_f - X_0 differences in both cases
    constraint_angle_wrap = [False, False, False, False, False, False, True, False]
    OE_lpe_dim_diff, OE_lpe_diff = print_OE_differences(OE_sol.y.T, lpe, "OE Integration", constraint_angle_wrap)
    OE_cart_dim_diff, OE_cart_diff = print_OE_differences(init_sol.y.T, lpe, "Cartesian Integration", constraint_angle_wrap)


    cart_IC_a = oe2cart_tf(init_sol.y[:,0].reshape((1,-1)), planet.mu, element_set='milankovitch').numpy()[0]
    cart_IC_b = oe2cart_tf(OE_sol.y[:,0].reshape((1,-1)), planet.mu, element_set='milankovitch').numpy()[0]
    assert np.allclose(cart_IC_a, cart_IC_b), \
        f"Cartesian value of both initial coordinates are not the same! Cartesian OE -> Cart {cart_IC_a} \n Lagrange  OE -> Cart {cart_IC_b}"

    # assert that both integration methods yield the same t0 coordinate
    assert np.allclose(init_sol.y[:,0], OE_sol.y[:,0]), \
        f"Initial Coordinates are not the same!\n Cartesian OE {init_sol.y[:,0]} \n Lagrange  OE {OE_sol.y[:,0]}"
    
    # assert that both integration methods yield the same tf coordinate
    assert np.allclose(init_sol.y[:,-1], OE_sol.y[:,-1]),\
        f"Final Coordinates are not the same!\n Cartesian OE {init_sol.y[:,-1]} \n Lagrange  OE {OE_sol.y[:,-1]}"


    # # Plot the coordinates if desired
    # plot_OE_1d(OE_sol.t, OE_sol.y.T, lpe.element_set)
    # plot_OE_1d(init_sol.t, init_sol.y.T, lpe.element_set)
    
    # # Plot the differences
    # plot_OE_1d(init_sol.t, init_sol.y.T-OE_sol.y.T, lpe.element_set)

    plt.show()


if __name__ == "__main__":
    main()