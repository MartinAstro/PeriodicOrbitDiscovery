import os
from FrozenOrbits.bvp import *

import GravNN
import matplotlib.pyplot as plt
import numpy as np
from FrozenOrbits.boundary_conditions import *
from FrozenOrbits.gravity_models import (pinnGravityModel)
from FrozenOrbits.LPE import *
from FrozenOrbits.utils import propagate_orbit
from FrozenOrbits.visualization import *
from GravNN.CelestialBodies.Asteroids import Eros

def plot_milankovitch_OE(OE):
    """OE [N x 6]"""
    plt.figure()
    plt.subplot(4,2,1)
    plt.plot(OE[:,0])
    plt.subplot(4,2,2)
    plt.plot(OE[:,1])
    plt.subplot(4,2,3)
    plt.plot(OE[:,2])
    plt.subplot(4,2,4)
    plt.plot(OE[:,3])
    plt.subplot(4,2,5)
    plt.plot(OE[:,4])
    plt.subplot(4,2,6)
    plt.plot(OE[:,5])
    plt.subplot(4,2,7)
    plt.plot(OE[:,6])

def sample_mirror_orbit(R, mu):
    # OE_trad = np.array([[7.93E+04/2, 1.00E-02, 1.53E+00, -7.21E-01, -1.61E-01, 2.09e+00]]) # Problems when eVec -> 0.0, there is some numerical challenges
    OE_trad = np.array([[7.93E+04/2, 1.00E-01, 1.53E+00, -7.21E-01, -1.61E-01, 2.09e+00]])
    T = 2*np.pi*np.sqrt(OE_trad[0,0]**3/mu)
    OE = oe2milankovitch_tf(OE_trad, mu).numpy()
    return OE, T

def main():
    """Solve a BVP problem using the dynamics of the cartesian state vector"""
    # tf.config.run_functions_eagerly(True)
    planet = Eros()
    np.random.seed(15)

    model = pinnGravityModel(os.path.dirname(GravNN.__file__) + \
        "/../Data/Dataframes/eros_BVP_PINN_III.data")  

    OE_0_dim, T_dim = sample_mirror_orbit(planet.radius, planet.mu)

    H_tilde_mag = np.linalg.norm(OE_0_dim[0,0:3])
    H_mag = 1.0

    # Run the solver
    element_set = 'milankovitch'
    lpe = LPE_Milankovitch_2(model.gravity_model, planet.mu, 
                                l_star=np.sqrt(T_dim*H_tilde_mag/H_mag), 
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
            t_eval=np.linspace(0,T, 100)
            )
    pbar.close()

    OE_sol.y = lpe.dimensionalize_state(OE_sol.y.T).numpy().T
    OE_sol.t = lpe.dimensionalize_time(OE_sol.t).numpy()
    plot_milankovitch_OE(OE_sol.y.T)
    plt.suptitle("OE from Integrated OE0")

    # Convert OE solution to cartesian
    OE_2_cart_sol = copy.deepcopy(OE_sol)
    OE_2_cart_sol.y = oe2cart_tf(OE_sol.y.T, planet.mu, element_set).numpy().T

    # Propagate cartesian state using cartesian accelerations
    x_0 = oe2cart_tf(OE_0_dim, planet.mu, element_set)[0]
    cart_sol = propagate_orbit(T_dim, x_0, model, tol=1E-7) 
    cart_sol_OE = cart2milankovitch_tf(cart_sol.y.T, planet.mu)

    plot_milankovitch_OE(cart_sol_OE)
    plt.suptitle("OE from Integrated X0")

    # Plot differences between the solutions
    dOE = OE_sol.y.T - cart_sol_OE
    plot_milankovitch_OE(dOE)
    plt.suptitle("delta OE")

    plot_energy(cart_sol, model)
    plot_energy(OE_2_cart_sol, model)

    plot_3d_trajectory(cart_sol, Eros().obj_8k)
    plot_3d_trajectory(OE_2_cart_sol, Eros().obj_8k)

    plt.show()


if __name__ == "__main__":
    main()