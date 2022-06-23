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
    OE_trad = np.array([[7.93E+04/2, 1.00E-02, 1.53E+00, -7.21E-01, -1.61E-01, 2.09e+00]])
    T = 2*np.pi*np.sqrt(OE_trad[0,0]**3/mu)
    return OE_trad, T

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
            t_eval=np.linspace(0,T, 100)
            )
    pbar.close()

    OE_sol.y = lpe.dimensionalize_state(OE_sol.y.T).numpy().T
    OE_sol.t = lpe.dimensionalize_time(OE_sol.t).numpy()

    # Propagate cartesian state using cartesian accelerations
    R_0, V_0 = oe2cart_tf(OE_0_dim, planet.mu, element_set)
    x_0 = np.hstack((R_0.numpy(), V_0.numpy()))[0]
    cart_sol = propagate_orbit(T_dim, x_0, model, tol=1E-7) 


    plot_3d_trajectory(cart_sol, Eros().obj_8k)
    plot_3d_trajectory(OE_sol, Eros().obj_8k)

    plot_energy(cart_sol, model)
    plot_energy(OE_sol, model)

    plt.show()


if __name__ == "__main__":
    main()