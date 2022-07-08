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
from Scripts.BVP.initial_conditions import near_periodic_IC

def main():
    """Solve a BVP problem using the dynamics of the cartesian state vector"""
    # tf.config.run_functions_eagerly(True)
    planet = Eros()
    np.random.seed(15)

    model = pinnGravityModel(os.path.dirname(GravNN.__file__) + \
        "/../Data/Dataframes/eros_BVP_PINN_III.data")  

    OE_0_dim, X_0_dim, T_dim, planet = near_periodic_IC()

    # Run the solver
    scale = 1.0
    lpe = LPE_Cartesian(model.gravity_model, planet.mu, 
                                l_star=np.linalg.norm(X_0_dim[0:3])/scale, 
                                t_star=T_dim, 
                                m_star=1.0)#planet.mu/(6.67430*1E-11))
    # scale = 1.0
    # lpe = LPE_Cartesian(model.gravity_model, planet.mu, 
    #                             l_star=1.0, 
    #                             t_star=1.0, 
    #                             m_star=1.0)#planet.mu/(6.67430*1E-11))

    # Propagate OE using LPE
    OE_0 = lpe.non_dimensionalize_state(X_0_dim.reshape((1,-1))).numpy().squeeze()
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

    plot_cartesian_state_3d(OE_sol.y.T, planet.obj_8k)
    plt.show()


if __name__ == "__main__":
    main()