import os
import copy
from FrozenOrbits.analysis import check_for_intersection, print_state_differences
from FrozenOrbits.bvp import *

import GravNN
import matplotlib.pyplot as plt
import numpy as np
from FrozenOrbits.boundary_conditions import *
from FrozenOrbits.gravity_models import (pinnGravityModel,
                                         polyhedralGravityModel)
from FrozenOrbits.LPE import *
from FrozenOrbits.utils import propagate_orbit
from FrozenOrbits.visualization import *
from GravNN.CelestialBodies.Asteroids import Eros
from FrozenOrbits.coordinate_transforms import cart2oe_tf
import OrbitalElements.orbitalPlotting as op
from FrozenOrbits.constraints import *

from GravNN.Networks.Layers import PreprocessingLayer, PostprocessingLayer

def sample_mirror_orbit(R, mu):
    r_mag = 5*R
    T = 2*np.pi*np.sqrt(r_mag**3/mu)
    OE = oe2milankovitch_tf(np.array([[r_mag, 0.4, np.pi/4, np.pi/4, np.pi/4, np.pi/4]]), mu).numpy()
    return OE, T

def compute_energy(sol, model):
    x = sol.y
    U = model.generate_potential(x[0:3,:].reshape((3,-1)).T).squeeze()
    T = (np.linalg.norm(x[3:6,:], axis=0)**2 / 2.0).squeeze()
    E = U + T
    return E

def plot_energy(sol, model):
    E = compute_energy(sol, model)         
    plt.figure()
    plt.plot(sol.t, E)
    plt.xlabel("Time")
    plt.ylabel("Energy")


def main():
    """Solve a BVP problem using the dynamics of the cartesian state vector"""
    planet = Eros()
    np.random.seed(15)
    # tf.config.run_functions_eagerly(True)

    model = pinnGravityModel(os.path.dirname(GravNN.__file__) + \
        "/../Data/Dataframes/eros_BVP_PINN_III.data")  

    OE_0, T = sample_mirror_orbit(planet.radius, planet.mu)

    H_tilde_mag = np.linalg.norm(OE_0[0,0:3])
    H_mag = 1.0
    # H = t/l**2 * H_tilde
    t_star = T
    l_star = np.sqrt(t_star*H_tilde_mag/H_mag)


    # Run the solver
    lpe = LPE_Milankovitch(model.gravity_model, planet.mu, 
                                l_star=l_star, 
                                t_star=t_star, 
                                m_star=planet.mu/(6.67430*1E-11))
    element_set = 'milankovitch'

    # normalized coordinates for semi-major axis
    bounds = ([-2*1/np.sqrt(3), -2*1/np.sqrt(3), -2*1/np.sqrt(3), -np.inf, -np.inf, -np.inf, 0.9*OE_0[0,6]],
              [2*1/np.sqrt(3), 2*1/np.sqrt(3), 2*1/np.sqrt(3), np.inf, np.inf, np.inf, 1.1*OE_0[0,6]])

    OE_0_sol, T_sol = general_variable_time_bvp_mil_OE_ND_scipy(T, OE_0, lpe, bounds)

    print(f"Initial OE: {OE_0}")
    print(f"BVP OE: {OE_0_sol}")


    # propagate the initial and solution orbits
    R_0, V_0 = oe2cart_tf(OE_0, planet.mu, element_set)
    x_0 = np.hstack((R_0.numpy(), V_0.numpy()))[0]

    init_sol = propagate_orbit(T, x_0, model, tol=1E-7) 
    plot_3d_trajectory(init_sol, planet.obj_8k)
    plt.title("Initial Guess")
    
    R_0_sol, V_0_sol = oe2cart_tf(OE_0_sol, planet.mu, element_set)
    x_0_sol = np.hstack((R_0_sol.numpy(), V_0_sol.numpy()))[0]
    bvp_sol = propagate_orbit(T_sol, x_0_sol, model, tol=1E-10) 
    check_for_intersection(bvp_sol, planet.obj_8k)
    print_state_differences(bvp_sol)

    plot_3d_trajectory(bvp_sol, planet.obj_8k)
    plt.title("BVP Solution")

    oe_list = []
    for i in range(len(bvp_sol.t)):
        y = bvp_sol.y[:,i].reshape((1,6))
        mil_oe = cart2milankovitch_tf(y, planet.mu)        
        oe_list.append(mil_oe[0,:].numpy())
    oe_array = np.array(oe_list).squeeze()
    ND_OE = lpe.non_dimensionalize_state(oe_array).numpy()
    op.plot_OE(bvp_sol.t, ND_OE.T, OE_set=element_set)

    plt.show()


if __name__ == "__main__":
    main()
