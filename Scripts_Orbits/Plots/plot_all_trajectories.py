import os

import GravNN
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from GravNN.CelestialBodies.Asteroids import Eros

import FrozenOrbits
from FrozenOrbits.bvp import *
from FrozenOrbits.constraints import *
from FrozenOrbits.gravity_models import pinnGravityModel
from FrozenOrbits.LPE import *
from FrozenOrbits.utils import propagate_orbit
from FrozenOrbits.visualization import *
from Scripts_Orbits.BVP.initial_conditions import *


def main():
    """Solve a BVP problem using the dynamics of the cartesian state vector"""
    np.random.seed(15)
    planet = Eros()
    directory = os.path.dirname(FrozenOrbits.__file__)
    # df = pd.read_pickle(directory + "/../Data/MC/orbit_solutions_cart.data")
    df = pd.read_pickle(directory + "/../Data/MC/orbit_solutions_trad.data")

    model = pinnGravityModel(
        os.path.dirname(GravNN.__file__) + "/../Data/Dataframes/eros_poly_071123.data",
    )

    # make a column in the dataframe calculating the norm of dX_0_sol
    dX_vec = [np.array(df["dX_0_sol"].values[i], dtype=float) for i in range(len(df))]
    df["distance"] = np.linalg.norm(dX_vec, axis=1)

    # filter out solutions for which this distance is greater than 100 m
    df = df[df["distance"] < 100]

    # filter out an invalid solutions
    df = df[df["valid"] == True]

    for i in range(10):  # len(df)):
        X_0 = df["X_0_sol"].values[i]
        T = df["T_0_sol"].values[i]

        # propagate the initial and solution orbits
        init_sol = propagate_orbit(T, X_0, model, tol=1e-9)
        new_fig = True if i == 0 else False
        plot_cartesian_state_3d(init_sol.y.T, planet.obj_8k, new_fig=new_fig)

    # OE_trad_init = cart2trad_tf(init_sol.y.T, planet.mu).numpy()
    # print_OE_differences(OE_trad_init, lpe, "IVP", constraint_angle_wrap)
    # plot_OE_1d(init_sol.t, OE_trad_init, "traditional", y0_hline=True)
    plt.savefig(directory + "/../Plots/eros_orbits.png")
    plt.show()


if __name__ == "__main__":
    main()
