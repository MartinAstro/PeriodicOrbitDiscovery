import os

import GravNN
import numpy as np
import pandas as pd
from GravNN.CelestialBodies.Asteroids import Eros

import FrozenOrbits
from FrozenOrbits.bvp import *
from FrozenOrbits.constraints import *
from FrozenOrbits.gravity_models import pinnGravityModel
from FrozenOrbits.LPE import *
from FrozenOrbits.visualization import *
from Scripts_Orbits.BVP.initial_conditions import *
from Scripts_Orbits.BVP.scipy_bvp_cart_OE import bvp_cart_OE


def sample_initial_conditions():
    planet = Eros()
    a = np.random.uniform(3 * planet.radius, 7 * planet.radius)
    e = np.random.uniform(0.1, 0.3)
    i = np.random.uniform(-np.pi, np.pi)
    omega = np.random.uniform(0.0, 2 * np.pi)
    Omega = np.random.uniform(0.0, 2 * np.pi)
    M = np.random.uniform(0.0, 2 * np.pi)

    trad_OE = np.array([[a, e, i, omega, Omega, M]])
    X = trad2cart_tf(trad_OE, planet.mu).numpy()[0]
    T = 2 * np.pi * np.sqrt(trad_OE[0, 0] ** 3 / planet.mu)
    return trad_OE, X, T, planet


def main():
    """Solve a BVP problem using the dynamics of the cartesian state vector"""
    np.random.seed(15)

    model = pinnGravityModel(
        os.path.dirname(GravNN.__file__) + "/../Data/Dataframes/eros_poly_061523.data",
    )

    directory = os.path.dirname(FrozenOrbits.__file__) + "/Data/"

    df = pd.DataFrame(
        {
            "T_0": [],
            "T_0_sol": [],
            "OE_0": [],
            "OE_0_sol": [],
            "X_0": [],
            "X_0_sol": [],
            "dOE_0": [],
            "dOE_sol": [],
            "dX_0": [],
            "dX_sol": [],
            "result": [],
        },
    )

    for k in range(10):
        print(f"Iteration {k}")
        OE_0, X_0, T_0, planet = sample_initial_conditions()
        data = bvp_cart_OE(OE_0, X_0, T_0, planet, model, show=False)
        data["index"] = k

        df_k = pd.DataFrame().from_dict(data).set_index("index")
        df = pd.concat([df, df_k], axis=0)

    pd.to_pickle(df, directory + "cartesian_fine_orbit_solutions.data")


if __name__ == "__main__":
    main()
