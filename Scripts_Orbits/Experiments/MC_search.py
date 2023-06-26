import os
import pickle
import sys

import GravNN
import numpy as np
from GravNN.CelestialBodies.Asteroids import Eros

import FrozenOrbits
from FrozenOrbits.bvp import *
from FrozenOrbits.constraints import *
from FrozenOrbits.gravity_models import pinnGravityModel
from FrozenOrbits.LPE import *
from FrozenOrbits.visualization import *
from Scripts_Orbits.BVP.initial_conditions import *
from Scripts_Orbits.BVP.scipy_bvp_cart_OE import bvp_cart_OE
from Scripts_Orbits.BVP.scipy_bvp_equi_OE import bvp_equi_OE
from Scripts_Orbits.BVP.scipy_bvp_mil_OE import bvp_mil_OE
from Scripts_Orbits.BVP.scipy_bvp_trad_OE import bvp_trad_OE


def sample_initial_conditions(idx):
    planet = Eros()
    for _ in range(idx):
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

    solvers = {
        "trad": bvp_trad_OE,
        "equi": bvp_equi_OE,
        "mil": bvp_mil_OE,
        "cart": bvp_cart_OE,
    }

    # replace command line arguments for debugging
    # sys.argv = ["", "trad", "0"]

    solver_key = sys.argv[1]
    index = int(sys.argv[2])
    solver = solvers.get(solver_key, None)

    directory = os.path.dirname(FrozenOrbits.__file__) + "/Data/"

    print(f"Iteration {index}")
    OE_0, X_0, T_0, planet = sample_initial_conditions(index)
    data = solver(OE_0, X_0, T_0, planet, model, show=False)
    data["index"] = index
    data["solver_key"] = [solver_key]

    # save data to pickle
    os.makedirs(directory + "MC/", exist_ok=True)
    with open(directory + f"MC/orbit_solutions_{solver_key}_{index}.data", "wb") as f:
        pickle.dump(data, f)


if __name__ == "__main__":
    main()
