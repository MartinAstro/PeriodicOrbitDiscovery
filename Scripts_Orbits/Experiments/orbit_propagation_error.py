import os
import pickle
import time

import GravNN
import numpy as np
import pandas as pd
from GravNN.CelestialBodies.Asteroids import Eros

import FrozenOrbits
from FrozenOrbits.bvp import *
from FrozenOrbits.constraints import *
from FrozenOrbits.gravity_models import pinnGravityModel, polyhedralGravityModel
from FrozenOrbits.LPE import *
from FrozenOrbits.utils import propagate_orbit
from FrozenOrbits.visualization import *
from Scripts_Orbits.BVP.initial_conditions import *

np.random.seed(15)


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


# a function that tries to load the output of the propagate_orbit function from a pickle
# if that function is not available, run the propagate orbit and pickle the output
def propagate_orbit_wrapped(T_0, X_0, model, model_name, tol=1e-9):
    directory = os.path.dirname(FrozenOrbits.__file__) + "/../Data/Prop/"
    os.makedirs(directory, exist_ok=True)

    file_name = directory + f"propagation_time_error_{T_0}_{X_0}_{model_name}.data"

    if os.path.exists(file_name):
        with open(file_name, "rb") as f:
            sol = pickle.load(f)
            dt = pickle.load(f)
    else:
        start_time = time.time()
        sol = propagate_orbit(T_0, X_0, model, tol=tol)
        dt = time.time() - start_time

        with open(file_name, "wb") as f:
            pickle.dump(sol, f)
            pickle.dump(dt, f)
    return dt, sol


def main():
    """Solve a BVP problem using the dynamics of the cartesian state vector"""

    model = pinnGravityModel(
        os.path.dirname(GravNN.__file__) + "/../Data/Dataframes/eros_poly_071123.data",
    )

    planet = model.config["planet"][0]
    poly_model = polyhedralGravityModel(planet, planet.obj_8k)
    poly_200_model = polyhedralGravityModel(planet, planet.obj_200k)

    df = pd.DataFrame(
        {
            "dt_pinn": [],
            "dt_poly": [],
            "Xf_pinn": [],
            "Xf_poly": [],
        },
    )

    for k in range(3):
        print(f"Iteration {k}")
        OE_0, X_0, T_0, planet = sample_initial_conditions()

        # propagate the initial and solution orbits
        pinn_start_time = time.time()
        pinn_sol = propagate_orbit(T_0, X_0, model, tol=1e-9)
        dt_pinn = time.time() - pinn_start_time

        dt_poly, poly_sol = propagate_orbit_wrapped(
            T_0,
            X_0,
            poly_model,
            "poly_model",
            tol=1e-9,
        )

        dt_poly_200, poly_200_sol = propagate_orbit_wrapped(
            T_0,
            X_0,
            poly_200_model,
            "poly_200_model",
            tol=1e-9,
        )

        df_k = (
            pd.DataFrame()
            .from_dict(
                {
                    "index": [k],
                    "dt_pinn": [dt_pinn],
                    "dt_poly": [dt_poly],
                    "dt_poly_200": [dt_poly_200],
                    "pinn_sol": [pinn_sol],
                    "poly_sol": [poly_sol],
                    "poly_200_sol": [poly_200_sol],
                    "Xf_pinn": [pinn_sol.y[:, -1]],
                    "Xf_poly_8k": [poly_sol.y[:, -1]],
                    "Xf_poly_200k": [poly_200_sol.y[:, -1]],
                    "semi": [OE_0[0, 0]],
                },
            )
            .set_index("index")
        )

        df = pd.concat([df, df_k], axis=0)

    directory = os.path.dirname(FrozenOrbits.__file__) + "/Data/"
    os.makedirs(directory, exist_ok=True)
    pd.to_pickle(df, directory + "propagation_time_error.data")


if __name__ == "__main__":
    main()
