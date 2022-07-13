import os
import copy
import time
from FrozenOrbits.analysis import check_for_intersection, print_OE_differences, print_state_differences
from FrozenOrbits.bvp import *

import FrozenOrbits
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
import OrbitalElements.orbitalPlotting as op
from FrozenOrbits.constraints import *
from Scripts.BVP.initial_conditions import *
np.set_printoptions(formatter={'float': "{0:0.2e}".format})

def sample_initial_conditions(df, k):
    planet = Eros()
    OE_0 = np.array([df.iloc[k]["OE_0_sol"]])
    T_0 = df.iloc[k]["T_0_sol"]
    X_0 = np.array(df.iloc[k]["X_0_sol"])

    # Don't include the bounded OE scenario
    # replace with the initial conditions instead
    if df.index[k] == "OE_Bounded":
        OE_0 = np.array([df.iloc[k]["OE_0"]])
        T_0 = df.iloc[k]["T_0"]
        X_0 = np.array(df.iloc[k]["X_0"])

    return OE_0, X_0, T_0, planet


def main():
    directory =  os.path.dirname(FrozenOrbits.__file__)+ "/Data/"
    df = pd.read_pickle(directory + "constrained_orbit_solutions.data")
    for k in range(len(df)):
        new_fig = True if k == 0 else False
        OE_0, X_0, T, planet = sample_initial_conditions(df, k)           

        if df.index[k] == 'cartesian':
            OE_0 = cart2oe_tf(OE_0, planet.mu)
            label = "Cartesian"
        elif df.index[k] == "OE_bounded":
            label = "Initial Condition"
        elif df.index[k] == "OE_constrained":
            label = "OE Constrained"
        elif df.index[k] == "OE":
            label = "Orbital Elements"

        print(f"{label} & {OE_0[0,0]} & {OE_0[0,1]} & {OE_0[0,2]} & {OE_0[0,3]} & {OE_0[0,4]} & {OE_0[0,5]} & {T} \\\\ \n \\hline ")
    plt.show()

if __name__ == "__main__":
    main()
