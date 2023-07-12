import os

import matplotlib.pyplot as plt
import numpy as np
from GravNN.CelestialBodies.Asteroids import Eros
import pandas as pd
import FrozenOrbits
from FrozenOrbits.bvp import *
from FrozenOrbits.constraints import *
from FrozenOrbits.LPE import *
from FrozenOrbits.visualization import *
from Scripts_Orbits.BVP.initial_conditions import *
from sigfig import round
np.set_printoptions(formatter={"float": "{0:0.2e}".format})


def get_original_IC(df, k):
    T_0 = df.iloc[k]["T_0"]
    X_0 = np.array(df.iloc[k]["X_0"])
    return T_0, X_0


def get_corrected_IC(df, k):
    T_0 = df.iloc[k]["T_0_sol"]
    X_0 = np.array(df.iloc[k]["X_0_sol"])
    return T_0, X_0


def round_elements(x):
    X_0[0:3] = round(x, decimals=2)
    X_0[3:6] = np.array([round(x, decimals=2) for x in X_0[0:3].tolist()])
    return X_0

def main():
    directory = os.path.dirname(FrozenOrbits.__file__) + "/Data/"
    df = pd.read_pickle(directory + "constrained_orbit_solutions.data")
    planet = Eros()
    # Print the original initial condition first
    T_0, X_0 = get_original_IC(df, 0)  # use 0 b/c they all share same IC
    OE_0 = cart2oe_tf(np.array([X_0]), planet.mu)

    # separate the elements of the array stored in df.OE_0_sol into individual columns
    df[["a", "e", "i", "omega", "Omega", "M"]] = df["OE_0_sol"].apply(pd.Series)
    df['dX_scalar'] = pd.Series(df.dX_sol.apply(lambda x: np.linalg.norm(x)))
    df['dX_0_scalar'] = pd.Series(df.dX_0.apply(lambda x: np.linalg.norm(x)))
    
    # make a row for the initial OE conditions
    IC_df = pd.DataFrame(df.OE_0.iloc[1]).T
    IC_df.index = ["Initial Conditions"]
    print(IC_df.to_latex(index=True, float_format="{:0.2f}".format))
    print(df.dX_0_scalar)
    

    # drop all remaining columns
    df = df.drop(columns=["X_0_sol", "OE_0_sol", "dX_sol", "dOE_sol", "OE_0", "X_0"])
    
    # write these columns to a latex table
    print(df.to_latex(index=True, columns=['a', 'e', 'i', 'omega', 'Omega', 'M', 'T_0_sol','dX_scalar'], float_format="{:0.2f}".format))

    # data = {}

    # df = pd.DataFrame()



    
    # print(
    #     f"Initial Condition & {OE_0[0,0]} & {OE_0[0,1]} & {OE_0[0,2]} & {OE_0[0,3]} & {OE_0[0,4]} & {OE_0[0,5]} & {T_0} \\\\ \n \\hline "
    # )

    # for k in range(len(df)):
    #     if df.index[k] == "OE_bounded":
    #         continue
    #     T_0_sol, X_0_sol = get_corrected_IC(df, k)
    #     OE_0 = cart2oe_tf(np.array([X_0_sol]), planet.mu)
    #     if df.index[k] == "cartesian":
    #         label = "Cartesian"
    #     elif df.index[k] == "OE_bounded":
    #         label = "Initial Condition"
    #     elif df.index[k] == "OE_constrained":
    #         label = "OE Constrained"
    #     elif df.index[k] == "OE":
    #         label = "Orbital Elements"

    #     print(
    #         f"{label} & {OE_0[0,0]} & {OE_0[0,1]} & {OE_0[0,2]} & {OE_0[0,3]} & {OE_0[0,4]} & {OE_0[0,5]} & {T_0_sol} \\\\ \n \\hline "
    #     )
    # plt.show()


if __name__ == "__main__":
    main()
