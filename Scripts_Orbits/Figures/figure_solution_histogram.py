import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import FrozenOrbits
from FrozenOrbits.bvp import *
from FrozenOrbits.constraints import *
from FrozenOrbits.LPE import *
from FrozenOrbits.visualization import *
from Scripts_Orbits.BVP.initial_conditions import *


def plot_solution_histogram(error, **kwargs):
    range = kwargs.get("range", [0, np.max(error)])
    kwargs.update({"range": range})
    # range=None
    plt.hist(
        error,
        bins=np.logspace(np.log10(1), np.log10(10**6), 50),
        **kwargs,
    )
    return range


def main():
    directory = os.path.dirname(FrozenOrbits.__file__)
    cart_df = pd.read_pickle(directory + "/../Data/MC/orbit_solutions_cart.data")
    mil_df = pd.read_pickle(directory + "/../Data/MC/orbit_solutions_mil.data")
    trad_df = pd.read_pickle(directory + "/../Data/MC/orbit_solutions_trad.data")

    df = pd.concat((cart_df, mil_df, trad_df), axis=0)
    df = df[df['valid'] == True]

    # compute percent error between initial and final state
    def compute_percent_error(row, key_prefix):
        if key_prefix == "OE":
            dX = row['dOE_sol']
            X_0 = row['OE_0_sol']
        else:
            dX = row['dX_sol']
            X_0 = row['X_0_sol']

        dX_percent = np.abs(dX / X_0) * 100
        dX_percent_mag = np.average(dX_percent)
        return dX_percent_mag

    # Save into the dataframe
    df["OE_percent"] = df.apply(lambda row: compute_percent_error(row, "OE"), axis=1)
    df["X_percent"] = df.apply(lambda row: compute_percent_error(row, "X"), axis=1)

    # filter the data based on the solver_key
    cart_df = df[df["solver_key"] == "cart"]
    mil_df = df[df["solver_key"] == "mil"]
    equi_df = df[df["solver_key"] == "equi"]
    trad_df = df[df["solver_key"] == "trad"]

    vis = VisualizationBase()
    vis.newFig()

    plot_solution_histogram(
        cart_df["X_percent"],
        color="orange",
        label="Cartesian Shooting Method",
        alpha=0.8,
    )
    plot_solution_histogram(
        trad_df["OE_percent"],
        color="blue",
        label="Traditional OE Shooting Method",
        alpha=0.8,
    )
    plot_solution_histogram(
        mil_df["OE_percent"],
        color="red",
        label="Milankovitch OE Shooting Method",
        alpha=0.8,
    )
    # plot_solution_histogram(
    #     equi_df["OE_percent"],
    #     color="red",
    #     label="Equinoctial OE Shooting Method",
    #     alpha=0.8,
    # )
    plt.xscale("log")
    plt.legend()
    plt.xlabel("Cartesian State Error after 10 Orbits [m]")
    plt.ylabel("\# of Solutions")
    os.makedirs(directory + "/../Plots/", exist_ok=True)
    vis.save(plt.gcf(), directory + "/../Plots/error_histogram.pdf")
    plt.show()


if __name__ == "__main__":
    main()
