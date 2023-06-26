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
    df = pd.read_pickle(directory + "/Data/orbit_solutions.data")

    # compute percent error between initial and final state
    def compute_percent_error(key):
        X_0 = df[key].y[:, 0]
        X_f = df[key].y[:, -1]
        dX = X_f - X_0
        dX_percent = np.abs(dX / X_0) * 100
        dX_percent_mag = np.average(dX_percent)
        return dX_percent_mag

    # Save into the dataframe
    df["OE_percent"] = df.apply(lambda: compute_percent_error("OE_sol"), axis=1)
    df["X_percent"] = df.apply(lambda: compute_percent_error("X_sol"), axis=1)

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
    plot_solution_histogram(
        equi_df["OE_percent"],
        color="red",
        label="Equinoctial OE Shooting Method",
        alpha=0.8,
    )
    plt.xscale("log")
    plt.legend()
    plt.xlabel("Cartesian State Error after 10 Orbits [m]")
    plt.ylabel("\# of Solutions")
    vis.save(plt.gcf(), directory + "/Plots/error_histogram.pdf")
    plt.show()


if __name__ == "__main__":
    main()
