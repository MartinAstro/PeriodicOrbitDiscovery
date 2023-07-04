import os

import GravNN
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import FrozenOrbits
from FrozenOrbits.bvp import *
from FrozenOrbits.constraints import *
from FrozenOrbits.gravity_models import pinnGravityModel, polyhedralGravityModel
from FrozenOrbits.LPE import *
from FrozenOrbits.visualization import *
from Scripts_Orbits.BVP.initial_conditions import *


def get_state(sol):
    state = {
        "t": sol.t,
        "r": sol.y[0:3, :].T,
        "v": sol.y[3:6, :].T,
    }
    return state


def get_error(state_1, state_2):
    pos_diff = state_1["r"] - state_2["r"]
    vel_diff = state_1["v"] - state_2["v"]
    acc_diff = state_1["a"] - state_2["a"]

    pos_diff_mag = np.linalg.norm(pos_diff, axis=1)
    vel_diff_mag = np.linalg.norm(vel_diff, axis=1)
    acc_diff_mag = np.linalg.norm(acc_diff, axis=1)

    error = {
        "t": state_1["t"],
        "pos": pos_diff,
        "vel": vel_diff,
        "acc": acc_diff,
        "pos_mag": pos_diff_mag,
        "vel_mag": vel_diff_mag,
        "acc_mag": acc_diff_mag,
    }
    return error


def plot_state_error(error, linestyle, label, color):
    plt.subplot(3, 1, 1)
    plt.plot(
        error["t"],
        error["pos_mag"],
        linewidth=1,
        linestyle=linestyle,
        label=label,
        color=color,
    )
    plt.ylabel(r"$\delta x$ [m]", fontsize=6)
    plt.gca().set_xticklabels([])
    plt.gca().yaxis.set_major_formatter(MathTextSciFormatter("%1.0e"))
    plt.gca().tick_params(labelsize=6)
    plt.yscale("log")
    plt.grid(which="both", linestyle="--", linewidth=0.1, color=".25", zorder=-10)
    plt.legend()

    # plt.figure(2)
    plt.subplot(3, 1, 2)
    plt.plot(
        error["t"],
        error["vel_mag"],
        linewidth=1,
        linestyle=linestyle,
        label=label,
        color=color,
    )
    plt.ylabel(r"$\delta v$ [m/s]", fontsize=6)
    plt.gca().set_xticklabels([])
    plt.gca().yaxis.set_major_formatter(MathTextSciFormatter("%1.0e"))
    plt.gca().tick_params(labelsize=6)
    plt.yscale("log")
    plt.grid(which="both", linestyle="--", linewidth=0.1, color=".25", zorder=-10)

    plt.subplot(3, 1, 3)
    plt.plot(
        error["t"],
        error["acc_mag"],
        linewidth=1,
        linestyle=linestyle,
        label=label,
        color=color,
    )
    plt.ylabel(r"$\delta a$ [m/s$^2$]", fontsize=6)
    plt.xlabel("Time [s]", fontsize=6)
    plt.gca().xaxis.set_major_formatter(MathTextSciFormatter("%1.1e"))
    plt.gca().yaxis.set_major_formatter(MathTextSciFormatter("%1.0e"))
    plt.gca().tick_params(labelsize=6)
    plt.xticks(rotation=30)
    plt.yscale("log")
    plt.grid(which="both", linestyle="--", linewidth=0.1, color=".25", zorder=-10)
    plt.tight_layout()


def main():
    pinn_model = pinnGravityModel(
        os.path.dirname(GravNN.__file__) + "/../Data/Dataframes/eros_poly_061523.data",
    )
    planet = pinn_model.config["planet"][0]
    poly_8_model = polyhedralGravityModel(planet, planet.obj_8k)
    poly_200_model = polyhedralGravityModel(planet, planet.obj_200k)

    directory = os.path.dirname(FrozenOrbits.__file__) + "/Data/"
    df = pd.read_pickle(directory + "propagation_time_error.data")

    vis = VisualizationBase(
        save_directory=os.path.dirname(FrozenOrbits.__file__) + "/../Plots/",
    )
    # (width, height)
    vis.fig_size = (vis.w_half, vis.h_half)
    vis.newFig()  # position
    vis.new3DFig()  # 3d cartesian

    semi_list = []
    dr_list = []
    dv_list = []
    dr_8k_list = []
    dv_8k_list = []
    T_8_list = []
    T_200_list = []
    T_PINN_list = []
    color_list = []
    colors = ["red", "blue", "green"]
    for k in range(0, 3):  # len(df)):
        pinn_sol = df["pinn_sol"][k]
        poly_8_sol = df["poly_sol"][k]
        poly_200_sol = df["poly_200_sol"][k]

        pinn_state = get_state(pinn_sol)
        poly_8_state = get_state(poly_8_sol)
        poly_200_state = get_state(poly_200_sol)

        # add accelerations to state
        pinn_pos = pinn_state["r"]
        pinn_state["a"] = pinn_model.compute_acceleration(pinn_pos)
        poly_8_state["a"] = poly_8_model.compute_acceleration(pinn_pos)
        poly_200_state["a"] = poly_200_model.compute_acceleration(pinn_pos)

        pinn_error = get_error(pinn_state, poly_200_state)
        poly_8_error = get_error(poly_8_state, poly_200_state)

        plt.figure(1)
        plot_state_error(pinn_error, "-", "PINN", colors[k])
        plot_state_error(poly_8_error, "--", "Polyhedron (8k)", colors[k])
        plt.subplot(3, 1, 1)
        plt.legend(
            [
                plt.Line2D([0], [0], color="black", linestyle="-"),
                plt.Line2D([0], [0], color="black", linestyle="--"),
            ],
            ["PINN", "Poly (8k)"],
            loc="upper right",
            fontsize=6,
        )

        plt.figure(2)
        plot_cartesian_state_3d(
            pinn_sol.y.T,
            planet.obj_8k,
            solid_color=colors[k],
            new_fig=False,
            plot_start_point=False,
            linewidth=1,
        )
        plot_cartesian_state_3d(
            poly_8_sol.y.T,
            planet.obj_8k,
            solid_color=colors[k],
            linestyle=".",
            new_fig=False,
            plot_start_point=False,
            linewidth=1,
        )
        plot_cartesian_state_3d(
            poly_200_sol.y.T,
            planet.obj_8k,
            solid_color=colors[k],
            linestyle="--",
            new_fig=False,
            plot_start_point=False,
            linewidth=1,
        )
        plt.gca().tick_params(labelsize=6)
        # plt.gca().xaxis.set_major_locator(plt.MaxNLocator(4))
        # plt.gca().yaxis.set_major_locator(plt.MaxNLocator(4))
        # plt.gca().zaxis.set_major_locator(plt.MaxNLocator(4))
        plt.gca().set_xticklabels([])
        plt.gca().set_yticklabels([])
        plt.gca().set_zticklabels([])
        plt.gca().set_xlabel("")
        plt.gca().set_ylabel("")

        dr = np.linalg.norm(df["Xf_pinn"][k][0:3] - df["Xf_poly_200k"][k][0:3])
        dv = np.linalg.norm(df["Xf_pinn"][k][3:6] - df["Xf_poly_200k"][k][3:6])
        dr_8k = np.linalg.norm(df["Xf_poly_8k"][k][0:3] - df["Xf_poly_200k"][k][0:3])
        dv_8k = np.linalg.norm(df["Xf_poly_8k"][k][3:6] - df["Xf_poly_200k"][k][3:6])

        semi_list.append(df["semi"][k])
        T_200_list.append(df["dt_poly_200"][k])
        T_8_list.append(df["dt_poly"][k])
        T_PINN_list.append(df["dt_pinn"][k])
        dr_list.append(dr)
        dv_list.append(dv)
        dr_8k_list.append(dr_8k)
        dv_8k_list.append(dv_8k)
        color_list.append(colors[k])

    vis.save(plt.figure(1), os.path.basename(__file__).split(".py")[0] + "_dr.pdf")
    # vis.save(plt.figure(2), os.path.basename(__file__).split('.py')[0] + '_dv.pdf')
    # vis.save(plt.figure(3), os.path.basename(__file__).split('.py')[0] + '_da.pdf')
    vis.save(plt.figure(2), os.path.basename(__file__).split(".py")[0] + "_3d.pdf")
    print("\\hline")
    print(
        "Model & $\\delta r$ [m] & $\\delta v$ [m/s] & $ T [s] & Speedup ($T_{\\text{200k}} / T_{\\text{model}}) \\\\",
    )
    print("\\hline")
    for k in range(len(dr_list)):
        # format a string with finite precition
        print(color_list[k].capitalize())
        print(
            "%s & %.1f & %.1E & %.1f & %.1f \\\\"
            % (
                "Poly 200",
                0.0,
                0.0,
                T_200_list[k],
                T_200_list[k] / T_200_list[k],
            ),
        )
        print(
            "%s & %.1f & %.1E & %.1f & %.1f \\\\"
            % (
                "Poly 8k",
                dr_8k_list[k],
                dv_8k_list[k],
                T_8_list[k],
                T_200_list[k] / T_8_list[k],
            ),
        )
        print(
            "%s & %.1f & %.1E & %.1f & %.1f \\\\"
            % (
                "PINN",
                dr_list[k],
                dv_list[k],
                T_PINN_list[k],
                T_200_list[k] / T_PINN_list[k],
            ),
        )

        # print(
        #     f"{color_list[k].capitalize()} & {dr_list[k]} & {dv_list[k]} & {semi_list[k]} & {T_200_list[k]} & {T_8_list[k]} & {T_PINN_list[k]} & {T_8_list[k]/T_PINN_list[k]} \\\\",
        # )
    print("\\hline")
    plt.show()


if __name__ == "__main__":
    main()
