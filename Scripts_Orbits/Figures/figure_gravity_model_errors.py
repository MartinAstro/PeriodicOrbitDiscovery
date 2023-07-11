import os

import GravNN
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from GravNN.Analysis.PlanesExperiment import PlanesExperiment
from GravNN.GravityModels.Polyhedral import Polyhedral
from GravNN.Networks.Model import load_config_and_model
from GravNN.Visualization.PlanesVisualizer import PlanesVisualizer
from FrozenOrbits.gravity_models import pinnGravityModel, polyhedralGravityModel

import FrozenOrbits


class PlanesVisualizerPanel(PlanesVisualizer):
    def __init__(self, exp):
        super().__init__(exp)
        self.fig_size = (self.w_tri * 3, self.w_tri)

    def plot(self, max=None, **kwargs):
        self.max = max

        x = self.experiment.x_test
        y = self.experiment.percent_error_acc
        # y = np.linalg.norm(self.experiment.a_test, axis=1, keepdims=True)

        gs = gridspec.GridSpec(1, 4, width_ratios=[1, 1, 1, 0.05])
        # gs.update(wspace=0.5)

        self.newFig()
        plt.subplot(gs[0])
        self.plot_plane(x, y, plane="xy", cbar=False, z_min=1e-2, **kwargs)
        plt.xticks([])
        plt.yticks([])
        plt.ylabel("")
        plt.xlabel("")

        plt.subplot(gs[1])
        self.plot_plane(x, y, plane="xz", cbar=False, z_min=1e-2, **kwargs)
        plt.xticks([])
        plt.yticks([])
        plt.ylabel("")
        plt.xlabel("")

        plt.subplot(gs[2])
        self.plot_plane(
            x,
            y,
            plane="yz",
            cbar=True,
            z_min=1e-2,
            cbar_gs=gs[3],
            **kwargs,
        )
        plt.sca(plt.gcf().axes[-2])
        plt.xticks([])
        plt.yticks([])
        plt.ylabel("")
        plt.xlabel("")


def main():
    gravnn_dir = os.path.dirname(GravNN.__file__) + "/../"
    model = pinnGravityModel(
        os.path.dirname(GravNN.__file__) + "/../Data/Dataframes/eros_poly_071123.data",
    )
    config = model.config
    planet = config["planet"][0]
    points = 100
    radius_bounds = [-5 * planet.radius, 5 * planet.radius]

    # Plot PINN Gravity Model Error
    planes_exp = PlanesExperiment(
        model,
        config,
        radius_bounds,
        points,
        remove_error=True,
    )
    planes_exp.run()

    vis = PlanesVisualizerPanel(planes_exp)
    vis.plot(max=100, annotate_stats=True, log=True)  # , contour=True)
    orbits_dir = os.path.dirname(FrozenOrbits.__file__) + "/../"
    vis.save(plt.gcf(), f"{orbits_dir}Plots/eros_pinn_planes_frozen.pdf")

    avg_error = np.nanmean(planes_exp.percent_error_acc)
    max_error = np.nanmax(planes_exp.percent_error_acc)
    print(f"PINN: Average Error: {avg_error}; \t Max Error: {max_error}")

    # Plot low fidelity polyhedral error
    model = Polyhedral(planet, planet.obj_8k)
    planes_exp = PlanesExperiment(
        model,
        config,
        radius_bounds,
        points,
        remove_error=True,
    )
    planes_exp.run()

    vis = PlanesVisualizerPanel(planes_exp)
    vis.plot(max=100, annotate_stats=True, log=True)  # , contour=True)
    vis.save(plt.gcf(), f"{orbits_dir}Plots/eros_poly_planes_frozen.pdf")

    avg_error = np.nanmean(planes_exp.percent_error_acc)
    max_error = np.nanmax(planes_exp.percent_error_acc)
    print(f"8k Poly: Average Error: {avg_error}; \t Max Error: {max_error}")

    plt.show()


if __name__ == "__main__":
    main()
