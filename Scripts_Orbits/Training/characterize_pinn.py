import os

import GravNN
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from GravNN.Analysis.PlanesExperiment import PlanesExperiment
from GravNN.Networks.Model import load_config_and_model
from GravNN.Visualization.PlanesVisualizer import PlanesVisualizer


def main():
    gravnn_dir = os.path.dirname(GravNN.__file__) + "/../"
    # df = pd.read_pickle(gravnn_dir + "Data/Dataframes/eros_poly_061523.data")
    # df = pd.read_pickle(gravnn_dir + "Data/Dataframes/eros_poly_070523.data")
    # df = pd.read_pickle(gravnn_dir + "Data/Dataframes/eros_poly_071023.data")
    df = pd.read_pickle(gravnn_dir + "Data/Dataframes/eros_poly_071123.data")
    model_id = df["id"].values[-1]
    config, model = load_config_and_model(model_id, df)

    planet = config["planet"][0]
    points = 100
    radius_bounds = [-5 * planet.radius, 5 * planet.radius]
    planes_exp = PlanesExperiment(
        model,
        config,
        radius_bounds,
        points,
        remove_error=True,
    )
    planes_exp.run()
    print(np.nanmean(planes_exp.percent_error_acc))
    print(np.nanmax(planes_exp.percent_error_acc))

    vis = PlanesVisualizer(planes_exp)
    vis.plot(percent_max=100, annotate_stats=True, log=True, z_min=1E-2)

    plt.show()


if __name__ == "__main__":
    main()
