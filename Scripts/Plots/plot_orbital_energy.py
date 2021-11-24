import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.polynomial import poly
from GravNN.Networks.Model import load_config_and_model
from GravNN.Visualization.VisualizationBase import VisualizationBase
from GravNN.CelestialBodies.Asteroids import Eros
import pathlib
import pickle
from FrozenOrbits.ivp import solve_ivp_pos_problem
import OrbitalElements.orbitalPlotting as op
import pandas as pd
from FrozenOrbits.LPE import LPE
from FrozenOrbits.gravity_models import *
def collect_plotting_data(solutions):
    periods = []
    ICs = []
    i = 1
    for solution in solutions:
        periods.append(solution['T_exact'])
        ICs.append(solution['IC'])
    return periods, ICs


def main():
    file_path = pathlib.Path(__file__).parent.absolute()
    file = file_path.as_posix() +  "/Data/BVP_Solutions/PINN_trajectories.data"
    with open(file, 'rb') as f:
        valid_solutions = pickle.load(f)
        unique_solutions = pickle.load(f)
    
    planet = Eros()
    df = pd.read_pickle("Data/Dataframes/eros_grav_model_minus_pm.data")
    config, model  = load_config_and_model(df.iloc[-1]['id'], df)
    lpe = LPE(model, config, planet.mu, element_set="traditional")

    model = simpleGravityModel(planet.mu)
    lpe = LPE(model, config, planet.mu, element_set="traditional")

    model = polyhedralGravityModel(planet, planet.model_potatok)
    lpe = LPE(model, config, planet.mu, element_set="traditional")

    periods, ICs = collect_plotting_data(unique_solutions)


    vis = VisualizationBase()
    # vis.newFig()
    start = 20 # 0
    for i in range(start, len(periods)):
        sol = solve_ivp_pos_problem(periods[i], ICs[i], lpe, t_eval=None, events=None, args=(ICs[i],), atol=1e-12, rtol=1e-14)
        rVec = sol.y[0:3, :]
        vVec = sol.y[3:6, :]
        hVec = np.cross(rVec.T, vVec.T)
        h_norm = np.linalg.norm(hVec, axis=1)
        plt.plot(np.linspace(0, 1, len(h_norm)), h_norm, label='orbit ' + str(i))
    plt.show()


if __name__ == "__main__":
    main()
