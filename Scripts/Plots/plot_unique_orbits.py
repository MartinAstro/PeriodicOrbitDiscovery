import numpy as np
import matplotlib.pyplot as plt
from GravNN.Networks.Model import load_config_and_model
from GravNN.Visualization.VisualizationBase import VisualizationBase
from GravNN.CelestialBodies.Asteroids import Eros
import pathlib
import pickle
from FrozenOrbits.ivp import solve_ivp_pos_problem
import OrbitalElements.orbitalPlotting as op
import pandas as pd
from FrozenOrbits.LPE import LPE
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

    periods, ICs = collect_plotting_data(unique_solutions)

    vis = VisualizationBase()
    start = 10 # 0
    for i in range(start, len(periods)):
        if i % 5 == 0:
            new = True
            obj_file = Eros().model_potatok if i == 0 else None
        else:
            new = False
            obj_file = None

        print(i)
        sol = solve_ivp_pos_problem(periods[i], ICs[i], lpe, t_eval=None, events=None, args=(ICs[i],))
        op.plot3d(None, sol.y, None, show=False, obj_file=obj_file, save=False, traj_cm=None, new_fig=new)
    plt.show()


if __name__ == "__main__":
    main()
