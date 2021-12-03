import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.polynomial import poly
import FrozenOrbits
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
from scipy.integrate import solve_ivp
import os
import FrozenOrbits


def collect_plotting_data(solutions):
    periods = []
    ICs = []
    i = 1
    for solution in solutions:
        periods.append(solution['T_exact'])
        ICs.append(solution['IC'])
    return periods, ICs


def main():
    path = os.path.dirname(FrozenOrbits.__file__)
    file = path +  "/../Data/BVP_Solutions/PINN_trajectories.data"
    with open(file, 'rb') as f:
        valid_solutions = pickle.load(f)
        unique_solutions = pickle.load(f)
    
    periods, ICs = collect_plotting_data(unique_solutions)


    planet = Eros()
    df = pd.read_pickle("Data/Dataframes/eros_grav_model_minus_pm.data")
    config, model  = load_config_and_model(df.iloc[-1]['id'], df)
    lpe = LPE(model, config, planet.mu, element_set="traditional")

    def pinn_fun(x,y,IC=None):
        "Return the first-order system"
        print(x)
        R, V = y[0:3], y[3:6]
        r = np.linalg.norm(R)
        a_pm_sph = np.array([[-lpe.mu/r**2, 0.0, 0.0]])
        r_sph = cart2sph(R.reshape((1,3)))
        a_pm_xyz = invert_projection(r_sph, a_pm_sph).reshape((3,))
        a = lpe.model.generate_acceleration(R.reshape((1,3))).numpy()
        dxdt = np.hstack((V, a_pm_xyz - a.reshape((3,))))
        return dxdt.reshape((6,))
    fun = pinn_fun

    # model = simpleGravityModel(planet.mu)
    # lpe = LPE(model, config, planet.mu, element_set="traditional")
    # def simple_fun(x,y,IC=None):
    #     "Return the first-order system"
    #     R, V = y[0:3], y[3:6]
    #     a = lpe.model.generate_acceleration(R.reshape((1,3))).numpy()
    #     dxdt = np.hstack((V, a.reshape((3,))))
    #     return dxdt.reshape((6,))
    # fun = simple_fun


    model = polyhedralGravityModel(planet, planet.model_potatok)
    lpe = LPE(model, config, planet.mu, element_set="traditional")
    def poly_fun(x,y,IC=None):
        "Return the first-order system"
        R, V = y[0:3], y[3:6]
        a = lpe.model.generate_acceleration(R.reshape((1,3))).numpy()
        dxdt = np.hstack((V, a.reshape((3,))))
        return dxdt.reshape((6,))
    fun = poly_fun



    vis = VisualizationBase()
    # vis.newFig()
    start = 20 # 0
    for i in range(0,1):#start, len(periods)):
        print("Period: " + str(i) + " " + str(periods[i]))
        sol = solve_ivp(fun, [0, periods[i]], ICs[i].reshape((-1,)), t_eval=None, events=None, atol=1e-8, rtol=1e-10, args=None)

        # sol = solve_ivp_pos_problem(periods[i], ICs[i], lpe, t_eval=None, events=None, args=(ICs[i],), atol=1e-12, rtol=1e-14)
        rVec = sol.y[0:3, :]
        vVec = sol.y[3:6, :]
        hVec = np.cross(rVec.T, vVec.T)
        h_norm = np.linalg.norm(hVec, axis=1)
        plt.plot(np.linspace(0, 1, len(h_norm)), h_norm, label='orbit ' + str(i))
    plt.show()


if __name__ == "__main__":
    main()
