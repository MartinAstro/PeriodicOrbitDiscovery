from GravNN.Networks.utils import configure_tensorflow
from GravNN.Networks.Model import load_config_and_model
from GravNN.CelestialBodies.Planets import Earth
from GravNN.CelestialBodies.Asteroids import Eros,Toutatis
from GravNN.Support.ProgressBar import ProgressBar
from coordinate_transforms import *
from LPE import LPE
tf = configure_tensorflow()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import hvplot.pandas
import pandas as pd
import hvplot.xarray
import xarray as xr
import panel as pn
from tqdm.notebook import tqdm
pn.extension()

from scipy.integrate import solve_bvp, solve_ivp



def solve_bvp_problem(T, OE, lpe):
    # Solve BVP
    def bc(ya, yb):
        "Orbital elements should be equal to one another"
        "Define residuals as a 6x1 vector"
        return yb - ya

    def fun(x,y,p=None):
        "Return the first-order system"
        dxdt = np.array([v.numpy() for v in lpe(y.T).values()])
        return dxdt.reshape((6,-1))

    t_mesh = np.linspace(0, T, 100)

    y_guess = [] # will be equal to (6, t_mesh.size)
    for _ in range(0,len(t_mesh)) : y_guess.append([OE]) 
    y_guess = np.array(y_guess).squeeze().T

    results = solve_bvp(fun, bc, t_mesh, y_guess, verbose=2) # returns a polynomial solution which can plotted
    t_plot = np.linspace(0, T, 1000)
    y_plot = results.sol(t_plot)
    a_plot, e_plot, i_plot = y_plot[0:3]

    plt.figure()
    plt.subplot(3,1,1)
    plt.plot(t_plot, a_plot) # semi major axis
    plt.subplot(3,1,2)
    plt.plot(t_plot, e_plot)
    plt.subplot(3,1,3)
    plt.plot(t_plot, i_plot)
    plt.show()

def main():
    planet = Eros()

    df = pd.read_pickle("Data/Dataframes/eros_grav_model_minus_pm.data")

    config, model  = load_config_and_model(df.iloc[-1]['id'], df)

    OE = np.array([[planet.radius*2, 0.1, np.pi/4, np.pi/3, np.pi/3, np.pi/3]]).astype(np.float32)
    n = np.sqrt(planet.mu/OE[0,0]**3)
    T = 2*np.pi/n
    lpe = LPE(model, config, planet.mu, OE, element_set='traditional')
    solve_bvp_problem(T, OE, lpe)


if __name__ == "__main__":
    main()