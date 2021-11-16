from GravNN.Networks.utils import configure_tensorflow
from GravNN.Networks.Model import load_config_and_model
from GravNN.CelestialBodies.Planets import Earth
from GravNN.CelestialBodies.Asteroids import Eros,Toutatis
from GravNN.Support.ProgressBar import ProgressBar
from coordinate_transforms import *
from LPE import LPE
from boundary_conditions import *
tf = configure_tensorflow()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import OrbitalElements.orbitalPlotting as op
import hvplot.pandas
import pandas as pd
import hvplot.xarray
import xarray as xr
import panel as pn
from tqdm.notebook import tqdm
pn.extension()
from visualization import plot_OE_results, plot_pos_results
from scipy.integrate import solve_bvp, solve_ivp
from ivp import solve_ivp_problem, solve_ivp_pos_problem


def solve_bvp_problem(T, OE, lpe):
    bc = get_bc(lpe.element_set)
    def fun(x,y,p=None):
        dxdt = np.array([v.numpy() for v in lpe(y.T).values()])
        return dxdt.reshape((6,-1))

    t_mesh = np.linspace(0, T, 100)
    y_guess = []
    for _ in range(0,len(t_mesh)) : y_guess.append(OE)
    y_guess = np.array(y_guess).squeeze().T
    states = solve_ivp_pos_problem(T, np.hstack(oe2cart_tf(OE, lpe.mu, lpe.element_set)), lpe, t_eval=t_mesh)
    y_guess = []
    for i in range(len(states.T)):
        state = states[:,i].reshape((6,1))
        y_guess.append(cart2oe_tf(state.T, lpe.mu, lpe.element_set).numpy())
    y_guess = np.array(y_guess).squeeze().T
    # y_guess = solve_ivp_problem(T, OE, lpe, t_eval=t_mesh)

    p = None # Initial guess for unknown parameters
    results = solve_bvp(fun, bc, t_mesh, y_guess, p=p, verbose=2, tol=1e-1) # returns a polynomial solution which can plotted
    plot_OE_results(results, T, lpe)


def main():
    planet = Eros()
    df = pd.read_pickle("Data/Dataframes/eros_grav_model_minus_pm.data")
    config, model  = load_config_and_model(df.iloc[-1]['id'], df)

    # Very hyperbolic but close 
    # Option 1 = (5 guesses, IC without integration)
    # Option 2 = (2 guesses, IC without integration) -- doesn't work
    # Option 3 = (2 guesses, IC with integration) -- does work
    trad_OE = np.array([[planet.radius*2, 0.1, np.pi/4, np.pi/3, np.pi/3, np.pi/3]]) 


    trad_OE = np.array([[planet.radius*3, 0.4, np.pi/4, np.pi/3, np.pi/3, np.pi/3]]) 
    n = np.sqrt(planet.mu/trad_OE[0,0]**3)
    T = 2*np.pi/n 
    
    
    OE = trad_OE
    element_set = "traditional"

    # OE = oe2equinoctial_tf(trad_OE).numpy()
    # element_set = "equinoctial"

    lpe = LPE(model, config, planet.mu, OE, element_set=element_set)
    solve_bvp_problem(T, OE, lpe)


if __name__ == "__main__":
    main()