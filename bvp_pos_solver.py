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
from GravNN.Support.transformations import invert_projection, cart2sph

def bc(ya, yb, p=None):
    periodic_res = yb - ya 
    bc_res = np.hstack((periodic_res))#, ic_res))
    return bc_res

def solve_bvp_pos_problem(T, state, lpe, t_eval=None):

    def fun(x,y,p=None):
        "Return the first-order system"
        # print(x)
        R = y[0:3]
        V = y[3:6]

        r = np.linalg.norm(R, axis=0)
        a_pm_sph = np.vstack((-lpe.mu/r**2, np.zeros((len(r),)),np.zeros((len(r),)))).T
        r_sph = cart2sph(R.T)

        a_pm_xyz = invert_projection(r_sph, a_pm_sph)

        a = lpe.model.generate_acceleration(R.T).numpy()
        dxdt = np.vstack((V, (a_pm_xyz - a).T))
        return dxdt.reshape((6,-1))
    
    t_mesh = np.linspace(0, T, 100)
    states = solve_ivp_pos_problem(T, state, lpe, t_eval=t_mesh)
    y_guess = states

    p = None # Initial guess for unknown parameters
    results = solve_bvp(fun, bc, t_mesh, y_guess, p=p, verbose=2, tol=1e-3) # returns a polynomial solution which can plotted
    print(results.y[:,0])
    plot_pos_results(results, T, lpe, obj_file=Eros().model_potatok)


def main():
    planet = Eros()
    df = pd.read_pickle("Data/Dataframes/eros_grav_model_minus_pm.data")
    config, model  = load_config_and_model(df.iloc[-1]['id'], df)
    # trad_OE = np.array([[planet.radius*2, 0.1, np.pi/4, np.pi/3, np.pi/3, np.pi/3]]) # passes through asteroid
    # trad_OE = np.array([[planet.radius*1.6, 0.1, np.pi/4, np.pi/3, np.pi/3, np.pi/3]]) # passes through asteroid
    trad_OE = np.array([[planet.radius*3, 0.1, np.pi/3, np.pi/3, np.pi/3, np.pi/3]]) # Doesn't pass through the asteroid
    # trad_OE = np.array([[planet.radius*2.5, 0.01, np.pi/3, np.pi/3, np.pi/3, np.pi/3]]) # passes through asteroid
    # trad_OE = np.array([[planet.radius*2.5, 0.01, np.pi/3, 0.0, 0.0, np.pi/3]]) # Works Nicely
    n = np.sqrt(planet.mu/trad_OE[0,0]**3)
    T = 2*np.pi/n 
    
    state = np.hstack(oe2cart_tf(trad_OE, planet.mu))
    
    OE = trad_OE
    element_set = "traditional"

    lpe = LPE(model, config, planet.mu, OE, element_set=element_set)
    solve_bvp_pos_problem(T, state, lpe)


if __name__ == "__main__":
    main()