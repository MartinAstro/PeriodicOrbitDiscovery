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

import pandas as pd
import xarray as xr
import panel as pn
from tqdm.notebook import tqdm
import OrbitalElements.orbitalPlotting as op
from scipy.integrate import solve_ivp
pn.extension()



def solve_ivp_problem(T, OE, lpe, t_eval=None):
    def fun(x,y,p=None):
        "Return the first-order system"
        print(x)
        results = lpe(y.reshape((1,-1)))
        dxdt = np.array([v.numpy() for v in results.values()])
        return dxdt.reshape((-1,))
    
    sol = solve_ivp(fun, [0, T], OE.reshape((-1,)), t_eval=t_eval, atol=1e-8, rtol=1e-10) #atol=1e-6, rtol=1e-8)

    t_plot = sol.t
    y_plot = sol.y
    op.plot_OE(t_plot, y_plot, lpe.element_set)

    r_plot = []
    for y in y_plot.T:
        R,V = oe2cart_tf(np.array([y.T]), lpe.mu)
        r_plot.append(R)

    op.plot_orbit_3d(np.array(r_plot).T.squeeze())

    plt.show()

def main():
    # planet = Earth()
    # OE = np.array([[planet.radius+ 200000, 0.01, np.pi/4, 0.1, 0.1, 0.1]])
    # df = pd.read_pickle("Data/Dataframes/earth_grav_model.data")

    planet = Eros()
    OE = np.array([[planet.radius*2, 0.01, np.pi/4, 0.1, 0.1, 0.1]]) 
    OE = np.array([[planet.radius*2, 0.01, np.pi/4, np.pi/3, np.pi/3, np.pi/3]]).astype(np.float32)
    OE = np.array([[planet.radius*2, 0.1, np.pi/4, np.pi/3, np.pi/3, np.pi/3]]).astype(np.float32)

    df = pd.read_pickle("Data/Dataframes/eros_grav_model_minus_pm.data")

    config, model  = load_config_and_model(df.iloc[-1]['id'], df)

    # T = 0.1
    # N = 20
    # t_eval = np.linspace(0,T, N)

    T = 100000
    t_eval = None
    
    lpe = LPE(model, config, planet.mu, OE, element_set='traditional')
    solve_ivp_problem(T, OE, lpe)

    # lpe = LPE(model, config, planet.mu, OE, element_set='Delaunay')
    # DelaunayOE = oe2delaunay_tf(OE, planet.mu).numpy()
    # solve_ivp_problem(T, DelaunayOE, lpe) 

    # Milankovitch actually requires the original OE as input
    # lpe = LPE(model, config, planet.mu, OE, element_set='Milankovitch')
    # MilankovitchOE = oe2milankovitch_tf(OE, planet.mu).numpy()
    # solve_ivp_problem(T, MilankovitchOE, lpe) 

    # lpe = LPE(model, config, planet.mu, OE, element_set='Equinoctial')
    # equinoctialOE = oe2equinoctial_tf(OE).numpy()
    # solve_ivp_problem(T, equinoctialOE, lpe, t_eval) 


if __name__ == "__main__":
    main()