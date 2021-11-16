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
pn.extension()

from scipy.integrate import solve_ivp
import OrbitalElements.orbitalPlotting as op
import OrbitalElements.oe as oe
from GravNN.Support.transformations import cart2sph, invert_projection

def solve_ivp_problem(T, state, model, planet, t_eval=None):

    def fun(x,y,p=None):
        "Return the first-order system"
        print(x)
        R = y[0:3]
        V = y[3:6]

        r = np.linalg.norm(R)
        a_pm_sph = np.array([[-planet.mu/r**2, 0.0, 0.0]])
        r_sph = cart2sph(R.reshape((1,3)))

        a_pm_xyz = invert_projection(r_sph, a_pm_sph).reshape((3,))

        a = model.generate_acceleration(R.reshape((1,3))).numpy()
        dxdt = np.hstack((V, a_pm_xyz - a.reshape((3,))))
        return dxdt.reshape((6,))
    

    sol = solve_ivp(fun, [0, T], state.reshape((-1,)), t_eval=t_eval, atol=1e-8, rtol=1e-10)

    t_plot = sol.t
    y_plot = sol.y # [6, N]

    op.plot_orbit_3d(y_plot[0:3,:])

    mil_oe_list = []
    equi_oe_list = []
    del_oe_list = []
    oe_list = []
    for i in range(len(t_plot)):
        y = y_plot[:,i].reshape((1,6))

        trad_oe = cart2trad_tf(y, planet.mu)        
        mil_oe = cart2milankovitch_tf(y, planet.mu)
        equi_oe = oe2equinoctial_tf(trad_oe)
        del_oe = oe2delaunay_tf(trad_oe, planet.mu)

        oe_list.append(trad_oe[0,:].numpy())
        mil_oe_list.append(mil_oe[0,:].numpy())
        equi_oe_list.append(equi_oe[0,:].numpy())
        del_oe_list.append(del_oe[0,:].numpy())
    op.plot_OE(t_plot, np.array(mil_oe_list).squeeze().T, OE_set='milankovitch')
    op.plot_OE(t_plot, np.array(oe_list).squeeze().T, OE_set='traditional')
    op.plot_OE(t_plot, np.array(equi_oe_list).squeeze().T, OE_set='equinoctial')
    op.plot_OE(t_plot, np.array(del_oe_list).squeeze().T, OE_set='delaunay')

    plt.show()


def main():
    planet = Eros()
    # OE = np.array([[planet.radius*2, 0.01, np.pi/4, 0.1, 0.1, 0.1]]).astype(np.float32)
    # OE = np.array([[planet.radius*2, 0.01, np.pi/4, np.pi/3, np.pi/3, np.pi/3]]).astype(np.float32)
    OE = np.array([[planet.radius*2, 0.1, np.pi/4, np.pi/3, np.pi/3, np.pi/3]]).astype(np.float32)
    OE = np.array([[planet.radius*2, 0.01, np.pi/4, np.pi/3, np.pi/3, np.pi/3]]).astype(np.float32)
    df = pd.read_pickle("Data/Dataframes/eros_grav_model_minus_pm.data")
    # df = pd.read_pickle("Data/Dataframes/eros_grav_model.data")
    config, model  = load_config_and_model(df.iloc[-1]['id'], df)

    T = 0.1
    N = 20
    t_eval = np.linspace(0, T, N)
    
    n = np.sqrt(planet.mu/OE[0,0]**3)
    T = 2*np.pi/n 
    # T = 100000
    t_eval = None
    R, V = trad2cart_tf(OE, planet.mu)
    state = np.hstack((R.numpy(),V.numpy()))

    n = np.sqrt(planet.mu/(planet.radius*3)**3)
    T = 2*np.pi/n *2
    state = np.array([-6.36256532e+02, -4.58656092e+04,  1.31640352e+04,  3.17126984e-01, -1.12030801e+00, -3.38751010e+00])
    solve_ivp_problem(T, state, model, planet, t_eval)


if __name__ == "__main__":
    main()