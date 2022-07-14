from asyncio import gather
import os
from FrozenOrbits.bvp import *
import FrozenOrbits
import GravNN
import matplotlib.pyplot as plt
import numpy as np
from FrozenOrbits.gravity_models import (pinnGravityModel)
from FrozenOrbits.LPE import *
from FrozenOrbits.visualization import *
from GravNN.CelestialBodies.Asteroids import Eros




def dr_coordinates(df):
    dX_0 = np.array(df["dX_0"])
    dX_0_sol = np.array(df["dX_sol"])

    r_0 = np.linalg.norm(dX_0[0:3])
    r_0_sol = np.linalg.norm(dX_0_sol[0:3])

    X_0 = r_0
    X_f = r_0_sol

    l_star = np.linalg.norm(r_0)
    Y_0 = r_0/l_star
    Y_f = r_0_sol/l_star

    return X_0, X_f, Y_0, Y_f

def dv_coordinates(df):
    dX_0 = np.array(df["dX_0"])
    dX_0_sol = np.array(df["dX_sol"])

    v_0 = np.linalg.norm(dX_0[3:6])
    v_0_sol = np.linalg.norm(dX_0_sol[3:6])

    X_0 = v_0
    X_f = v_0_sol

    l_star = np.linalg.norm(v_0)
    Y_0 = v_0/l_star
    Y_f = v_0_sol/l_star

    return X_0, X_f, Y_0, Y_f

def dOE_coordinates(df):

    planet = Eros()
    OE_tilde_0 = df['OE_0'].reshape((-1,6))
    lpe = LPE_Traditional(None, planet.mu, 
                                l_star=OE_tilde_0[0,0]/1.0, 
                                t_star=df['T_0'], 
                                m_star=1.0)

    dOE_tilde_0 = df["dOE_0"].reshape((-1,6))
    dOE_tilde_0_sol = df["dOE_sol"].reshape((-1,6))

    dOE_0 = lpe.non_dimensionalize_state(dOE_tilde_0)
    dOE_0_sol = lpe.non_dimensionalize_state(dOE_tilde_0_sol)


    # Get scalar magnitudes
    dOE_tilde_0_mag = np.linalg.norm(dOE_tilde_0)
    dOE_tilde_0_sol_mag = np.linalg.norm(dOE_tilde_0_sol)
    
    dOE_0_mag = np.linalg.norm(dOE_0)
    dOE_0_sol_mag = np.linalg.norm(dOE_0_sol)

    X_0 = dOE_tilde_0_mag
    X_f = dOE_tilde_0_sol_mag

    Y_0 = dOE_0_mag
    Y_f = dOE_0_sol_mag

    return X_0, X_f, Y_0, Y_f

def plot_scatter_with_arrow(df, coord_fcn, **kwargs):
    options = {
        "start_color" : 'black',
        "end_color" : 'black',
        "start_marker" : 'o',
        "end_marker" : 's',
        "arrow_color" : 'black',
        "arrow_opacity" : 1.0,
        "log_scale" : True,
        "x_label" : "X",
        "y_label" : "Y",
        "x_label_color" : 'black',
        "y_label_color" : 'black',
    }
    options.update(kwargs)

    X_0, X_f, Y_0, Y_f = coord_fcn(df)

    plt.scatter(X_0, Y_0, marker=options['start_marker'], c=options['start_color'])
    plt.scatter(X_f, Y_f, marker=options['end_marker'], c=options['end_color'])
    plt.plot([X_0, X_f], [Y_0, Y_f],
                color=options['arrow_color'],
                alpha=options['arrow_opacity'])

    plt.xlabel(options['x_label'], c=options['x_label_color'])
    plt.ylabel(options['y_label'], c=options['y_label_color'])
    if options['log_scale']:
        plt.xscale('log')
        plt.yscale('log')

def plot_results(df, coord_fcn, **kwargs):
    color_list = ['blue', 'red', 'green']
    for k in range(len(df)):
        if k < 3:
            options = {"arrow_color" : color_list[k],
                      "start_color" : color_list[k],
                      "end_color" : color_list[k]}
        else:
            options = {"arrow_opacity" : 0.5}
        options.update(kwargs)
        plot_scatter_with_arrow(df.iloc[k], 
                                coord_fcn,
                                # start_color='red', 
                                # end_color='blue',
                                **options)

def plot_dr_results(df_coarse, df_fine, base_figure_name, **kwargs):
    vis = VisualizationBase()
    vis.newFig()
    plot_results(df_coarse, dr_coordinates)
    plot_results(df_fine, dr_coordinates, 
                    start_marker='s', 
                    end_marker='*', 
                    x_label=r"$\delta \tilde{r}$ [m]",
                    y_label=r"$\delta r$ [-]", **kwargs)
    vis.save(plt.gcf(), f"{base_figure_name}_dr.pdf")

def plot_dv_results(df_coarse, df_fine, base_figure_name, **kwargs):
    vis = VisualizationBase()
    vis.newFig()
    plot_results(df_coarse, dv_coordinates)
    plot_results(df_fine, dv_coordinates, 
                    start_marker='s', 
                    end_marker='*', 
                    x_label=r"$\delta \tilde{v}$ [m/s]",
                    y_label=r"$\delta v$ [-]", **kwargs)
    vis.save(plt.gcf(), f"{base_figure_name}_dv.pdf")
    
def plot_dOE_results(df_coarse, df_fine, base_figure_name, **kwargs):
    vis = VisualizationBase()
    vis.newFig()
    plot_results(df_coarse, dOE_coordinates)
    plot_results(df_fine, dOE_coordinates, 
                    start_marker='s', 
                    end_marker='*', 
                    x_label=r"$\delta \tilde{\oe}$ [-]",
                    y_label=r"$\delta \oe$ [-]", **kwargs)

    vis.save(plt.gcf(), f"{base_figure_name}_dOE.pdf")


def main():
    directory =  os.path.dirname(FrozenOrbits.__file__)+ "/Data/"

    base_figure_name = os.path.basename(__file__).split('.py')[0] + "_cartesian"
    df_coarse = pd.read_pickle(directory + "cartesian_coarse_orbit_solutions.data")
    df_fine = pd.read_pickle(directory + "cartesian_fine_orbit_solutions.data")
    plot_dr_results(df_coarse, df_fine, base_figure_name, y_label_color='red')
    plot_dv_results(df_coarse, df_fine, base_figure_name, y_label_color='red')
    plot_dOE_results(df_coarse, df_fine, base_figure_name)

    base_figure_name = os.path.basename(__file__).split('.py')[0]
    df_coarse = pd.read_pickle(directory + "coarse_orbit_solutions.data")
    df_fine = pd.read_pickle(directory + "fine_orbit_solutions.data")
    plot_dr_results(df_coarse, df_fine, base_figure_name)
    plot_dv_results(df_coarse, df_fine, base_figure_name)
    plot_dOE_results(df_coarse, df_fine, base_figure_name, y_label_color='red')

    plt.show()

if __name__ == "__main__":
    main()
