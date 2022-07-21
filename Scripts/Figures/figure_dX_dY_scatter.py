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


vis = VisualizationBase()
vis.fig_size = vis.tri_page
vis.fig_size = (2.1, 2.1*1.25)
plt.rc('font', size=8)
plt.rc('axes', labelsize=8)
plt.rc('axes', linewidth=1)
plt.rc('lines', markersize=1)
plt.rc('xtick.major', size=4)
plt.rc('xtick.minor', size=4)
plt.rc('ytick', labelsize=8)
plt.rc('xtick', labelsize=8)
plt.rc('xtick', direction='in')
plt.rc('ytick', direction='in')
plt.rc('axes', labelpad=1)
plt.rc('lines', markersize=6)
# plt.rc('xaxis', labellocation='right')
# plt.rc('yaxis', labellocation='top')



def dr_coordinates(df, **kwargs):
    X_0 = np.array(df["X_0"])
    dX_0 = np.array(df["dX_0"])
    dX_0_sol = np.array(df["dX_sol"])

    l_star = kwargs.get("l_star")
    t_star = kwargs.get("t_star")

    r_0 = np.linalg.norm(X_0[0:3])
    dr_0 = np.linalg.norm(dX_0[0:3])
    dr_0_sol = np.linalg.norm(dX_0_sol[0:3])

    X_0 = dr_0
    X_f = dr_0_sol

    Y_0 = dr_0/l_star
    Y_f = dr_0_sol/l_star

    return X_0, X_f, Y_0, Y_f

def dv_coordinates(df, **kwargs):
    X_0 = np.array(df["X_0"])
    dX_0 = np.array(df["dX_0"])
    dX_0_sol = np.array(df["dX_sol"])

    l_star = kwargs.get("l_star")
    t_star = kwargs.get("t_star")

    v_0 = np.linalg.norm(X_0[3:6])
    dv_0 = np.linalg.norm(dX_0[3:6])
    dv_0_sol = np.linalg.norm(dX_0_sol[3:6])

    X_0 = dv_0
    X_f = dv_0_sol

    non_dim = t_star / l_star
    Y_0 = dv_0*non_dim
    Y_f = dv_0_sol*non_dim

    return X_0, X_f, Y_0, Y_f

def dOE_coordinates(df, **kwargs):

    l_star = kwargs.get("l_star")
    t_star = kwargs.get("t_star")

    planet = Eros()
    OE_tilde_0 = df['OE_0'].reshape((-1,6))
    lpe = LPE_Traditional(None, planet.mu, 
                                l_star=l_star, 
                                t_star=t_star, 
                                m_star=1.0,
                                theta_star=2*np.pi)

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

    X_0, X_f, Y_0, Y_f = coord_fcn(df, **kwargs)

    plt.scatter(X_0, Y_0, marker=options['start_marker'], c=options['start_color'])
    plt.scatter(X_f, Y_f, marker=options['end_marker'], c=options['end_color'])
    plt.plot([X_0, X_f], [Y_0, Y_f],
                color=options['arrow_color'],
                alpha=options['arrow_opacity'],
                linewidth=1)

    plt.xlabel(options['x_label'], c=options['x_label_color'])
    plt.ylabel(options['y_label'], c=options['y_label_color'])
    # plt.title(f"{options['x_label']} vs {options['y_label']}")
    if options['log_scale']:
        plt.xscale('log')
        plt.yscale('log')

def plot_results(df, coord_fcn, **kwargs):
    color_list = ['blue', 'red', 'green']
    for k in range(len(df)):
        if k < 3:
            options = {"arrow_color" : color_list[k],
                      "start_color" : color_list[k],
                      "end_color" : color_list[k],
                      "l_star" : kwargs.get("l_star_vec")[k],
                      "t_star" : kwargs.get("t_star_vec")[k],
                      }
        else:
            continue
            options = {"arrow_opacity" : 0.5}
        options.update(kwargs)
        plot_scatter_with_arrow(df.iloc[k], 
                                coord_fcn,
                                **options)





####################
## Main Functions ##
####################


def plot_dr_results(df_coarse, df_fine, base_figure_name, **kwargs):
    vis.newFig()
    plot_results(df_coarse, dr_coordinates,**kwargs)
    plot_results(df_fine, dr_coordinates, 
                    start_marker='s', 
                    end_marker='*', 
                    x_label=r"$\delta \tilde{r}$ [m]",
                    y_label=r"$\delta r$ [-]", **kwargs)
    vis.save(plt.gcf(), f"{base_figure_name}_dr.pdf")

def plot_dv_results(df_coarse, df_fine, base_figure_name, **kwargs):
    vis.newFig()
    plot_results(df_coarse, dv_coordinates, **kwargs)
    plot_results(df_fine, dv_coordinates, 
                    start_marker='s', 
                    end_marker='*', 
                    x_label=r"$\delta \tilde{v}$ [m/s]",
                    y_label=r"$\delta v$ [-]", **kwargs)
    vis.save(plt.gcf(), f"{base_figure_name}_dv.pdf")
    
def plot_dOE_results(df_coarse, df_fine, base_figure_name, **kwargs):

    vis.newFig()
    plot_results(df_coarse, dOE_coordinates, **kwargs)
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

    l_star_vec = np.array([np.linalg.norm(df_coarse['X_0'][i][0:3]) for i in range(len(df_coarse))])/100.0
    t_star_vec = np.array([np.linalg.norm(df_coarse['X_0'][i][3:6])*10000/l_star_vec[i] for i in range(len(df_coarse))])
    plot_dr_results(df_coarse, df_fine, base_figure_name, y_label_color='red', l_star_vec=l_star_vec, t_star_vec=t_star_vec)
    plot_dv_results(df_coarse, df_fine, base_figure_name, y_label_color='red', l_star_vec=l_star_vec, t_star_vec=t_star_vec)
    plot_dOE_results(df_coarse, df_fine, base_figure_name,l_star_vec=l_star_vec, t_star_vec=t_star_vec)

    base_figure_name = os.path.basename(__file__).split('.py')[0]
    df_coarse = pd.read_pickle(directory + "coarse_orbit_solutions.data")
    df_fine = pd.read_pickle(directory + "fine_orbit_solutions.data")
    l_star_vec = np.array([np.linalg.norm(df_coarse['X_0'][i][0:3]) for i in range(len(df_coarse))])
    t_star_vec = df_coarse['T_0']
    plot_dr_results(df_coarse, df_fine, base_figure_name,l_star_vec=l_star_vec, t_star_vec=t_star_vec)
    plot_dv_results(df_coarse, df_fine, base_figure_name,l_star_vec=l_star_vec, t_star_vec=t_star_vec)
    plot_dOE_results(df_coarse, df_fine, base_figure_name, y_label_color='red',l_star_vec=l_star_vec, t_star_vec=t_star_vec)

    plt.show()

if __name__ == "__main__":
    main()
