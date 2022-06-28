import numpy as np
import copy
from FrozenOrbits.ivp import *
from FrozenOrbits.coordinate_transforms import *
import matplotlib.pyplot as plt
from GravNN.Visualization.VisualizationBase import VisualizationBase
import OrbitalElements.orbitalPlotting as op
import matplotlib.animation as animation

def plot_OE_suite_from_state_sol(sol, planet):
    t_plot = sol.t
    y_plot = sol.y # [6, N]

    mil_oe_list = []
    equi_oe_list = []
    del_oe_list = []
    oe_list = []
    for i in range(len(t_plot, **kwargs)):
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

def plot_energy(sol, model):
    x = sol.y
    U = model.generate_potential(x[0:3,:].reshape((3,-1)).T).squeeze()
    T = (np.linalg.norm(x[3:6,:], axis=0)**2 / 2.0).squeeze()
    E = U + T

    plt.figure()
    plt.plot(sol.t, E)
    plt.xlabel("Time")
    plt.ylabel("Energy")



#########################
## Low Level Functions ##
#########################

def plot_1d(x, y, y0_hline=False, hlines=[], vlines=[], **kwargs):
    plt.plot(x,y)

    if y0_hline:
        hlines.append(y[0])

    for hline in hlines:
        plt.hlines(hline, np.min(x), np.max(x))
    for vline in vlines: 
        plt.vlines(vline, np.min(y), np.max(y))


##########################
## High Level Functions ##
##########################


def __plot_TraditionalOE(t, OE, **kwargs):    
    vis = VisualizationBase()
    vis.newFig()
    plt.subplot(3,2,1)
    plot_1d(t, OE[:,0], **kwargs) 
    plt.ylabel("Semi Major Axis")
    plt.subplot(3,2,2)
    plot_1d(t, OE[:,1], **kwargs)
    plt.ylabel("Eccentricity")
    plt.subplot(3,2,3)
    plot_1d(t, OE[:,2], **kwargs)
    plt.ylabel("Inclination")

    plt.subplot(3,2,4)
    plot_1d(t, OE[:,3], **kwargs) 
    plt.ylabel("Arg of Periapsis")
    plt.subplot(3,2,5)
    plot_1d(t, OE[:,4], **kwargs)
    plt.ylabel("RAAN")
    plt.subplot(3,2,6)
    plot_1d(t, OE[:,5], **kwargs)
    plt.ylabel("M")
    plt.suptitle("Traditional Elements")

def __plot_DelaunayOE(t, DelaunayOE, **kwargs):
    vis = VisualizationBase()
    vis.newFig()
    plt.subplot(3,2,1)
    plot_1d(t, DelaunayOE[:,0], **kwargs) 
    plt.ylabel("l")
    plt.subplot(3,2,2)
    plot_1d(t, DelaunayOE[:,1], **kwargs) 
    plt.ylabel("g")
    plt.subplot(3,2,3)
    plot_1d(t, DelaunayOE[:,2], **kwargs) 
    plt.ylabel("h")

    plt.subplot(3,2,4)
    plot_1d(t, DelaunayOE[:,3], **kwargs) 
    plt.ylabel("L")
    plt.subplot(3,2,5)
    plot_1d(t, DelaunayOE[:,4], **kwargs) 
    plt.ylabel("G")
    plt.subplot(3,2,6)
    plot_1d(t, DelaunayOE[:,5], **kwargs) 
    plt.ylabel("H")
    plt.suptitle("Delaunay Elements")

def __plot_EquinoctialOE(t, EquinoctialOE, **kwargs):   
    vis = VisualizationBase()
    vis.newFig()
    plt.subplot(3,2,1)
    plot_1d(t, EquinoctialOE[:,0], **kwargs) 
    plt.ylabel("p")
    plt.subplot(3,2,2)
    plot_1d(t, EquinoctialOE[:,1], **kwargs) 
    plt.ylabel("f")
    plt.subplot(3,2,3)
    plot_1d(t, EquinoctialOE[:,2], **kwargs) 
    plt.ylabel("g")

    plt.subplot(3,2,4)
    plot_1d(t, EquinoctialOE[:,3], **kwargs) 
    plt.ylabel("L")
    plt.subplot(3,2,5)
    plot_1d(t, EquinoctialOE[:,4], **kwargs) 
    plt.ylabel("h")
    plt.subplot(3,2,6)
    plot_1d(t, EquinoctialOE[:,5], **kwargs) 
    plt.ylabel("k")
    plt.suptitle("Equinoctial Elements")

def __plot_MilankovitchOE(t, MilankovitchOE, **kwargs):   
    vis = VisualizationBase()
    vis.newFig()
    plt.subplot(3,3,1)
    plot_1d(t, MilankovitchOE[:,0], **kwargs) 
    plt.ylabel("H1")
    plt.subplot(3,3,2)
    plot_1d(t, MilankovitchOE[:,1], **kwargs) 
    plt.ylabel("H2")
    plt.subplot(3,3,3)
    plot_1d(t, MilankovitchOE[:,2], **kwargs) 
    plt.ylabel("H3")

    plt.subplot(3,3,4)
    plot_1d(t, MilankovitchOE[:,3], **kwargs) 
    plt.ylabel("e1")
    plt.subplot(3,3,5)
    plot_1d(t, MilankovitchOE[:,4], **kwargs) 
    plt.ylabel("e2")
    plt.subplot(3,3,6)
    plot_1d(t, MilankovitchOE[:,5], **kwargs) 
    plt.ylabel("e3")

    plt.subplot(3,3,8)
    plot_1d(t, MilankovitchOE[:,6], **kwargs) 
    plt.ylabel("l")
    plt.suptitle("Milankovitch Elements")


###################
## API Functions ##
###################

def plot_OE_1d(t, OE, element_set, **kwargs):
    """Plot the orbital elements in subplots

    Args:
        t (np.array): time vector
        OE (np.array): orbital element vector [N x -1] (time x state_dim)
        element_set (str): element set key

    Keyword Args:
        y0_hlines (bool) : add an hline at y0

    """
    if element_set.lower() == "traditional":
        __plot_TraditionalOE(t, OE, **kwargs)
    elif element_set.lower() == "delaunay":
        __plot_DelaunayOE(t, OE, **kwargs)
    elif element_set.lower() == "equinoctial":
        __plot_EquinoctialOE(t, OE, **kwargs)
    elif element_set.lower() == "milankovitch":
        __plot_MilankovitchOE(t, OE, **kwargs)
    else:
        raise NotImplementedError(element_set + " is not a currently supported orbital element set!")

def plot_cartesian_state_1d(t, X, **kwargs):
    """Plot the cartesian state vector in subplots

    Args:
        t (np.array): time vector
        X (np.array): state vector [N x 6]

    Keyword Args:
        y0_hlines (bool) : add an hline at y0
    """

    plt.figure()
    plt.subplot(2,3,1)
    plot_1d(t, X[:,0], **kwargs)
    plt.ylabel("$x$")
    plt.subplot(2,3,2)
    plot_1d(t, X[:,1], **kwargs)
    plt.ylabel("$y$")
    plt.subplot(2,3,3)
    plot_1d(t, X[:,2], **kwargs)
    plt.ylabel("$z$")
    plt.xlabel("Time")

    plt.subplot(2,3,4)
    plot_1d(t, X[:,3], **kwargs)
    plt.ylabel("$\dot{x}$")
    plt.subplot(2,3,5)
    plot_1d(t, X[:,4], **kwargs)
    plt.ylabel("$\dot{y}$")
    plt.subplot(2,3,6)
    plot_1d(t, X[:,6], **kwargs)
    plt.ylabel("$\dot{z}$")
    plt.xlabel("Time")

def plot_cartesian_state_3d(X, obj_file=None, **kwargs):
    """Plot the cartesian state vector in subplots

    Args:
        t (np.array): time vector
        X (np.array): state vector [N x 6]
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x_i, y_i, z_i = X[0,0:3]
    op.plot3d(X[:,0:3].T, obj_file=obj_file, new_fig=False, traj_cm=plt.cm.PiYG) 
    plt.gca().scatter(x_i, y_i, z_i, s=3, c='r')
