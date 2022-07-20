import numpy as np
import copy
import os
from FrozenOrbits.ivp import *
from FrozenOrbits.coordinate_transforms import *
import matplotlib.pyplot as plt
from GravNN.Visualization.VisualizationBase import VisualizationBase
import matplotlib.animation as animation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.ticker as mticker


class MathTextSciFormatter(mticker.Formatter):
    def __init__(self, fmt="%1.2e"):
        self.fmt = fmt
    def __call__(self, x, pos=None):
        s = self.fmt % x
        decimal_point = '.'
        positive_sign = '+'
        tup = s.split('e')
        significand = tup[0].rstrip(decimal_point)
        sign = tup[1][0].replace(positive_sign, '')
        exponent = tup[1][1:].lstrip('0')
        if exponent:
            exponent = '10^{%s%s}' % (sign, exponent)
        if significand and exponent:
            s =  r'%s{\times}%s' % (significand, exponent)
        else:
            s =  r'%s%s' % (significand, exponent)
        return "${}$".format(s)

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
    options = {
        "label" : None,
        "color" : 'black',
        "new_fig" : False
    }
    options.update(kwargs)

    if options['new_fig']:
        vis = VisualizationBase()
        vis.newFig()

    plt.plot(x,y, label=options['label'], color=options['color'])

    hlines_local = copy.deepcopy(hlines)
    if y0_hline:
        hlines_local.append(y[0])

    for hline in hlines_local:
        plt.hlines(hline, np.min(x), np.max(x),linestyle='--')
    for vline in vlines: 
        plt.vlines(vline, np.min(y), np.max(y),linestyle='--')

def plot_3d(rVec, traj_cm=plt.cm.jet, solid_color=None, reverse_cmap=False, **kwargs):

    ax = plt.gca()
    rx = [rVec[0]]
    ry = [rVec[1]]
    rz = [rVec[2]]

    label = kwargs.get('label', None)
    legend = kwargs.get('legend', None)
    linewidth = kwargs.get('linewidth', None)
    # if there is a cmap specified, break the line into segments
    # and vary the color to show time evolution
    if traj_cm is not None and solid_color is None:
        N = len(rx[0])
        cVec = np.zeros((N,4))
        for i in range(N-1):
            if reverse_cmap:
                cVec[i] = traj_cm(1 - i/N)
            else:
                cVec[i] = traj_cm(i/N)
            ax.plot(rx[0][i:i+2], ry[0][i:i+2], rz[0][i:i+2], 
                    color=cVec[i], alpha=kwargs['line_opacity'], label=label, linewidth=linewidth)
    else:
        # Just plot a line, gosh.
        if solid_color is None:
            solid_color = 'black'
        ax.plot(rx[0], ry[0], rz[0], c=solid_color, alpha=kwargs['line_opacity'], label=label, linewidth=linewidth)

    maxVal = max([max(np.abs(np.concatenate((plt.gca().get_xlim(), rx[0])))), 
                  max(np.abs(np.concatenate((plt.gca().get_ylim(), ry[0])))), 
                  max(np.abs(np.concatenate((plt.gca().get_zlim(), rz[0]))))])
    minVal =  -maxVal

    ax.set_xlim(minVal, maxVal)
    ax.set_ylim(minVal, maxVal)
    ax.set_zlim(minVal, maxVal)

    ax.set_xlabel('x')
    ax.set_ylabel('y')

    ax.xaxis.set_major_formatter(MathTextSciFormatter("%1.0e"))
    ax.yaxis.set_major_formatter(MathTextSciFormatter("%1.0e"))
    ax.zaxis.set_major_formatter(MathTextSciFormatter("%1.0e"))
    
    # ax.view_init(90, 30)
    ax.view_init(45, 45)
    plt.tight_layout()

    return ax

##########################
## High Level Functions ##
##########################


def __plot_TraditionalOE(t, OE, **kwargs):    
    options = {
        "horizontal" : False,
        "individual_figures" : False
    }
    options.update(kwargs)
    if options['horizontal']:
        grid_rows, grid_columns = 2, 3
    else:
        grid_rows, grid_columns = 3, 2

    element_labels = ["$a$", "$e$", "$i$",
                      "$\omega$", "$\Omega$", "$M$"]
    
    for i in range(len(OE[0])):

        # Plot elements separately or on a single plot
        if options['individual_figures']:
            kwargs.update({"new_fig" : True})
        else:
            plt.subplot(grid_rows, grid_columns, i+1)
        plot_1d(t, OE[:,i], **kwargs) 
        plt.ylabel(element_labels[i])
    
    if not options['individual_figures']:
        plt.suptitle("Traditional Elements")

def __plot_DelaunayOE(t, DelaunayOE, **kwargs):
    options = {
        "horizontal" : False,
        "individual_figures" : False
    }
    options.update(kwargs)
    if options['horizontal']:
        grid_rows, grid_columns = 2, 3
    else:
        grid_rows, grid_columns = 3, 2

    element_labels = ["l", "g", "h", "L", "G", "H"]
    for i in range(len(OE[0])):
        # Plot elements separately or on a single plot
        if options['individual_figures']:
            kwargs.update({"new_fig" : True})
        else:
            plt.subplot(grid_rows, grid_columns, i+1)
        plot_1d(t, DelaunayOE[:,i], **kwargs) 
        plt.ylabel(element_labels[i])
    plt.suptitle("Delaunay Elements")

def __plot_EquinoctialOE(t, EquinoctialOE, **kwargs):   
    options = {
        "horizontal" : False,
        "individual_figures" : False
    }
    options.update(kwargs)
    if options['horizontal']:
        grid_rows, grid_columns = 2, 3
    else:
        grid_rows, grid_columns = 3, 2

    element_labels = ["p", "f", "g", "L", "h", "k"]
    for i in range(len(OE[0])):
        # Plot elements separately or on a single plot
        if options['individual_figures']:
            kwargs.update({"new_fig" : True})
        else:
            plt.subplot(grid_rows, grid_columns, i+1)
        plot_1d(t, EquinoctialOE[:,i], **kwargs) 
        plt.ylabel(element_labels[i])
    plt.suptitle("Equinoctial Elements")

def __plot_MilankovitchOE(t, MilankovitchOE, **kwargs):   
    options = {
        "individual_figures" : False
    }
    element_labels = ["$h_1$", "$h_2$", "$h_3$", "$e_1$", "$e_2$", "$e_3$", "$l$"]
    for i in range(len(OE[0])):
        # Plot elements separately or on a single plot
        if options['individual_figures']:
            kwargs.update({"new_fig" : True})
        else:
            plt.subplot(3, 3, i+1)
        plot_1d(t, MilankovitchOE[:,i], **kwargs) 
        plt.ylabel(element_labels[i])
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
    if kwargs.get('new_fig', True):
        vis = VisualizationBase()
        vis.newFig()

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
    plot_1d(t, X[:,5], **kwargs)
    plt.ylabel("$\dot{z}$")
    plt.xlabel("Time")

def plot_cartesian_state_3d(X, obj_file=None, **kwargs):
    """Plot the cartesian state vector in subplots

    Args:
        t (np.array): time vector
        X (np.array): state vector [N x 6]
    """
    options = {
        "cmap" : plt.cm.winter,
        "plot_start_point" : True,
        "new_fig" : True,
        "line_opacity" : 1.0,
        }
    options.update(kwargs)

    if options["new_fig"]:
        vis = VisualizationBase(formatting_style='AIAA')
        fig, ax = vis.new3DFig()
    else:
        ax = plt.gca()

    plot_3d(X[:,0:3].T, obj_file=obj_file, traj_cm=options['cmap'], **options) 

    # Import the asteroid shape model if appropriate. 
    if obj_file is not None:
        import trimesh
        filename, file_extension = os.path.splitext(obj_file)
        mesh = trimesh.load_mesh(obj_file, file_type=file_extension[1:])
        cmap = plt.get_cmap('Greys')
        tri = Poly3DCollection(mesh.triangles*1000, cmap=cmap, facecolors=cmap(128), alpha=0.5)
        p = plt.gca().add_collection3d(tri)

    if options['plot_start_point']:
        plt.gca().scatter(X[0,0], X[0,1], X[0,2], s=3, c='r')
