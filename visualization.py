import numpy as np
from ivp import *
from coordinate_transforms import *
import matplotlib.pyplot as plt
import OrbitalElements.orbitalPlotting as op
import matplotlib.animation as animation
def plot_OE_results(results, T, lpe):
    t_plot = np.linspace(0, T, 1000)
    y_plot = results.sol(t_plot)

    if np.isnan(y_plot).any():
        exit("NaNs found") 

    op.plot_OE(t_plot, y_plot, OE_set=lpe.element_set)

    OE_init = np.array(results.y[:,0]).reshape((1,6))

    R, V = oe2cart_tf(OE_init, lpe.mu, element_set=lpe.element_set)
    y = solve_ivp_pos_problem(T, np.hstack((R,V)), lpe, t_eval=t_plot)
    r_plot = y[0:3,:]

    # y = solve_ivp_problem(T, OE_init, lpe, t_eval=t_plot)
    # r_plot = []
    # for y in y_plot.T:
    #     y_i = np.array([y.T])
    #     R, V = oe2cart_tf(y_i ,lpe.mu, element_set=lpe.element_set)
    #     r_plot.append(R)
    #r_plot = np.array(r_plot).T.squeeze())

    op.plot_orbit_3d(r_plot)
    plt.show()

def plot_pos_results(results, T, lpe, obj_file=None, animate=False):
    t_plot = np.linspace(0, T, 1000)
    y_plot = results.sol(t_plot)

    if np.isnan(y_plot).any():
        exit("NaNs found") 

    op.plot_orbit_3d(y_plot, obj_file=obj_file)

    # if True:

    #     def animate_fcn(angle):
    #         plt.gca().view_init(angle,30)
    #         return plt.gcf(),
    #     # for angle in range(0, 360):
        
    #     anim = animation.FuncAnimation(plt.gcf(), animate_fcn, frames=360, interval=20, blit=True)
    #     # ani = animation.FuncAnimation(plt.gcf(), animate_fcn, fargs=(range(0,360)), interval=20, blit=False, save_count=50)

    #     anim.save("movie.gif")
        
        # or
        
        # writer = animation.FFMpegWriter(
        #     fps=15, metadata=dict(artist='Me'), bitrate=1800)
        # ani.save("movie.mp4", writer=writer)

    plt.show()