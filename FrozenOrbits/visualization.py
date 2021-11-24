import numpy as np
from FrozenOrbits.ivp import *
from FrozenOrbits.coordinate_transforms import *
import matplotlib.pyplot as plt
import OrbitalElements.orbitalPlotting as op
import matplotlib.animation as animation

def plot_OE_suite_from_state_sol(sol, planet):
    t_plot = sol.t
    y_plot = sol.y # [6, N]

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