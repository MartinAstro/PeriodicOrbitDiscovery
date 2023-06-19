import os

import GravNN
import matplotlib.pyplot as plt
import numpy as np

import FrozenOrbits
from FrozenOrbits.boundary_conditions import *
from FrozenOrbits.bvp import *
from FrozenOrbits.constraints import *
from FrozenOrbits.gravity_models import pinnGravityModel, polyhedralGravityModel
from FrozenOrbits.LPE import *
from FrozenOrbits.visualization import *
from Scripts.BVP.initial_conditions import *


def main():
    pinn_model = pinnGravityModel(os.path.dirname(GravNN.__file__) + \
        "/../Data/Dataframes/eros_BVP_PINN_III.data")  
    planet = pinn_model.config['planet'][0]
    poly_model = polyhedralGravityModel(planet, planet.obj_8k)
    
    directory =  os.path.dirname(FrozenOrbits.__file__)+ "/Data/"
    df = pd.read_pickle(directory + "propagation_time_error.data")

    vis = VisualizationBase(save_directory=os.path.dirname(FrozenOrbits.__file__) + "/../Plots/")
    # (width, height)
    vis.newFig(fig_size=(3.25*0.9, 3.25)) # position
    # vis.newFig(fig_size=(2.1, 3.25*0.618)) # position
    # vis.newFig(fig_size=vis.half_page) # position
    # vis.newFig(fig_size=(2.1, 1.1)) # velocity
    # vis.newFig(fig_size=(2.1, 1.1)) # acceleration
    vis.new3DFig(fig_size=(3.25, 3.25)) # 3d cartesian

    semi_list = []
    dr_list = []
    dv_list = []
    T_8_list = []
    T_200_list = []
    T_PINN_list = []
    color_list = []
    for k in range(0,3):#len(df)):
        pinn_sol = df["pinn_sol"][k]
        poly_sol = df["poly_sol"][k]
    
        pinn_pos = pinn_sol.y[0:3,:].T
        pinn_vel = pinn_sol.y[3:6,:].T
        pinn_acc = pinn_model.generate_acceleration(pinn_pos)

        poly_pos = poly_sol.y[0:3,:].T
        poly_vel = poly_sol.y[3:6,:].T
        poly_acc = poly_model.generate_acceleration(pinn_pos)
        
        pos_diff = pinn_pos - poly_pos
        vel_diff = pinn_vel - poly_vel
        acc_diff = pinn_acc - poly_acc

        pos_diff_mag = np.linalg.norm(pos_diff, axis=1)
        vel_diff_mag = np.linalg.norm(vel_diff, axis=1)
        acc_diff_mag = np.linalg.norm(acc_diff, axis=1)
        
        plt.figure(1)
        plt.subplot(3,1,3)
        plt.plot(pinn_sol.t, acc_diff_mag, linewidth=1)
        plt.ylabel(r"$\delta a$ [m/s$^2$]", fontsize=8)
        plt.xlabel("Time [s]", fontsize=8)
        plt.gca().xaxis.set_major_formatter(MathTextSciFormatter("%1.1e"))
        plt.gca().yaxis.set_major_formatter(MathTextSciFormatter("%1.0e"))
        plt.gca().tick_params(labelsize=8)
        plt.xticks(rotation=30)
        plt.yscale('log')
        plt.grid(which='both',linestyle="--", linewidth=0.1, color='.25', zorder=-10)


        plt.subplot(3,1,1)
        plt.plot(pinn_sol.t, pos_diff_mag, linewidth=1)
        plt.ylabel(r"$\delta x$ [m]", fontsize=8)
        plt.gca().set_xticklabels([])
        plt.gca().yaxis.set_major_formatter(MathTextSciFormatter("%1.0e"))
        plt.gca().tick_params(labelsize=8)
        plt.yscale('log')
        plt.grid(which='both',linestyle="--", linewidth=0.1, color='.25', zorder=-10)

        # plt.xlabel("Time [s]")

        # plt.figure(2)
        plt.subplot(3,1,2)
        plt.plot(pinn_sol.t, vel_diff_mag, linewidth=1)
        plt.ylabel(r"$\delta v$ [m/s]", fontsize=8)
        plt.gca().set_xticklabels([])
        plt.gca().yaxis.set_major_formatter(MathTextSciFormatter("%1.0e"))
        plt.gca().tick_params(labelsize=8)
        plt.yscale('log')
        plt.grid(which='both',linestyle="--", linewidth=0.1, color='.25', zorder=-10)
        plt.tight_layout()
        # plt.xlabel("Time [s]")

        color = plt.gca().lines[-1].get_color()
        
        plt.figure(2)
        plot_cartesian_state_3d(pinn_sol.y.T, planet.obj_8k, solid_color=color, 
                                new_fig=False,plot_start_point=False,
                                linewidth=1)
        plot_cartesian_state_3d(poly_sol.y.T, planet.obj_8k, solid_color=color, linestyle='--', 
                                new_fig=False,plot_start_point=False,
                                linewidth=1)
        plt.gca().tick_params(labelsize=8)
        plt.gca().xaxis.set_major_locator(plt.MaxNLocator(4))
        plt.gca().yaxis.set_major_locator(plt.MaxNLocator(4))
        plt.gca().zaxis.set_major_locator(plt.MaxNLocator(4))

        df["Xf_pinn"][k][0:3]
        df["Xf_poly"][k][0:3]
        dr = np.linalg.norm(df["Xf_pinn"][k][0:3] - df["Xf_poly"][k][0:3])
        dv = np.linalg.norm(df["Xf_pinn"][k][3:6] - df["Xf_poly"][k][3:6])
        

        semi_list.append(df["semi"][k])
        T_200_list.append( df["dt_poly_200"][k])
        T_8_list.append( df["dt_poly"][k])
        T_PINN_list.append( df["dt_pinn"][k])
        dr_list.append(dr)
        dv_list.append(dv)
        color_list.append(color)


    vis.save(plt.figure(1), os.path.basename(__file__).split('.py')[0] + '_dr.pdf')
    # vis.save(plt.figure(2), os.path.basename(__file__).split('.py')[0] + '_dv.pdf')
    # vis.save(plt.figure(3), os.path.basename(__file__).split('.py')[0] + '_da.pdf')
    vis.save(plt.figure(2), os.path.basename(__file__).split('.py')[0] + '_3d.pdf')
    print("\\hline")
    print("Orbit & $\\delta r$ [m] & $\\delta v$ [m/s] & $a$ & $T_ Speedup \\\\")
    print("\\hline")
    for k in range(len(dr_list)):
        print(f"{color_list[k].capitalize()} & {dr_list[k]} & {dv_list[k]} & {semi_list[k]} & {T_200_list[k]} & {T_8_list[k]} & {T_PINN_list[k]} & {T_8_list[k]/T_PINN_list[k]} \\\\")
    print("\\hline")
    plt.show()

if __name__ == "__main__":
    main()
