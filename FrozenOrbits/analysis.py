import numpy as np
import trimesh

from FrozenOrbits.utils import calc_angle_diff

def print_state_differences(sol):
    pos_diff = np.linalg.norm(sol.y[0:3, -1] - sol.y[0:3, 0])
    vel_diff = np.linalg.norm(sol.y[3:6, -1] - sol.y[3:6, 0])

    print(f"Integrated Position Difference {pos_diff} [m]")
    print(f"Integrated Velocity Difference {vel_diff} [m/s]")


def print_OE_differences(OE_sol, lpe, prefix, constraint_angle_wrap):

    OE_dim_start = OE_sol[0,:]
    OE_dim_end = OE_sol[-1,:]
    OE_dim_diff = OE_dim_end - OE_dim_start

    OE_start = lpe.non_dimensionalize_state(OE_dim_start.reshape((1,-1))).numpy()[0]
    OE_end = lpe.non_dimensionalize_state(OE_dim_end.reshape((1,-1))).numpy()[0]
    OE_diff = OE_end - OE_start

    for i in range(len(OE_end)):
        if constraint_angle_wrap[i]:
            OE_diff[i] = calc_angle_diff(OE_start[i], OE_end[i])
            OE_dim_diff[i] = calc_angle_diff(OE_dim_start[i], OE_dim_end[i])
    
    print(f"{prefix} OE Differences {OE_dim_diff} \t Total dOE: {np.linalg.norm(OE_dim_diff)}")
    print(f"{prefix} OE Non-Dim Differences {OE_diff} \t Total dOE: {np.linalg.norm(OE_diff)}")

    return OE_dim_diff, OE_diff


def check_for_intersection(sol, obj_file):
    mesh = trimesh.load_mesh(obj_file)
    proximity = trimesh.proximity.ProximityQuery(mesh)
    traj = sol.y[0:3, :]
    cart_pos_in_km = traj/1E3 
    distance = proximity.signed_distance(cart_pos_in_km.reshape((3,-1)).T) 
    if np.any(distance > 0):
        print(" Solution Intersected Body!")
        valid = False
    
