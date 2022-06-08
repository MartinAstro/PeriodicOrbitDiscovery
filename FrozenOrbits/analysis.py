import numpy as np
import trimesh

def print_state_differences(sol):
    pos_diff = np.linalg.norm(sol.y[0:3, -1] - sol.y[0:3, 0])
    vel_diff = np.linalg.norm(sol.y[3:6, -1] - sol.y[3:6, 0])

    print(f"Integrated Position Difference {pos_diff} [m]")
    print(f"Integrated Velocity Difference {vel_diff} [m/s]")


def check_for_intersection(sol, obj_file):
    mesh = trimesh.load_mesh(obj_file)
    proximity = trimesh.proximity.ProximityQuery(mesh)
    traj = sol.y[0:3, :]
    cart_pos_in_km = traj/1E3 
    distance = proximity.signed_distance(cart_pos_in_km.reshape((3,-1)).T) 
    if np.any(distance > 0):
        print(" Solution Intersected Body!")
        valid = False
    
