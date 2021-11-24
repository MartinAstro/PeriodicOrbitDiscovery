import numpy as np
import os
from scipy.integrate import solve_ivp
import pathlib
import pickle
from FrozenOrbits.ivp import solve_ivp_pos_problem
import numpy as np
import pandas as pd
from GravNN.CelestialBodies.Asteroids import Eros
from GravNN.Networks.Model import load_config_and_model

from FrozenOrbits.coordinate_transforms import *
from FrozenOrbits.LPE import LPE
from FrozenOrbits.coordinate_transforms import cart2oe_tf


from scipy.integrate import solve_ivp
import OrbitalElements.orbitalPlotting as op
from FrozenOrbits.ivp import solve_ivp_pos_problem
from FrozenOrbits.utils import compute_period, compute_period_from_state
from FrozenOrbits.events import coarse_periodic, fine_periodic, very_coarse_periodic
import trimesh

obj_file = Eros().model_potatok
filename, file_extension = os.path.splitext(obj_file)
mesh = trimesh.load_mesh(obj_file, file_type=file_extension[1:])
proximity = trimesh.proximity.ProximityQuery(mesh)

def collision(t, y, IC):
    cart_pos = y[0:3].reshape((1,3))
    cart_pos_in_km = cart_pos/1E3
    distance = proximity.signed_distance(cart_pos_in_km.reshape((1,3)))
    return distance
collision.terminal = True

def depart(t, y, IC):
    cart_pos = y[0:3].reshape((1,3))
    distance = 10*Eros().radius - np.linalg.norm(cart_pos)
    return distance
depart.terminal = True


def compute_valid_solutions(lpe, results):
    events = [collision, depart, coarse_periodic, very_coarse_periodic]

    valid_solutions_list = []
    i = 0
    for result in results:
        i += 1
        print("Orbit %d / %d " %(i, len(results)))
        # Draw the stats from the BVP solver
        time = result.time
        IC = result.y[:,0]
        
        if np.any(np.isnan(IC)):
            print("NaN in IC")
            continue

        # Run solve IVP on the IC and check if it intercepts / collides / departs
        t_period_2BP = compute_period_from_state(lpe.mu, IC.reshape((1,6)))
        
        if t_period_2BP < 1:
            print("Invalid Period")
            continue
        
        print("Period: " + str(t_period_2BP))
        sol = solve_ivp_pos_problem(t_period_2BP*1.5, IC, lpe, t_eval=None, events=events, args=(IC,))
        
        if len(sol.t_events[0]) > 0 or len(sol.t_events[1]) > 0:
            print("BVP violates gravity model bounds")
            continue

        if len(sol.t_events[3]) >= 3:
            print("Coarsely periodic boundary satisfied")

            # Find point of closest approach
            t_fine_mesh = np.linspace(sol.t_events[3][-2], sol.t_events[3][-1], 5000)
            sol = solve_ivp_pos_problem(t_period_2BP*1.5, IC, lpe, t_fine_mesh, args=(IC,))
            dRn = np.linalg.norm(sol.y[0:3, :] - IC[0:3].reshape((3,1)) , axis=0)
            t_period_exact_n = sol.t[np.where(dRn == np.min(dRn))][0]
            
            valid_solution = {
                "IC" : IC,
                "T_exact" : t_period_exact_n,
                "Min_dist" : np.min(dRn),
                "t_compute" : time
            }
            valid_solutions_list.append(valid_solution)
    return valid_solutions_list


def compute_unique_solutions(valid_solutions_list):

    unique_solutions_list = []
    unique_solutions_list.append(valid_solutions_list[0])
    for i in range(1, len(valid_solutions_list)):
        current_solution = valid_solutions_list[i]
        
        # Check if current solution is sufficiently different than 
        # the unique solutions
        unique = True
        for j in range(len(unique_solutions_list)):
            mean_percent_diff = np.mean(100*(np.abs(current_solution['IC'] - unique_solutions_list[j]['IC']))/np.abs(unique_solutions_list[j]['IC']))
            print("Mean Percent Diff " + str(mean_percent_diff))
            if mean_percent_diff < 1.0:
                unique = False
        if unique:
            unique_solutions_list.append(current_solution)

    return unique_solutions_list

def main():

    planet = Eros()
    df = pd.read_pickle("Data/Dataframes/eros_grav_model_minus_pm.data")
    config, model  = load_config_and_model(df.iloc[-1]['id'], df)
    lpe = LPE(model, config, planet.mu, element_set="traditional")

    file_path = pathlib.Path(__file__).parent.absolute()
    file = file_path.as_posix() +  "/Data/BVP_Solutions/PINN_bvp.data"
    with open(file, 'rb') as f:
        results = pickle.load(f)

    valid_solutions_list = compute_valid_solutions(lpe, results)
    unique_solutions_list = compute_unique_solutions(valid_solutions_list)

    file = file_path.as_posix() +  "/Data/BVP_Solutions/PINN_trajectories.data"
    os.makedirs(os.path.dirname(file), exist_ok=True)
    with open(file, 'wb') as f:
        pickle.dump(valid_solutions_list, f) 
        pickle.dump(unique_solutions_list, f) 



if __name__ == "__main__":
    main()