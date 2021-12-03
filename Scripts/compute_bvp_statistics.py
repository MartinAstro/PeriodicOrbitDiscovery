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
import GravNN
from FrozenOrbits.coordinate_transforms import *
from FrozenOrbits.LPE import LPE
from FrozenOrbits.coordinate_transforms import cart2oe_tf
import FrozenOrbits

from scipy.integrate import solve_ivp
import OrbitalElements.orbitalPlotting as op
from FrozenOrbits.gravity_models import simpleGravityModel, polyhedralGravityModel, noneGravityModel
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

class Result:
    def __init__(self):
        self.time = None
        self.y = None
        pass


def compute_bvp_solution_stats(lpe, results):
    events = [collision, depart, coarse_periodic, very_coarse_periodic]

    solutions_list = []
    i = 0
    for result in results:
        i += 1
        print("Orbit %d / %d " %(i, len(results)))

        IC = result.y[:,0]
        t_period_2BP = compute_period_from_state(lpe.mu, IC.reshape((1,6)))

        invalid_period = (t_period_2BP < 1)
        outside_bounds = np.linalg.norm(IC[0:3]) > Eros().radius*10

        if invalid_period or outside_bounds:
            print("Invalid Solution")
            solution = {
                "valid" : False,
                "failure" : "invalid",
                "initial_condition" : IC,
                "t_closest_approach" : None,
                "closest_state" : None,
                "percent_error" : None,
                "t_compute" : result.time,
                "closest_approach" : None,
            }
            solutions_list.append(solution)
            continue

        sol = solve_ivp_pos_problem(t_period_2BP*1.5, IC, lpe, t_eval=None, events=events, args=(IC,))

        if len(sol.t_events[0]) > 0 or len(sol.t_events[1]) > 0:
            print("BVP violates gravity model bounds")
            solution = {
                "valid" : False,
                "failure" : "Crash/Depart",
                "initial_condition" : IC,
                "t_closest_approach" : None,
                "closest_state" : None,
                "percent_error" : None,
                "t_compute" : result.time,
                "closest_approach" : None,
            }
            solutions_list.append(solution)
            continue

        if len(sol.t_events[3]) >= 3:
            print("Very coarsely periodic boundary satisfied")

            # Find point of closest approach
            t_fine_mesh = np.linspace(sol.t_events[3][-2], sol.t_events[3][-1], 5000)
            sol = solve_ivp_pos_problem(t_period_2BP*1.5, IC, lpe, t_fine_mesh, args=(IC,))
            dRn = np.linalg.norm(sol.y[0:3, :] - IC[0:3].reshape((3,1)) , axis=0)
            t_period_exact_n = sol.t[np.where(dRn == np.min(dRn))][0]
            closest_state = sol.y[:, np.where(dRn == np.min(dRn))].reshape((6,))
            percent_error = np.abs((closest_state.reshape((6,)) - IC.reshape((6,)))/ IC.reshape((6,)))*100
        
            solution = {
                "valid" : True,
                "failure" : "None",
                "initial_condition" : IC,
                "t_closest_approach" : t_period_exact_n,
                "closest_state" : closest_state,
                "percent_error" : percent_error,
                "t_compute" : result.time,
                "closest_approach" : np.min(dRn),
            }
        else:
            solution = {
                "valid" : False,
                "failure" : "Not periodic",
                "initial_condition" : IC,
                "t_closest_approach" : None,
                "closest_state" : None,
                "percent_error" : None,
                "t_compute" : result.time,
                "closest_approach" : None,
            }
        print(solution)
        solutions_list.append(solution)


    return solutions_list



def main():

    planet = Eros()
    df = pd.read_pickle(os.path.dirname(GravNN.__file__) + "/../Data/Dataframes/eros_grav_model_minus_pm.data")
    model_name = "PINN"
    # model_name = "Poly"
    # model_name = "Simple"

    config, model  = load_config_and_model(df.iloc[-1]['id'], df)
    lpe = LPE(model, config, planet.mu, element_set="traditional")

    path = os.path.dirname(FrozenOrbits.__file__)
    file = path +  "/../Data/BVP_Solutions/V2/" + model_name + "_bvp_solutions.data"
    with open(file, 'rb') as f:
        results = pickle.load(f)

    stats = compute_bvp_solution_stats(lpe, results)
    file = path +  "/../Data/BVP_Solutions/V2/" + model_name + "_stats.data"
    os.makedirs(os.path.dirname(file), exist_ok=True)
    with open(file, 'wb') as f:
        pickle.dump(stats, f) 

if __name__ == "__main__":
    main()