import numpy as np
import matplotlib.pyplot as plt
import FrozenOrbits
from GravNN.Visualization.VisualizationBase import VisualizationBase
import pathlib
import pickle
import os
def collect_plotting_data(solutions):
    tVec = []
    yVec = []
    i = 1
    for solution in solutions:
        tVec.append(solution['t_compute'])
        yVec.append(i)
        i += 1
    return tVec, yVec

def load_data(model_name):
    path = os.path.dirname(FrozenOrbits.__file__)
    file = path +  "/../Data/BVP_Solutions/V2/" + model_name + "_stats.data"
    with open(file, 'rb') as f:
        results = pickle.load(f)
    return results

def compute_unique_solutions_count(results):
    unique_count_vec = []
    k = 0
    for result in results:
        criteria = result['valid']
        if criteria:
            criteria = criteria * (result['closest_approach'] < 3000)
            criteria = criteria * (np.linalg.norm(result['closest_state'][3:6] - result['initial_condition'][3:6]) < 0.3)
        if criteria:
            k += 1
            unique_count_vec.append(k)
        else:
            unique_count_vec.append(k)
    return unique_count_vec



def main():
    pinn_results = load_data(model_name="PINN")
    simple_results = load_data(model_name="Simple")

    pinn_counts = compute_unique_solutions_count(pinn_results)
    simple_counts = compute_unique_solutions_count(simple_results)

    IC_counts = np.linspace(1, 100, 100)

    vis = VisualizationBase(formatting_style="AIAA")
    vis.fig_size = vis.AIAA_full_page
    vis.newFig()
    plt.plot(IC_counts, pinn_counts, label='PINN Count')
    plt.plot(IC_counts, simple_counts, label='Simple Count')
    plt.xlabel("Number of IC")
    plt.ylabel("Near-Periodic Orbits Found")
    plt.legend()
    path = os.path.dirname(FrozenOrbits.__file__)
    # vis.save(plt.gcf(), path + "/../Plots/FrozenOrbitCount.pdf")
    plt.show()


    

if __name__ == "__main__":
    main()
