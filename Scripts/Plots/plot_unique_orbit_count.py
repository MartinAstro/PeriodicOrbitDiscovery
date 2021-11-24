import numpy as np
import matplotlib.pyplot as plt
from GravNN.Visualization.VisualizationBase import VisualizationBase
import pathlib
import pickle

def collect_plotting_data(solutions):
    tVec = []
    yVec = []
    i = 1
    for solution in solutions:
        tVec.append(solution['t_compute'])
        yVec.append(i)
        i += 1
    return tVec, yVec


def main():
    file_path = pathlib.Path(__file__).parent.absolute()
    file = file_path.as_posix() +  "/Data/BVP_Solutions/PINN_trajectories.data"
    with open(file, 'rb') as f:
        valid_solutions = pickle.load(f)
        unique_solutions = pickle.load(f)
    
    t_valid, y_valid = collect_plotting_data(valid_solutions)
    t_unique, y_unique = collect_plotting_data(unique_solutions)

    vis = VisualizationBase()
    vis.newFig()
    plt.plot(t_valid, y_valid, label='Valid Count')
    plt.plot(t_unique, y_unique, label='Unique Count')
    plt.legend()
    plt.show()


    

if __name__ == "__main__":
    main()
