import pickle
import numpy as np
import matplotlib.pyplot as plt


def main():
    with open("bvp_pos_pinn.data", 'rb') as f:
        data = pickle.load(f)
    
    init_idx_list = data['init_idx_list']  
    sol_idx_list = data['sol_idx_list']  
    init_state_list = data['init_state_list']  
    sol_state_list = data['sol_state_list']  
    init_period_list = data['init_period_list']  
    sol_period_list = data['sol_period_list']  
    time_list = data['time_list']  

    num_solutions = []
    k = 0
    for i in range(len(time_list)):
        if i in sol_idx_list:
            k += 1
        num_solutions.append(k)
    
    
    IC_idx = np.arange(0, len(time_list), 1)
    BVP_idx = num_solutions #np.arange(0, len(sol_state_list), 1)
    plt.figure()
    plt.plot(time_list, IC_idx, label='Total IC')
    plt.plot(np.array(time_list), BVP_idx, label='Found Solutions')
    plt.xlabel("Time [s]")
    plt.ylabel("Count")
    plt.show()  

if __name__ == "__main__":
    main()