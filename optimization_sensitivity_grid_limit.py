import multiprocessing
import optimization as m
import matplotlib.pyplot as plt
import numpy as np

def opt(grid_limit):
    
    fast = False
    printed = 0
    
    nb = 104
    
    plant = m.H2plant()
    plant.change_grid_limit(grid_limit)
    plant.get_data_from_excel(nb)
    plant.power_manager()   
    avg = np.mean(plant.data['excess_power'])
    
    # STEP 1
    
    if fast == True:
        n_best_design = 1
        n_iter = 1
        len_pop = 17
        nb_proc = 3
    else:
        n_best_design = 2
        n_iter = 15
        len_pop = 40
        nb_proc = 14
    
    C_min = 1000
    C_max = int(max(plant.data['excess_power']))
    
    if C_min >= C_max:
        C_min = C_max/8
        
    S_min = 10
    S_max = 2500
    N_min = 1
    N_max = 200
    

    
    inputs = (plant, n_best_design, n_iter, len_pop, C_min, C_max, S_min, S_max, N_min, N_max, printed)
    args = [inputs]*nb_proc
    
    # Multiprocessing
    
    print("Step 1")
    with multiprocessing.Pool(processes=nb_proc) as pool:
        results = pool.starmap_async(m.optimization, args)
        results = results.get()
    
    # Result processing
    
    best_configs_LCOHs = []
    
    for res in results:
        best_configs_LCOHs.append(res[1])
    
    # New boundaries
    best_configs_LCOHs_sorted = sorted(best_configs_LCOHs, key=lambda x:x[-1])
    best_configs_LCOHs_sorted = best_configs_LCOHs_sorted[0]
    
    shr = 20/100
    
    S_min_old, S_max_old = S_min, S_max
    N_min_old, N_max_old = N_min, N_max
    
    C_b = best_configs_LCOHs_sorted[0]
    C_min, C_max = int(C_b/3), int(C_b*3/3)
        
    S_b = best_configs_LCOHs_sorted[1]
    S_r = S_max - S_min
    S_min, S_max =  max(int(S_b - S_r*shr), S_min_old), min(int(S_b + S_r*shr), S_max_old)
    
    N_b = best_configs_LCOHs_sorted[2]
    N_r = N_max-N_min
    N_min, N_max =  max(int(N_b - N_r*shr), N_min_old) , min(int(N_b + N_r*shr), N_max_old) 

    # STEP 2
    
    inputs = (plant, n_best_design, n_iter, len_pop, C_min, C_max, S_min, S_max, N_min, N_max, printed)
    args = [inputs]*nb_proc
    
    # Multiprocessing
    
    print("Step 2")
    with multiprocessing.Pool(processes=nb_proc) as pool2:
        results2 = pool2.starmap_async(m.optimization, args)
        results2 = results2.get()
    
    # Result processing
    
    best_configs_LCOHs = []
    
    for res in results2:
        best_configs_LCOHs.append(res[1])
    
    # New boundaries
    best_configs_LCOHs_sorted = sorted(best_configs_LCOHs, key=lambda x:x[-1])
    best_configs_LCOHs_sorted= best_configs_LCOHs_sorted[0]
    
    C_min_old, C_max_old = C_min, C_max
    S_min_old, S_max_old = S_min, S_max
    N_min_old, N_max_old = N_min, N_max

    C_b = best_configs_LCOHs_sorted[0]
    C_r = C_max - C_min
    C_min, C_max = max(int(C_b - C_r*shr), C_min_old), min(int(C_b + C_r*shr), C_max_old)
    
    S_b = best_configs_LCOHs_sorted[1]
    S_r = S_max - S_min
    S_min, S_max =  max(int(S_b - S_r*shr), S_min_old), min(int(S_b + S_r*shr), S_max_old)
    
    N_b = best_configs_LCOHs_sorted[2]
    N_r = N_max-N_min
    N_min, N_max =  max(int(N_b - N_r*shr), N_min_old) , min(int(N_b + N_r*shr), N_max_old) 
    
    # STEP 3
    
    inputs = (plant, n_best_design, n_iter, len_pop, C_min, C_max, S_min, S_max, N_min, N_max, printed)
    args = [inputs]*nb_proc
    
    # Multiprocessing
    
    print("Step 3")
    with multiprocessing.Pool(processes=nb_proc) as pool2:
        results2 = pool2.starmap_async(m.optimization, args)
        results2 = results2.get()
    
    # Result processing
    
    best_configs_LCOHs = []
    
    for res in results2:
        best_configs_LCOHs.append(res[1])
        
    # New boundaries
    best_configs_LCOHs_sorted = sorted(best_configs_LCOHs, key=lambda x:x[-1])
    best_configs_LCOHs_sorted= best_configs_LCOHs_sorted[0]
    
    C_min_old, C_max_old = C_min, C_max
    S_min_old, S_max_old = S_min, S_max
    N_min_old, N_max_old = N_min, N_max
    
    C_b = best_configs_LCOHs_sorted[0]
    C_r = C_max - C_min
    C_min, C_max = max(int(C_b - C_r*shr), C_min_old), min(int(C_b + C_r*shr), C_max_old)
    
    S_b = best_configs_LCOHs_sorted[1]
    S_r = S_max - S_min
    S_min, S_max =  max(int(S_b - S_r*shr), S_min_old), min(int(S_b + S_r*shr), S_max_old)
    
    N_b = best_configs_LCOHs_sorted[2]
    N_r = N_max-N_min
    N_min, N_max =  max(int(N_b - N_r*shr), N_min_old) , min(int(N_b + N_r*shr), N_max_old) 
    
    # STEP 4
    
    inputs = (plant, n_best_design, n_iter, len_pop, C_min, C_max, S_min, S_max, N_min, N_max, printed)
    args = [inputs]*nb_proc
    
    # Multiprocessing
    
    print("Step 4")
    with multiprocessing.Pool(processes=nb_proc) as pool3:
        results3 = pool3.starmap_async(m.optimization, args)
        results3 = results3.get()
    
    # Result processing
    
    best_configs_LCOHs = []
    
    for res in results3:
        best_configs_LCOHs.append(res[1])
    
    # New boundaries
    best_configs_LCOHs_sorted = sorted(best_configs_LCOHs, key=lambda x:x[-1])
    best_configs_LCOHs_sorted=best_configs_LCOHs_sorted[0]
    
    return best_configs_LCOHs_sorted, avg

def main():
    # grid_limits = [1e6, 1e6+2e5, 1e6+3e5, 1e6+4e5, 1e6+5e5]
    grid_limits = [1e6, 1e6+1e5, 1e6+2e5, 1319414, 1e6+4e5, 1e6+5e5, 1e6+6e5, 1e6+7e5]

    cpt = 1
    
    Cs = []
    Ss = []
    Ns = []
    LCOHs = []
    
    avg_ep = []
    for grid_limit in grid_limits:   
        print(f"{cpt}/{len(grid_limits)} iterations.")
        res, avg = opt(grid_limit)
        Cs.append(res[0])
        Ss.append(res[1])
        Ns.append(res[2])
        LCOHs.append(res[3])
        
        avg_ep.append(avg)
        cpt+=1
    
    # Results show 
    fig, axis = plt.subplots(2, 2)
    
    axis[0, 0].plot(grid_limits, Cs)

    axis[0, 0].set_xlabel("Grid limit [kW]")
    axis[0, 0].set_ylabel("Electrolyzer capacity[kW]")
    
    axis[0, 1].plot(grid_limits, Ss)

    axis[0, 1].set_xlabel("Grid limit [kW]")
    axis[0, 1].set_ylabel("Storage capacity [m3]")
    
    axis[1, 0].plot(grid_limits, Ns)

    axis[1, 0].set_xlabel("Grid limit [kW]")
    axis[1, 0].set_ylabel("Number of trucks")
    
    axis[1, 1].plot(grid_limits, LCOHs)

    axis[1, 1].set_xlabel("Grid limit [kW]")
    axis[1, 1].set_ylabel("LCOH [EUR/kWh]")
    
    fig.suptitle("Evolution of the different variables and the LCOH in relation to the grid limit")
    fig.show()

    fig, ax = plt.subplots()
    
    ax.plot(grid_limits, Cs, label="Electrolyzer capacity [kW]")
    ax.plot(grid_limits, avg_ep, label="Avergae excess power [kW]")
    axis[0, 0].set_ylabel("Power [kW]")
    
    ax.legend()  
    
    fig.suptitle("Evolution of the the electrolyzer capacity and the excess power in relation to the grid limit")

    fig.show()
    
    return Cs, Ss, Ns, LCOHs, avg_ep
    
if __name__ == '__main__':
    res = main()
    print(res)