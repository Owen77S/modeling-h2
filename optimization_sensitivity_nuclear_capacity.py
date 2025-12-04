import multiprocessing
import optimization as m
import matplotlib.pyplot as plt
import numpy as np

def opt(NC):
    
    printed = 0
    
    nb = 104
    
    plant = m.H2plant()
    plant.get_data_from_excel(nb)
    plant.change_nuclear_capacity(NC)
    plant.power_manager()
    
    # STEP 1
    
    fast = True
    
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

    nb_proc = 14
    
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
    
    return best_configs_LCOHs_sorted

def main():
    share_of_WC = [100, 200, 422, 600, 800, 1000]
    NCs = [343.2*s/100 for s in share_of_WC]
    cpt = 0
    
    Cs = []
    Ss = []
    Ns = []
    LCOHs = []
    
    for NC in NCs:
        print(f"{cpt+1}/{len(NCs)} iterations.")
        res = opt(NC)
        Cs.append(res[0])
        Ss.append(res[1])
        Ns.append(res[2])
        LCOHs.append(res[3])
        cpt += 1
    
    # Results show 

    fig, axis = plt.subplots(2, 2)
    
    axis[0, 0].plot(share_of_WC, Cs)

    axis[0, 0].set_xlabel("Share of the wind plant capacity [%]")
    axis[0, 0].set_ylabel("Electrolyzer capacity[kW]")
    
    axis[0, 1].plot(share_of_WC, Ss)

    axis[0, 1].set_xlabel("Share of the wind plant capacity [%]")
    axis[0, 1].set_ylabel("Storage capacity [m3]")
    
    axis[1, 0].plot(share_of_WC, Ns)

    axis[1, 0].set_xlabel("Share of the wind plant capacity [%]")
    axis[1, 0].set_ylabel("Number of trucks")
    
    axis[1, 1].plot(share_of_WC, LCOHs)

    axis[1, 1].set_xlabel("Share of the wind plant capacity [%]")
    axis[1, 1].set_ylabel("LCOH [EUR/kWh]")
    
    fig.suptitle("Evolution of the different variables and the LCOH in relation to the nuclear capacity determined as a share of the wind capacity")
    fig.show()

    return Cs, Ss, Ns, LCOHs
    
if __name__ == '__main__':
    res = main()