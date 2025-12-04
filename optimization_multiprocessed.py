import multiprocessing
import optimization as m
import matplotlib.pyplot as plt
import numpy as np

def main():
    
    # STEP 1
    n_best_design = 5
    n_iter = 20
    len_pop = 70
    # n_best_design = 5
    # n_iter = 20
    # len_pop = 90
    # len_pop = 100

    plant = m.empty_plant(int(0.85*1450/3.3))
    plant.power_manager()
    
    C_min = 10000
    C_max = int(max(plant.data['excess_power']))
    
    if C_min >= C_max:
        C_min = C_max/8

    S_min = 500
    S_max = 1500
    N_min = 30
    N_max = 50

    nb_proc = 14
    
    inputs = (plant, n_best_design, n_iter, len_pop, C_min, C_max, S_min, S_max, N_min, N_max)
    args = [inputs]*nb_proc

    # Multiprocessing
    
    print("Step 1")
    with multiprocessing.Pool(processes=nb_proc) as pool:
        results = pool.starmap_async(m.optimization, args)
        # results = pool.starmap_async(m.optimization2, args)
        results = results.get()
    
    # Result processing
    
    Cs = []
    Ss = []
    Ns = []
    LCOHs = []
    
    best_configs_LCOHs = []
    
    for res in results:
        listes = res[0]
        Cs += listes[0]
        Ss += listes[1]
        Ns += listes[2]
        LCOHs += listes[3]
        
        best_configs_LCOHs.append(res[1])

    # New boundaries
    best_configs_LCOHs_sorted = sorted(best_configs_LCOHs, key=lambda x:x[-1])
    
    best_configs_LCOHs_sorted = best_configs_LCOHs_sorted[0]

    shr = 0.2
    C_min_old, C_max_old = C_min, C_max
    S_min_old, S_max_old = S_min, S_max
    N_min_old, N_max_old = N_min, N_max
    
    C_b = best_configs_LCOHs_sorted[0]
    C_r = C_max - C_min
    C_min, C_max = int(C_b/2), int(3*C_b/2)
    
    S_b = best_configs_LCOHs_sorted[1]
    S_r = S_max - S_min
    S_min, S_max =  max(int(S_b - S_r*shr), S_min_old), min(int(S_b + S_r*shr), S_max_old)
    
    N_b = best_configs_LCOHs_sorted[2]
    N_r = N_max-N_min
    N_min, N_max =  max(int(N_b - N_r*shr), N_min_old) , min(int(N_b + N_r*shr), N_max_old) 

    LCOH_b = best_configs_LCOHs_sorted[3]
    
    fig, axis = plt.subplots(2, 2)
    
    axis[0, 0].boxplot(Cs)
    axis[0, 0].set_title("Boxplot of the capacity")
    axis[0, 0].set_ylabel("Capacity [kW]")
    axis[0, 0].axhline(y=C_min, color='r', linestyle='--', label=f'New minimum : {C_min}')
    axis[0, 0].axhline(y=C_max, color='r', linestyle='--', label=f'New maximum : {C_max}')
    axis[0, 0].axhline(y=C_b, color='g', linestyle='--', label=f'Best capacity : {C_b}')
    
    axis[0, 0].axhline(y=C_min_old, color='black', linestyle='--', label=f'Actual minimum : {C_min_old}')
    axis[0, 0].axhline(y=C_max_old, color='black', linestyle='--', label=f'Actual maximum : {C_max_old}')
    
    axis[0, 0].legend()
    
    axis[0, 1].boxplot(Ss)
    axis[0, 1].set_title("Boxplot of the storage capacity")
    axis[0, 1].set_ylabel("Storage [m3]")
    axis[0, 1].axhline(y=S_min, color='r', linestyle='--', label=f'New minimum : {S_min}')
    axis[0, 1].axhline(y=S_max, color='r', linestyle='--', label=f'New maximum : {S_max}')
    axis[0, 1].axhline(y=S_b, color='g', linestyle='--', label=f'Best capacity : {S_b}')
    
    axis[0, 1].axhline(y=S_min_old, color='black', linestyle='--', label=f'Actual minimum : {S_min_old}')
    axis[0, 1].axhline(y=S_max_old, color='black', linestyle='--', label=f'Actual maximum : {S_max_old}')
    
    axis[0, 1].legend()
    
    axis[1, 0].boxplot(Ns)
    axis[1, 0].set_title("Boxplot of the number of trucks")
    axis[1, 0].set_ylabel("Number of trucks")
    axis[1, 0].axhline(y=N_min, color='r', linestyle='--', label=f'New minimum : {N_min}')
    axis[1, 0].axhline(y=N_max, color='r', linestyle='--', label=f'New maximum : {N_max}')
    axis[1, 0].axhline(y=N_b, color='g', linestyle='--', label=f'Best capacity : {N_b}')
    
    axis[1, 0].axhline(y=N_min_old, color='black', linestyle='--', label=f'Actual minimum : {N_min_old}')
    axis[1, 0].axhline(y=N_max_old, color='black', linestyle='--', label=f'Actual maximum : {N_max_old}')
    
    axis[1, 0].legend()
    
    axis[1, 1].boxplot(LCOHs)
    axis[1, 1].set_title("Boxplot of the LCOH")
    axis[1, 1].axhline(y=LCOH_b, color='g', linestyle='--', label=f'Best LCOH : {LCOH_b}')
    axis[1, 1].set_ylabel("LCOH")
    axis[1, 1].legend()
   
    fig.suptitle("Boxplots step 1")
    fig.show()

    # STEP 2
    # n_best_design = 1
    # n_iter = 1
    # len_pop = 20
    n_best_design = 5
    n_iter = 20
    len_pop = 50
    
    inputs = (plant, n_best_design, n_iter, len_pop, C_min, C_max, S_min, S_max, N_min, N_max)
    args = [inputs]*nb_proc

    # Multiprocessing
    
    print("Step 2")
    with multiprocessing.Pool(processes=nb_proc) as pool2:
        results2 = pool2.starmap_async(m.optimization, args)
        # results2 = pool2.starmap_async(m.optimization2, args)
        results2 = results2.get()
        
    # Result processing
    
    Cs = []
    Ss = []
    Ns = []
    LCOHs = []
    
    best_configs_LCOHs = []
    
    for res in results2:
        listes = res[0]
        Cs += listes[0]
        Ss += listes[1]
        Ns += listes[2]
        LCOHs += listes[3]
        
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

    LCOH_b = best_configs_LCOHs_sorted[3]
    
    fig, axis = plt.subplots(2, 2)
    
    axis[0, 0].boxplot(Cs)
    axis[0, 0].set_title("Boxplot of the capacity")
    axis[0, 0].set_ylabel("Capacity [kW]")
    axis[0, 0].axhline(y=C_min, color='r', linestyle='--', label=f'New minimum : {C_min}')
    axis[0, 0].axhline(y=C_max, color='r', linestyle='--', label=f'New maximum : {C_max}')
    axis[0, 0].axhline(y=C_b, color='g', linestyle='--', label=f'Best capacity : {C_b}')
    
    axis[0, 0].axhline(y=C_min_old, color='black', linestyle='--', label=f'Actual minimum : {C_min_old}')
    axis[0, 0].axhline(y=C_max_old, color='black', linestyle='--', label=f'Actual maximum : {C_max_old}')
    
    axis[0, 0].legend()
    
    axis[0, 1].boxplot(Ss)
    axis[0, 1].set_title("Boxplot of the storage capacity")
    axis[0, 1].set_ylabel("Storage [m3]")
    axis[0, 1].axhline(y=S_min, color='r', linestyle='--', label=f'New minimum : {S_min}')
    axis[0, 1].axhline(y=S_max, color='r', linestyle='--', label=f'New maximum : {S_max}')
    axis[0, 1].axhline(y=S_b, color='g', linestyle='--', label=f'Best capacity : {S_b}')
    
    axis[0, 1].axhline(y=S_min_old, color='black', linestyle='--', label=f'Actual minimum : {S_min_old}')
    axis[0, 1].axhline(y=S_max_old, color='black', linestyle='--', label=f'Actual maximum : {S_max_old}')
    
    axis[0, 1].legend()
    
    axis[1, 0].boxplot(Ns)
    axis[1, 0].set_title("Boxplot of the number of trucks")
    axis[1, 0].set_ylabel("Number of trucks")
    axis[1, 0].axhline(y=N_min, color='r', linestyle='--', label=f'New minimum : {N_min}')
    axis[1, 0].axhline(y=N_max, color='r', linestyle='--', label=f'New maximum : {N_max}')
    axis[1, 0].axhline(y=N_b, color='g', linestyle='--', label=f'Best capacity : {N_b}')
    
    axis[1, 0].axhline(y=N_min_old, color='black', linestyle='--', label=f'Actual minimum : {N_min_old}')
    axis[1, 0].axhline(y=N_max_old, color='black', linestyle='--', label=f'Actual maximum : {N_max_old}')
    
    axis[1, 0].legend()
    
    axis[1, 1].boxplot(LCOHs)
    axis[1, 1].set_title("Boxplot of the LCOH")
    axis[1, 1].axhline(y=LCOH_b, color='g', linestyle='--', label=f'Best LCOH : {LCOH_b}')
    axis[1, 1].set_ylabel("LCOH")
    axis[1, 1].legend()
   
    fig.suptitle("Boxplots step 2")
    fig.show()
    
    # STEP 3
    
    # n_best_design = 1
    # n_iter = 1
    # len_pop = 20
    n_best_design = 6
    n_iter = 25
    len_pop = 70
    
    inputs = (plant, n_best_design, n_iter, len_pop, C_min, C_max, S_min, S_max, N_min, N_max)
    args = [inputs]*nb_proc

    # Multiprocessing
    
    print("Step 3")
    with multiprocessing.Pool(processes=nb_proc) as pool3:
        results3 = pool3.starmap_async(m.optimization, args)
        # results3 = pool3.starmap_async(m.optimization2, args)
        results3 = results3.get()
    
    # Result processing
    
    Cs = []
    Ss = []
    Ns = []
    LCOHs = []
    
    best_configs_LCOHs = []
    
    for res in results3:
        listes = res[0]
        Cs += listes[0]
        Ss += listes[1]
        Ns += listes[2]
        LCOHs += listes[3]
        
        best_configs_LCOHs.append(res[1])
    
    # New boundaries
    best_configs_LCOHs_sorted = sorted(best_configs_LCOHs, key=lambda x:x[-1])
    best_configs_LCOHs_sorted=best_configs_LCOHs_sorted[0]

    C_b = best_configs_LCOHs_sorted[0]
    
    S_b = best_configs_LCOHs_sorted[1]  
    
    N_b = best_configs_LCOHs_sorted[2]

    fig, axis = plt.subplots(2, 2)
    
    axis[0, 0].boxplot(Cs)
    axis[0, 0].set_title("Boxplot of the capacity")
    axis[0, 0].set_ylabel("Capacity [kW]")
    axis[0, 0].axhline(y=C_b, color='g', linestyle='--', label=f'Best capacity : {C_b}')
    axis[0, 0].legend()
    
    axis[0, 1].boxplot(Ss)
    axis[0, 1].set_title("Boxplot of the storage capacity")
    axis[0, 1].set_ylabel("Storage [m3]")
    axis[0, 1].axhline(y=S_b, color='g', linestyle='--', label=f'Best capacity : {S_b}')
    axis[0, 1].legend()
    
    axis[1, 0].boxplot(Ns)
    axis[1, 0].set_title("Boxplot of the number of trucks")
    axis[1, 0].set_ylabel("Number of trucks")
    axis[1, 0].axhline(y=N_b, color='g', linestyle='--', label=f'Best capacity : {N_b}')
    axis[1, 0].legend()
    
    axis[1, 1].boxplot(LCOHs)
    axis[1, 1].set_title("Boxplot of the LCOH")
    axis[1, 1].set_ylabel("LCOH")
    axis[1, 1].axhline(y=best_configs_LCOHs_sorted[3], color='g', linestyle='--', label=f'Best LCOH : {best_configs_LCOHs_sorted[3]}')
    axis[1, 1].legend()
    
    fig.suptitle("Boxplots step 3")
    fig.show()
    
    # Best configs
    
    print(f"Best C : {C_b} kW")
    print(f"Best S : {S_b} m3")
    print(f"Best N : {N_b}")
    print(f"Best LCOH : {best_configs_LCOHs_sorted[3]} EUR/kW")
    

if __name__ == '__main__':
    main()