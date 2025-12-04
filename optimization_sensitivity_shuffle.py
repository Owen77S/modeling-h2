import multiprocessing
import optimization as m
import matplotlib.pyplot as plt
import numpy as np

def opt(what):
    
    printed = 0
    
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
        len_pop = 50
        nb_proc = 14
    
    C_min = 1
    S_min = 10
    S_max = 2500
    N_min = 1
    N_max = 200

    inputs = (what, n_best_design, n_iter, len_pop, C_min, S_min, S_max, N_min, N_max, printed=1)
    args = [inputs]*nb_proc
    
    # Multiprocessing
    
    print("Step 1")
    with multiprocessing.Pool(processes=nb_proc) as pool:
        results = pool.starmap_async(m.optimization_shuffle, args)
        results = results.get()
    
    # Result processing
    
    best_configs_LCOHs = []
    
    for res in results:
        best_configs_LCOHs.append(res[1])
    
    # New boundaries
    best_configs_LCOHs_sorted = sorted(best_configs_LCOHs, key=lambda x:x[-1])
    best_configs_LCOHs_sorted = best_configs_LCOHs_sorted[0]
    
    shr = 20/100
    
    C_b = best_configs_LCOHs_sorted[0]
    C_r = C_max - C_min
    C_min, C_max = int(C_b/2), int(3*C_b/2)
    
    S_b = best_configs_LCOHs_sorted[1]
    S_r = S_max - S_min
    S_min, S_max =  int(S_b - S_r*shr), int(S_b + S_r*shr)
    
    N_b = best_configs_LCOHs_sorted[2]
    N_r = N_max-N_min
    N_min, N_max =  int(N_b - N_r*shr), int(N_b + N_r*shr) 
    
    # STEP 2
    
    inputs = (plant, n_best_design, n_iter, len_pop, C_min, C_max, S_min, S_max, N_min, N_max, printed)
    args = [inputs]*nb_proc
    
    # Multiprocessing
    
    print("Step 2")
    with multiprocessing.Pool(processes=nb_proc) as pool2:
        results2 = pool2.starmap_async(m.optimization_shuffle, args)
        results2 = results2.get()
    
    # Result processing
    
    best_configs_LCOHs = []
    
    for res in results2:
        best_configs_LCOHs.append(res[1])
    
    # New boundaries
    best_configs_LCOHs_sorted = sorted(best_configs_LCOHs, key=lambda x:x[-1])
    best_configs_LCOHs_sorted = best_configs_LCOHs_sorted[0]
    
    C_b = best_configs_LCOHs_sorted[0]
    C_r = C_max - C_min
    C_min, C_max = int(C_b - C_r*shr), int(C_b + C_r*shr)
    
    S_b = best_configs_LCOHs_sorted[1]
    S_r = S_max - S_min
    S_min, S_max =  int(S_b - S_r*shr), int(S_b + S_r*shr)
    
    N_b = best_configs_LCOHs_sorted[2]
    N_r = N_max-N_min
    N_min, N_max =  int(N_b - N_r*shr), int(N_b + N_r*shr)  
    
    # STEP 3
 
    inputs = (plant, n_best_design, n_iter, len_pop, C_min, C_max, S_min, S_max, N_min, N_max, printed)
    args = [inputs]*nb_proc
    
    # Multiprocessing
    
    print("Step 3")
    with multiprocessing.Pool(processes=nb_proc) as pool3:
        results3 = pool3.starmap_async(m.optimization_shuffle, args)
        results3 = results3.get()
    
    # Result processing
    
    best_configs_LCOHs = []
    
    for res in results3:
        best_configs_LCOHs.append(res[1])
    
    # New boundaries
    best_configs_LCOHs_sorted = sorted(best_configs_LCOHs, key=lambda x:x[-1])
    best_configs_LCOHs_sorted= best_configs_LCOHs_sorted[0]
    
    C_b = best_configs_LCOHs_sorted[0]
    C_r = C_max - C_min
    C_min, C_max = int(C_b - C_r*shr), int(C_b + C_r*shr)
    
    S_b = best_configs_LCOHs_sorted[1]
    S_r = S_max - S_min
    S_min, S_max =  int(S_b - S_r*shr), int(S_b + S_r*shr)
    
    N_b = best_configs_LCOHs_sorted[2]
    N_r = N_max-N_min
    N_min, N_max =  int(N_b - N_r*shr), int(N_b + N_r*shr)  
    
    # STEP 4
  
    inputs = (plant, n_best_design, n_iter, len_pop, C_min, C_max, S_min, S_max, N_min, N_max, printed)
    args = [inputs]*nb_proc
    
    # Multiprocessing
    
    print("Step 4")
    with multiprocessing.Pool(processes=nb_proc) as pool4:
        results4 = pool4.starmap_async(m.optimization_shuffle, args)
        results4 = results4.get()
    
    # Result processing
    
    best_configs_LCOHs = []
    
    for res in results4:
        best_configs_LCOHs.append(res[1])
    
    # New boundaries
    best_configs_LCOHs_sorted = sorted(best_configs_LCOHs, key=lambda x:x[-1])
    best_configs_LCOHs_sorted=best_configs_LCOHs_sorted[0]
    
    return best_configs_LCOHs_sorted



def main(what):
    nb = 20
    
    Cs = []
    Ss = []
    Ns = []
    LCOHs = []
    
    for i in range(nb):
        print(f"{i+1}/{nb} iterations.")
        res = opt(what)
        Cs.append(res[0])
        Ss.append(res[1])
        Ns.append(res[2])
        LCOHs.append(res[3])
        
    
    # Results show 
    fig, axis = plt.subplots(2, 2)
    
    axis[0, 0].boxplot(Cs)

    axis[0, 0].set_ylabel("Electrolyzer capacity [kW]")
    
    axis[0, 1].boxplot(Ss)

    axis[0, 1].set_ylabel("Storage capacity [m3]")
    
    axis[1, 0].boxplot(Ns)
    
    axis[1, 0].set_ylabel("Number of trucks")
    
    axis[1, 1].boxplot(LCOHs)

    axis[1, 1].set_ylabel("LCOH [EUR/kWh]")
    
    fig.suptitle("Evolution of the different variables and the LCOH while shuffling the yield power from the nuclear plant")
    fig.show()

    return Cs, Ss, Ns, LCOHs
    
if __name__ == '__main__':
    res = main("NP")
    print("NP", res)
    res = main("WP")
    print("WP", res)