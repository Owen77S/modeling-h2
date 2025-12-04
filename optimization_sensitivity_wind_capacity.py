import multiprocessing
import optimization as m
import matplotlib.pyplot as plt
import numpy as np

def opt(WC):
    """
    Optimization of the H2 plant for a wind plant capacity of WC [MW]

    Parameters
    ----------
    WC : int
        Wind power capacity in MW.

    Returns
    -------
    best_configs_LCOHs_sorted : list
        

    """
    printed = 0
    nb = int(WC/3.3)
    
    if nb <= 0:
        print("Wind capacity too low or negative.")
        return None
    
    plant = m.H2plant()
    plant.get_data_from_excel(nb)
    plant.power_manager()
    print(f"nb turbines : {nb}, WC : {WC}, mean : {np.mean(plant.data['WP'])}")
    avg = np.mean(plant.data['excess_power'])
    
    # STEP 1
    
    fast = False
    
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
    
    C_b = best_configs_LCOHs_sorted[0]
    C_min, C_max = int(C_b/3), int(C_b*3/3)
    
    S_b = best_configs_LCOHs_sorted[1]
    S_r = S_max - S_min
    S_min, S_max = int(S_b - shr*S_r), int(S_b + shr*S_r)
    
    N_b = best_configs_LCOHs_sorted[2]
    N_r = N_max - N_min
    N_min, N_max = int(N_b - shr*N_r), int(N_b + shr*N_r)

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
    
    C_b = best_configs_LCOHs_sorted[0]
    C_r = C_max - C_min
    C_min, C_max = int(C_b - shr*C_r), int(C_b + shr*C_r)
    
    S_b = best_configs_LCOHs_sorted[1]
    S_r = S_max - S_min
    S_min, S_max = int(S_b - shr*S_r), int(S_b + shr*S_r)
    
    N_b = best_configs_LCOHs_sorted[2]
    N_r = N_max - N_min
    N_min, N_max = int(N_b - shr*N_r), int(N_b + shr*N_r)
    
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
    
    C_b = best_configs_LCOHs_sorted[0]
    C_r = C_max - C_min
    C_min, C_max = int(C_b - shr*C_r), int(C_b + shr*C_r)
    
    S_b = best_configs_LCOHs_sorted[1]
    S_r = S_max - S_min
    S_min, S_max = int(S_b - shr*S_r), int(S_b + shr*S_r)
    
    N_b = best_configs_LCOHs_sorted[2]
    N_r = N_max - N_min
    N_min, N_max = int(N_b - shr*N_r), int(N_b + shr*N_r)
    
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
    NC = 1450
    # percentage_of_NP_capacity = [1, 5, 10, 343.2/1450]
    # percentage_of_NP_capacity = [343.2/1450*100]
    # percentage_of_NP_capacity = [35, 50, 65]
    percentage_of_NP_capacity = [85, 100]
    WCs = [NC*p/100 for p in percentage_of_NP_capacity]

    cpt = 0
    
    Cs = []
    Ss = []
    Ns = []
    LCOHs = []
    EP = []
    for WC in WCs:
        print(f"{cpt+1}/{len(WCs)} iterations.")
        res, avg = opt(WC)
        Cs.append(res[0])
        Ss.append(res[1])
        Ns.append(res[2])
        LCOHs.append(res[3])
        EP.append(avg)
        cpt += 1
    
    # Results show 

    fig, axis = plt.subplots(2, 2)
    
    axis[0, 0].plot(percentage_of_NP_capacity, Cs)

    axis[0, 0].set_xlabel("Share of the nuclear plant capacity [%]")
    axis[0, 0].set_ylabel("Electrolyzer capacity[kW]")
    
    axis[0, 1].plot(percentage_of_NP_capacity, Ss)

    axis[0, 1].set_xlabel("Share of the nuclear plant capacity [%]")
    axis[0, 1].set_ylabel("Storage capacity [m3]")
    
    axis[1, 0].plot(percentage_of_NP_capacity, Ns)

    axis[1, 0].set_xlabel("Share of the nuclear plant capacity [%]")
    axis[1, 0].set_ylabel("Number of trucks")
    
    axis[1, 1].plot(percentage_of_NP_capacity, LCOHs)

    axis[1, 1].set_xlabel("Share of the nuclear plant capacity [%]")
    axis[1, 1].set_ylabel("LCOH [EUR/kWh]")
    
    fig.suptitle("Evolution of the different variables and the LCOH in relation to the nuclear capacity determined as a share of the nuclear capacity")
    fig.show()
    
    fig, ax = plt.subplots()
    
    ax.plot(WCs, EP, label = "Average excess power obtained for each different wind plant")
    ax.legend()
    
    fig.show()

    return Cs, Ss, Ns, LCOHs, WCs, EP

    
if __name__ == '__main__':
    res = main()
    print(res)