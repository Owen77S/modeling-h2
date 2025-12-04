# REMAINING 

def sensitivity_analysis(plant, param_analysis, mini, maxi):
    """
    To conduct a sensitivity analysis in relation to a parameter. 100 different
    values will be taken by the parameter. The function will plot the influence
    of this analysis on all the KPIs.
    Return the list of lists of KPIs.

    Parameters
    ----------
    plant : H2plant
        Hydrogen plant with data (get_data_from_excel or from_python).
    param_analysis : str
        The parameter in relation to which the analysis will be conducted.
    mini : int
        The minimum value taken by the parameter.
    maxi : TYPE
        The maximum value taken by the parameter.

    Returns
    -------
    KPIs : list
        List of lists of KPIs.
        KPIs = [[LCOH_1, H2_1, ..., %time_full_1"], [LCOH_2, H2_2, ..., %time_full_2"]]

    """
    name_KPIs = ["LCOH", "H2", "wasted_power", "benefit", "wasted_hydrogen", "%time_storage_full"]
    KPIs = [[], [], [], [], [], []]

    params = np.linspace(mini, maxi, 10)
    compteur = 1
    for param in params:
        plant.param[param_analysis] = param
        plant.power_manager()
        plant.electrolyzer_production()
        plant.hydrogen_management()
        plant.get_KPI()
        for i in range(6):
            KPIs[i].append(plant.KPI[name_KPIs[i]])
       
        if compteur%20 == 0:
            print(compteur, "% complété(s).")
        compteur += 1

    figure, axis = plt.subplots(3, 2)
    for i in range(6):
        KPIi = KPIs[i]
        axis[i//2, i%2].plot(params, KPIi)
    
    plt.show()
    return KPIs


def pareto(v_min, v_max, nb):
    plant = empty_plant()
    
    # C = 20000
    # plant.set_electrolyzer_capacity(C)
    
    S = 1e3
    plant.set_storage_capacity(S)
    
    N = 200
    plant.set_number_of_trucks(N)
    
    variables = np.arange(v_min, v_max, nb)
    LCOHs = []
    wasted_H2s = []
    wasted_powers = []
    
    for val in variables:
        plant.set_electrolyzer_capacity(val)
        
        plant.electrolyzer_production()
        plant.hydrogen_management()
        plant.get_KPI()
        
        LCOHs.append(plant.KPI['LCOH'])
        wasted_H2s.append(plant.KPI['wasted_hydrogen'])
        wasted_powers.append(plant.KPI['wasted_power'])
    
    # Plots
    
    fig, ax = plt.subplots(1, 2)
    
    ax[0].plot(variables, LCOHs)
    ax[0].set_title('LCOH in relation to the electrolyser capacity')
    ax[0].set_xlabel('Capacity [kW]')
    ax[0].set_ylabel('LCOH [EUR/kWh]')
    
    ax[1].plot(wasted_H2s, wasted_powers)     

def optimisation(plant, mini_electrolyzer, maxi_electrolyzer, incr, min_storage, max_storage, want_progression=1):
    
    total_H2_list = []
    total_LCOH = []
    
    capacities = range(mini_electrolyzer, maxi_electrolyzer, incr)
    cpt = 0
    
    storages = range(min_storage, max_storage, 5)
    
    fig = plt.figure() 
    ax = fig.add_subplot(111, projection='3d')
    "To prepare the 3D graph"
    
    for capacity in capacities:
        
        total_H2_wasted_list = []
        total_LCOH = []
        total_wasted_power = []
        "To reset our variables"
        plant.set_electrolyzer_capacity(capacity)
        
        for storage in storages: 
            plant.set_storage_capacity(storage)
            plant.electrolyzer_production()
            plant.hydrogen_management()
            plant.get_KPI()
            total_H2_wasted_list.append(plant.KPI["wasted_hydrogen"])
            total_LCOH.append(plant.KPI['LCOH'])
        
        
        total_H2_wasted_array = np.array(total_H2_wasted_list)
        total_LCOH_array = np.array(total_LCOH)
        total_wasted_power_array = np.array(total_wasted_power)
        'Convertir les listes en tableaux numpy'
        
        X = total_H2_wasted_array.reshape(-1, 1)
        Y = total_LCOH_array.reshape(-1, 1)
        Z = total_wasted_power_array.reshape(-1, 1)
        'Remodeler les tableaux en tableaux bidimensionnels'
        
        X, Y = np.meshgrid(X, Y)
        ax.plot_surface(Z, Y, X, cmap='viridis')
        'cmap for the graph color'
        
        if want_progression == 1:
            cpt += 1
            if cpt%10 == 0:
                print(100*cpt/len(capacities), "% completed.")
            
            
    # plt.xlabel("LCOH [€/kWh]")
    # plt.ylabel("Total Hydrogen wasted [kg]")
    ax.set_xlabel('Total Hydrogen wasted [kg]')
    ax.set_ylabel('LCOH [€/kWh]')
    ax.set_zlabel('Total Power wasted [kW]')
    fig.show()    
        # plt.figure()
        # plt.xlabel("Capacity [kW]")
        # plt.ylabel("LCOH [€/kWh]")
        # plt.plot(capacities, total_LCOH)
        
        # plt.figure()
        # plt.xlabel("LCOH [€/kWh]")
        # plt.ylabel("Total hydrogen produced [kg]")
        # plt.plot(total_LCOH, total_H2_list,'+')
    
    # # Selection rule for the best capacity
    # for i in range(len(total_H2_list)):
    #     if total_H2_list[i] == max(total_H2_list):
    #         design_capacity = capacities[i]
    #         break
    # return design_capacity

# For sensitivity analysis