import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt
import random

class H2plant():
    
    def __init__(self):
        
        self.duration = 8760
        
        self.data = {
            "WP" : [0]*self.duration, #List containing yield power from wind plant [kW]
            "NP" : [0]*self.duration, #List containing yield power from nuclear power plant [kW]
            "excess_power" : [0]*self.duration, #List containing excess power (risk of congestion) [kW]
            "supply_power"  : [0]*self.duration #List containing supplied power to the electrolyzer [kW]
            } 
        
        self.economics = {
            "install_fee" : 1800, #of the electrolyzer €/kW
            "OPEX_PEM" : 54, #of the PEM €/kW/year
            "water_price" : 3e-3, #water price €/kg
            "water_consumption" : 9, #water consumption kG H2O/kg H2
            "price_H2" : 2.7
            } #hydrogen price €/kg
        
        self.res = {
            "mass_H2" : [0]*self.duration, #List containing mass of hydrogen produced per hour [kg]
            "H2" : [0]*self.duration, #List containing hydrogen produced per hour [m^3]
            "H2_compressed" : [0]*self.duration, #List containing hydrogen produced per hour, after compression [m^3]
            "stored" : [0]*self.duration, #List containing hydrogen stored in the storage, after selling it, per hour [m^3]
            "wasted" : [0]*self.duration, #List containing hydrogen that can't be stored per hour [m^3]
            "sold" : [0]*self.duration #List containing hydrogen sold per hour [m^3]
            }
        
        self.KPI = {
            "LCOH" : 0, #LCOH [€]
            "H2" : 0, #Total hydrogen produced over a year [kg]
            "H2_used" : 0, #Total hydrogen effectively used over a year [kg]
            "wasted_power" : 0, #Share of the excess power wasted due to an undersized electrolyzer [kW]
            "benefit" : 0, #Benefit from selling hydrogen [€]
            "wasted_hydrogen" : 0, #%of the total hydrogen that couldn't be stored (when storage full) [%]
            "%time_storage_full" : 0 #% of time when the storage is full [%]
            }
        
        self.param = {
            "eta_F_characteristic" : 0.04409448818, #Characteristic of the electrolyzer
            "grid_limit" : 1319414, #Grid limit [kW]
            "LHV_kg" : 33.3, #kWh/kg
            "LHV_NM3" : 3, #kWh/Nm3
            "truck_capacity" : 29.36, #Amount of hydrogen a truck can contain [m^3]
            "unavailable_hours" : 3, #Time for a truck to go back and forth (so we can't sell any hydrogen while trucks are not back) [hour]
            "H2_price" : 2.7, #Hydrogen price [€/kg]
            "NM3_to_kg" : 101325*2e-3/(8.314*273.15)
            } 
        
        self.gas_model = {
            "T_op" : 80, #Operating temperature (outlet of PEM) [°C]
            "P_op" : 15, #Operating pressure (outlet of PEM) [bar]
            "M" : 2e-3, #Molecular mass of hydrogen (= dihydrogen) [kg/mol]
            "R" : 8.314, #Ideal gas constant
            "T_c" : -240, #Critical temperature of hydrogen [°C]
            "P_c" : 130, #Critical pressure of hydrogen [bar]
            "n" : 1.4, #cp/cv
            "P_out" : 250 #Desired output pressure [bar]
            }
        
        self.var = {
            "electrolyzer_capacity" : 0, #Electrolyzer capacity [kW]
            "storage_capacity" : 0, #Storage capacity [m^3]
            "number_of_trucks" : 0, #Number of trucks 
            "threshold" : 0 #Threshold above which we start selling hydrogen 0 < T < 1 [1]
            }
        
    def get_data_from_excel(self, nb = 104, path_excel = "data_2.xlsx"):
        '''
        Get the parameters (wind and nuclear power plant production) from Excel
        Update self.data
        Parameters
        ----------
        nb : int
            The number of wind turbines in the wind plant.
            
        path_excel : str, optional
            The excel file path. The default is "data_2.xlsx".

        Returns
        -------
        None.

        '''
        
        wind_nuclear = pd.read_excel("data_2.xlsx",
                                     usecols = 'A:B')
        
        self.data["WP"] = [nb*power for power in wind_nuclear["Wind power [kW]"].tolist()]
        self.data["NP"] = wind_nuclear['Nuclear power plant [kW]'].tolist()
        
    def get_data_from_python(self, WP, NP):
        '''
        Get the parameters (wind and nuclear power plant production) from Excel

        Parameters
        ----------
        WP : List
            List containing the yield power of the wind power plant each hour for A YEAR.
        NP : List
            List containing the yield power of the nuclear power plant each hour for A YEAR.

        Returns
        -------
        None.

        '''
        if len(WP) != self.duration or len(NP) != self.duration:
            print("Error : the yield production is not for a year.")
            return
        self.data["WP"], self.data["NP"] = WP, NP
    
    def power_manager(self):
        """
        Get the excess electricity. Need to have the nuclear and wind
        yield production already defined.
        
        -------
        None.

        """

        for t in range(self.duration):
            excess_electricity = self.data["WP"][t] + self.data["NP"][t] - self.param['grid_limit']
            if excess_electricity > 0:
                #We produce to much energy : congestion
                self.data["excess_power"][t] = excess_electricity
            else:
                #We don't produce to much energy : no congestion
                self.data["excess_power"][t] = 0
    
    def n_F(self, electrolyzer_capacity, supply_power):
        """
        To compute faradic efficiency.

        Parameters
        ----------
        electrolyzer_capacity : int
            The electrolyzer_capacity of the electrolyzer [kW].
        supply_power : int
            The supply power to the electrolyzer [kW].

        Returns
        -------
        n_F
            The faradic efficiency [%].

        """
        return 1 - np.exp(-(supply_power/electrolyzer_capacity)/self.param["eta_F_characteristic"])
    
    def electrolyzer_production(self):
        """
        Determine the hourly hydrogen production and the total hydrogen produced throughout
        the year by the electrolyzer of a specific electrolyzer capacity.        

        Parameters
        ----------
        None.

        Returns
        -------
        None.

        """
        #Auxiliaries power requirement : 3% of the power supply
        n_aux = 1 - 0.03
        for t in range(self.duration):
            if self.data["excess_power"][t] >= self.var["electrolyzer_capacity"]:
                #The supply power can't excess the electrolyzer's capacity
                self.data["supply_power"][t] = self.var["electrolyzer_capacity"]
            else:
                self.data["supply_power"][t] = self.data["excess_power"][t]
            
            mass_H2 = n_aux*self.n_F(self.var["electrolyzer_capacity"], self.data["supply_power"][t])*self.data["supply_power"][t]/self.param['LHV_kg']
            self.res["mass_H2"][t] = mass_H2
            # Ideal gas law
            self.res["H2"][t] = mass_H2*self.gas_model["R"]*(self.gas_model["T_op"]+273.15)/(self.gas_model["M"]*self.gas_model["P_op"]*10**5)
            # Compression
            self.res["H2_compressed"][t] = self.res["H2"][t]*(self.gas_model["P_op"]/self.gas_model["P_out"])**(1/self.gas_model["n"])
        
    def hydrogen_management(self):
        """
        To manage the produced hydrogen in the storage part and the selling part.

        Returns
        -------
        None.

        """
        list_of_unavailabilities = []
        trucks_available = self.var['number_of_trucks']
        available_next_hour = 0
        
        # breakpoint()
        #We assume that we don't sell any hydrogen during the first hourE = define_plant_fast(2e3, 25, 10, 1)E = define_plant_fast(2e3, 25, 10, 1)
        self.res['stored'][0] = min(self.res['H2_compressed'][0], self.var["storage_capacity"])
        # breakpoint()
        for t in range(1, self.duration):
            #Initialisation
            share_wasted = 0
            share_stored = 0
            quantity_sold = 0
            
            # Storage part
            availability_in_storage = self.var["storage_capacity"] - self.res["stored"][t-1]

            if self.res["H2_compressed"][t] > availability_in_storage:
                share_wasted = self.res["H2_compressed"][t] - availability_in_storage
                share_stored = availability_in_storage               
            
            else:
                share_stored = self.res['H2_compressed'][t]
                share_wasted = 0
            
            self.res["wasted"][t] = share_wasted  
            self.res["stored"][t] = self.res["stored"][t-1] + share_stored
 
            if self.res["stored"][t] >= self.var["storage_capacity"]*self.var['threshold'] and trucks_available > 0:
                # We sell
                # We fill the *maximum* of trucks                
                #STORAGE CAPACITY HAVE TO BE GREATER THAN A TRUCK CAPACITY
                number_of_trucks_used = min(self.res['stored'][t]//self.param['truck_capacity'], trucks_available)
                trucks_available -= number_of_trucks_used
                quantity_sold = number_of_trucks_used*self.param['truck_capacity']
                list_of_unavailabilities.append([number_of_trucks_used, self.param["unavailable_hours"]])
                self.res["sold"][t] = quantity_sold
                self.res["stored"][t] = self.res["stored"][t] - quantity_sold
                
            #We will be able to sold when the truck will comeback
            if len(list_of_unavailabilities) > 0:
                for i in range(len(list_of_unavailabilities)):
                    # If the first fleet of trucks come in one hour, then they are available in the next hour
                    if list_of_unavailabilities[0][1] == 1:
                        available_next_hour = 1
                    # We reduce their unavailability by one hour
                    list_of_unavailabilities[i][1] -= 1

            
            if available_next_hour:
                # If the first fleet of trucks is available, we remove them from the list
                trucks_available += list_of_unavailabilities[0][0]
                list_of_unavailabilities.pop(0)
                available_next_hour = 0
    
 
    def get_KPI(self):
        """
        Update KPIs

        Returns
        -------
        None.

        """
        # Conversion to kg
        self.KPI['H2'] = sum(self.res["mass_H2"])
        self.KPI["H2_used"] = ((self.gas_model['P_out']*1e5)*sum(self.res["sold"])*self.gas_model['M'])/(self.gas_model['R']*(self.gas_model['T_op']+273.15)) #in kg
        
        storage_in_kg = ((self.gas_model['P_out']*1e5)*self.var['storage_capacity']*self.gas_model['M'])/(self.gas_model['R']*(self.gas_model['T_op']+273.15))
        CAPEX_PEM = self.economics['install_fee']*self.var['electrolyzer_capacity']
        CAPEX_storage = 490*storage_in_kg
        CAPEX_selling = 93296 + self.var["number_of_trucks"]*610000
        CAPEX = CAPEX_PEM + CAPEX_storage + CAPEX_selling
        
        OPEX_PEM = self.economics['OPEX_PEM']*self.var['electrolyzer_capacity'] 
        water_price = self.economics['water_price']*self.economics["water_consumption"]*self.KPI['H2']
        OPEX_compressor = 4665
        OPEX_selling = 30500*self.var['number_of_trucks']
        OPEX = OPEX_PEM + water_price + OPEX_compressor + OPEX_selling
        
        energy_in_H2 = self.KPI['H2_used']*self.param['LHV_kg'] #kWh
        
        self.KPI['LCOH'] = (CAPEX+OPEX)/energy_in_H2

        self.KPI["wasted_power"] = 1- sum(self.data["supply_power"])/sum(self.data["excess_power"])
        self.KPI["benefit"] = self.param["H2_price"]*self.KPI['H2_used']
        self.KPI["wasted_hydrogen"] = sum(self.res["wasted"])/sum(self.res['H2_compressed'])
        self.KPI["%time_storage_full"] = sum([self.res['stored'][t] == self.var["storage_capacity"] for t in range(self.duration)])/self.duration
  
    def objective(self, C, S, N, T):
        """
        To compute the objective function : LCOH

        Parameters
        ----------
        C : int
            Electrolyzer capacity [kW].
        S : int
            Storage capacity [m3].
        N : int
            Number of trucks.
        T : float
            Threshold.

        Returns
        -------
        None.

        """
        self.clear()
        
        self.set_electrolyzer_capacity(C)
        self.set_storage_capacity(S)
        self.set_number_of_trucks(N)
        self.set_threshold(T)
        
        self.electrolyzer_production()
        self.hydrogen_management()
        self.get_KPI()
        
        return self.KPI['LCOH']       
    
    def constraints(self):
        """
        To verify whether the constraints are verified or not.
        Assume that the plant parameters have already been defined,
        and the KPIs are calculated.
        The first constraint is KPI["wasted_power"] < 70%.
        The second constraint is KPI["wasted_hydrogen"] < 70% 

        Parameters
        ----------
        None
        
        Returns
        -------
        verif_WP : boolean.
            True : if the wasted power constraint is verified
            False otherwise.
            
        verif_WH : boolean.
            True : if the wasted hydrogen constraint is verified
            False otherwise.

        """
        verif_WP = 0 if self.KPI["wasted_power"] < 0.8 else 1
        verif_WH = 0 if self.KPI["wasted_hydrogen"] < 0.8 else 1
        
        return verif_WP, verif_WH
        
    
    def shuffle(self, what):
        """
        Shuffle the yield power from the wind and the nuclear power plant.
        For sensitivity analysis.
        
        Parameters
        ----------
        what : str. Default value : "all".
            If what = "all" : we shuffle both
        Returns
        -------
        None.
        """
        if what == "WP":
            random.shuffle(self.data["WP"])
        elif what == "NP":
            random.shuffle(self.data["NP"])
        else:
            print("No shuffle has been done. The argument should either be 'NP' or 'WP'.")
    
    def change_grid_limit(self, new_grid_limit):
        """
        To change the grid limit.
        Parameters
        ----------
        new_grid_limit : int
            The new grid limit, in kW.
        Returns
        -------
        None.
        """
        self.param["grid_limit"] = new_grid_limit
  
    def set_electrolyzer_capacity(self, C):
        self.var["electrolyzer_capacity"] = C
    
    def set_storage_capacity(self, C):
        self.var["storage_capacity"] = C
        
    def set_number_of_trucks(self, N):
        self.var["number_of_trucks"] = N
        
    def set_threshold(self, T):
        self.var['threshold'] = T
    
    def clear(self):
        self.data["supply_power"] = [0]*self.duration 
        
        self.res = {
            "mass_H2" : [0]*self.duration, #List containing mass of hydrogen produced per hour [kg]
            "H2" : [0]*self.duration, #List containing hydrogen produced per hour [m^3]
            "H2_compressed" : [0]*self.duration, #List containing hydrogen produced per hour, after compression [m^3]
            "stored" : [0]*self.duration, #List containing hydrogen stored in the storage, after selling it, per hour [m^3]
            "wasted" : [0]*self.duration, #List containing hydrogen that can't be stored per hour [m^3]
            "sold" : [0]*self.duration #List containing hydrogen sold per hour [m^3]
            }
        
        self.KPI = {
            "LCOH" : 0, #LCOH [€]
            "H2" : 0, #Total hydrogen produced over a year [kg]
            "H2_used" : 0, #Total hydrogen effectively used over a year [kg]
            "wasted_power" : 0, #Share of the excess power wasted due to an undersized electrolyzer [kW]
            "benefit" : 0, #Benefit from selling hydrogen [€]
            "wasted_hydrogen" : 0, #%of the total hydrogen that couldn't be stored (when storage full) [%]
            "%time_storage_full" : 0 #% of time when the storage is full [%]
            }
        

#STUDY PART 1

def empty_plant():
    plant = H2plant()
    plant.get_data_from_excel()
    plant.power_manager()
    return plant

def define_plant(C_electrolyzer, C_storage, N, T):
    plant = H2plant()
    plant.get_data_from_excel()
    plant.power_manager()
    
    plant.set_electrolyzer_capacity(C_electrolyzer)
    plant.set_storage_capacity(C_storage)
    plant.set_number_of_trucks(N)
    plant.set_threshold(T)
    return plant 

def define_plant_fast(C_electrolyzer, C_storage, N, T):
    plant = define_plant(C_electrolyzer, C_storage, N, T)
    plant.electrolyzer_production()
    plant.hydrogen_management()
    return plant

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

#STUDY PART 2

def test_shuffle(plant):
    """
    To test whether or not shuffling is working

    Parameters
    ----------
    plant : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    t = range(len(plant.data["WP"]))
    
    plt.figure()
    plt.plot(t, plant.data["WP"])
    plant.shuffle("WP")
    plt.plot(t, plant.data["WP"])
    plant.shuffle("WP")
    plt.plot(t, plant.data["WP"])
    plt.show()
    
    plt.figure()
    plt.plot(t, plant.data["NP"])
    plant.shuffle("NP")
    plt.plot(t, plant.data["NP"])
    plant.shuffle("NP")
    plt.plot(t, plant.data["NP"])
    plt.show()
    
def sensitivity_analysis_data(plant, nb_shuffle):
    """
    Sensitivity analysis of the form of the yield power of nuclear + wind

    Parameters
    ----------
    plant : H2_plant
        The H2_plant instance that you use.
    nb_shuffle : int
        The number of shuffling you want to do for the sensitivity analysis.

    """
    
    # We shuffle the nuclear power plant time series
    NP_shuffle_capacity = []
    for i in range(nb_shuffle):
        print(i)
        plant.shuffle("NP")
        plant.power_manager()
        # optmised_variables = optimisation()
        NP_shuffle_capacity.append(capacity)
        
      
    # We shuffle the wind plant time series
    WP_shuffle_capacity = []
    for i in range(nb_shuffle):
        print(i)
        plant.shuffle("WP")
        plant.power_manager()
        # optmised_variables = optimisation()
        WP_shuffle_capacity.append(capacity)
        
    X = range(nb_shuffle)
    figure, axis = plt.subplots(2, 1)
    axis[0].plot(X, NP_shuffle_capacity)
    axis[0].set_title("Best capacity for shuffled nuclear power plants")
    
    axis[1].plot(X, WP_shuffle_capacity)
    axis[1].set_title("Best capacity for shuffled windplants")

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
    params = np.linspace(mini, maxi, 100)
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

#STUDY PART 3
        
def wind_power_prod(wind):
    """
    To get the expected yield power from a wind turbine with a specific power curve.

    Parameters
    ----------
    wind : float
        The wind speed at which the wind turbine has to produce.

    Returns
    -------
    power : float
        The yield power from the wind turbine.

    """
    w = int(wind)
    x = wind - w
    power_curve = [0, 0, 33, 197, 447, 804, 1298, 1936, 2635, 3091, 3281, 3300];
    if w >= len(power_curve) - 1:
        power = power_curve[-1]
    else:
        power = power_curve[w] + x*(power_curve[w+1] - power_curve[w])
    return power

def simulate_wind_distribution(L, k, duration = 8760):
    """
    Simulate a wind distribution, based on the Weibull law.

    Parameters
    ----------
    L : float
        Lambda parameter of the Weibull law.
    k : float
        K parameter of the Weibull law.
    duration : int, optional
        Number of samples in the distribution. The default is 8760 (a year).

    Returns
    -------
    wind : list
        A list containing the wind speeds.

    """
    rng = np.random.default_rng()
    wind = [0]*duration
    for i in range(duration):
        wind[i] = L*rng.weibull(k)
    return wind

def simulate_wind_power_plant(L, k, duration = 8760):
    #TO VARY
    """
    Simulate a wind power plant.

    Parameters
    ----------
    L : float
        Lambda parameter of the Weibull law.
    k : float
        K parameter of the Weibull law.
    duration : int, optional
        Number of samples in the distribution. The default is 8760 (a year).

    Returns
    -------
    power_produced : list
        The power produced by the wind power plant at each time step.

    """
    power_produced = [0]*duration
    wind = simulate_wind_distribution(L, k, duration)
    for i in range(duration):
        power_produced[i] = wind_power_prod(wind[i])
    return power_produced

def simulate_nuclear_power_plant(capacity, duration = 8760):
    CF = 0.92
    #TO BE DONE
def show_production(plant):
    """
    To show the primary, excess and supplied power,

    Parameters
    ----------
    plant : H2plant
        The hydrogen plant.

    Returns
    -------
    None.

    """ 
    WP, NP = plant.data['WP'], plant.data["NP"]
    EP, SP = plant.data["excess_power"], plant.data["supply_power"]
    HP = plant.res['H2_compressed']
    
    figure, axis = plt.subplots(3, 1)
    
    # Wind power
    axis[0].plot(WP)
    axis[0].set_title("Wind power")

    #Nuclear power 
    axis[1].plot(NP)
    axis[1].set_title("Nuclear power")

    #NP + WP, grid limit
    temps = range(plant.duration)
    NP_WP = [WP[t] + NP[t] for t in temps]
    axis[2].plot(temps, NP_WP, label="Nuclear power + Wind power")
    axis[2].plot(temps, [plant.param['grid_limit']]*plant.duration, label=F"Grid limitation : {plant.param['grid_limit']} kW")
    
    axis[2].set_title("Wind power + Nuclear power [kW]")
    axis[2].legend()    
    
    # Labels
    plt.setp(axis[2], xlabel="Time (hour)")
    for i in range(3):
        plt.setp(axis[i], ylabel='Power [kW]')
        
    figure2, axis2 = plt.subplot_mosaic([['A', 'B'],
                                  ['C', 'C']])

    #Excess power, capacity
    axis2['A'].plot(temps, EP, label='Excess power')
    axis2['A'].plot(temps, [plant.var['electrolyzer_capacity']]*len(temps), label="Electrolyzer capacity")
    axis2['A'].legend()    
    axis2['A'].set_title('Excess power')
    plt.setp(axis2['A'], xlabel='Time (hour)')
    plt.setp(axis2['A'], ylabel='Power [kW]')
    
    #Supplied produced
    axis2['B'].plot(temps, SP, label="Supplied power")
    axis2['B'].legend()
    axis2['B'].set_title('Supplied power')
    plt.setp(axis2['B'], xlabel='Time (hour)')
    plt.setp(axis2['B'], ylabel='Power [kW]')

    
    #Hydrogen produced
    axis2['C'].plot(temps, HP, label="Hydrogen produced")
    axis2['C'].legend()
    axis2['C'].set_title('Hydrogen produced (after compression')
    plt.setp(axis2['C'], xlabel='Time (hour)')
    plt.setp(axis2['C'], ylabel='Hydrogen produced [m3]')
    figure2.show()

def show_management(plant):
    """
    To show the hydrogen flow in the system.

    Parameters
    ----------
    plant : H2plant
        The hydrogen plant for which we want to show the hydrogen flow.

    Returns
    -------
    None.

    """
    t = np.arange(plant.duration)
    figure, axis = plt.subplots(2, 1)
    
    axis[0].plot(t, plant.res['H2_compressed'], label="Hydrogen produced")
    axis[0].plot(t, plant.res['stored'], label="Hydrogen stored")
    axis[0].plot([0, plant.duration-1], [plant.var['storage_capacity']]*2, 'r', label="Storage capacity")
    axis[0].plot([0, plant.duration-1], [plant.var['storage_capacity']*plant.var['threshold']]*2, 'r--', label="Threshold")
    
    plt.setp(axis[0], xlabel = "Time [hour]")
    plt.setp(axis[0], ylabel = "Volume of hydrogen produced/stored [m3]")
    
    axis[0].set_title(f'''Hydogen capacity : {plant.var["electrolyzer_capacity"]} kW /
              Storage capacity : {plant.var["storage_capacity"]} m3 /
              Number of trucks : {plant.var['number_of_trucks']}.\n\n
              Hydrogen produced and stored throughout the year.''')
    
    axis[0].legend()
    
    tmp = plant.res['H2_compressed']
    tmp2 = plant.res["wasted"]
    axis[1].plot(t, [sum(tmp[:T]) for T in t], label='Amount of hydrogen produced')
    axis[1].plot(t, [sum(tmp2[:T]) for T in t], label='Amount of hydrogen wasted')
    
    plt.setp(axis[1], xlabel = 'Time [hour]')
    plt.setp(axis[0], ylabel = "Total volume hydrogen produced/wasted[m3]")
    
    axis[1].set_title(f'''Hydrogen produced and stored throughout the year. \n
                      Share of hydrogen wasted : {int(100*sum(tmp2)/sum(tmp))}%''')
    axis[1].legend()
    
    figure.show()
     
def close():
    plt.close("all")


# Initialisation

# plant = empty_plant()

def create_pop(len_population, C_min, C_max, S_min, S_max, N_min, N_max):
    population = []
    
    for i in range(len_population):
        population.append([random.randint(C_min, C_max), random.randint(S_min, S_max), random.randint(N_min, N_max)])
    
    return population

def initialization(plant, len_population, C_min, C_max, S_min, S_max, N_min, N_max):

    population = []
    cpt = 0
    while len(population) != len_population:
        pop = [random.randint(C_min, C_max), random.randint(S_min, S_max), random.randint(N_min, N_max)]
        LCOH_associated = plant.objective(*pop, 1)
        verif_WP, verif_WH = plant.constraints()
        
        if verif_WP + verif_WH == 0:
            population.append(pop)
        cpt += 1
        print(f"Taille de la liste : {len(population)} / Nombre d'essais {cpt}")
    
    print("Initialisation completed")
    return population

def improver(plant, el, C_min, C_max, S_min, S_max, N_min, N_max):
    element = el[:]
    C_max = 1e6
    S_max = 1e5
    
    C_rate = int(C_max/100)
    S_rate = int(S_max/100)
    N_rate = 3
    rates = [C_rate, S_rate, N_rate]
    
    configu = element[0]
    new_elements = [element]
    # breakpoint()
    for i in range(3):
        for j in range(2):
            new_config = configu.copy()
            new_config[i] = configu[i] + (2*j-1)*rates[i]
            plant.clear()
            LCOH_associated = plant.objective(*new_config, 1)
            verif_WP, verif_WH = plant.constraints()
           
            if verif_WP + verif_WH == 0:
                tmp = [new_config, LCOH_associated]
                new_elements.append(tmp)  
            # breakpoint()
                
    sorted_elem = sorted(new_elements, key=lambda x: x[1])   
    plant.clear()    
    return sorted_elem[0]
    
def sort_selection(plant, population, nb_improvements, C_min, C_max, S_min, S_max, N_min, N_max):
    # We initialize the res list
    res = []
    
    # We compute the LCOH for each chromosome and we verify 
    # if each chromosome verify the constraints or not
    for pop in population:
        C, S, N = pop
        LCOH_associated = plant.objective(C, S, N, 1)
        verif_WP, verif_WH = plant.constraints()
        # breakpoint()
        # ------------- CONSTRAINTS INTEGRATION 
        # Constraints are taken as hard penalities, since LCOH tends to be < 1
        if verif_WP + verif_WH == 0:
            tmp = [pop, LCOH_associated]
            res.append(tmp)  
        # breakpoint()
    # We sort the res list to get the best chromosomes
    # breakpoint()
    sorted_population = sorted(res, key=lambda x: x[1])       
    best_LCOH = sorted_population[0][1]
    # breakpoint()
    sorted_population_without_LCOH = [pop[0] for pop in sorted_population]
    # breakpoint()
    # Improvement of the best one
    for i in range(nb_improvements):
        best_couple = sorted_population[i]
        couple_improved = improver(plant, best_couple, C_min, C_max, S_min, S_max, N_min, N_max)
        sorted_population[i] = couple_improved
        # breakpoint()
    # breakpoint()
    return sorted_population_without_LCOH, sorted_population, best_LCOH

def parent_selection_1(sorted_population_LCOH, share=1/3):
    propositions = []
    
    for i in range(int(share*len(sorted_population_LCOH))):
        tmp = random.choice(sorted_population_LCOH)
        propositions.append(tmp)
    sorted_proposition = sorted(propositions, key=lambda x: x[1])

    return sorted_proposition[0]

def crossover_1(p1_LCOH, p2_LCOH, probability):

    p_best, p_worst = sorted([p1_LCOH, p2_LCOH], key=lambda x: x[1])
    p_best, p_worst = p_best[0], p_worst[0]
   
    child = p_best.copy()
    do_crossover = random.randint(0,1)
    
    if do_crossover < probability:
        i_property = random.randint(0, 2)
        child[i_property] = p_worst[i_property]
    
    return child

def crossover_2(p1_LCOH, p2_LCOH, alpha, probability):
    do_crossover = random.randint(0,1)
    if do_crossover < probability:
        p1 = p1_LCOH[0]
        p2 = p2_LCOH[0]
        child = []
        i_property = random.randint(0, 2)
        # breakpoint()
        for i in range(3):
            # breakpoint()
            if i == i_property:
                child.append(p1[i]*alpha + p2[i]*(1-alpha))
            else:
                select = random.randint(0, 1)
                child.append(p1[i]*select + p2[i]*(1-select))
    else:
        p_best, p_worst = sorted([p1_LCOH, p2_LCOH], key=lambda x: x[1])
        p_best, p_worst = p_best[0], p_worst[0]
        child = p_best.copy()
        
    return child

def crossover_3(p1_LCOH, p2_LCOH, probability):

    do_crossover = random.randint(0,1)
    if do_crossover < probability:
        p1 = p1_LCOH[0]
        p2 = p2_LCOH[0]
        child = []
        for i in range(3):
            alpha = random.uniform(0, 1)
            child.append(p1[i]*alpha + p2[i]*(1-alpha))
    else:
        p_best, p_worst = sorted([p1_LCOH, p2_LCOH], key=lambda x: x[1])
        p_best, p_worst = p_best[0], p_worst[0]
        child = p_best.copy()
    
    return child

def crossover_4(p1_LCOH, p2_LCOH, probability):
    do_crossover = random.randint(0,1)
    if do_crossover < probability:
        child = []
        for i in range(3):
            child.append((p1_LCOH[0][i]+p2_LCOH[0][i])/2)
    
    else:
        alea = random.randint(0, 1)
        child = [p1_LCOH[0], p2_LCOH[0]][alea]
    
    return child

def mutation_1(p, probability, C_min, C_max, S_min, S_max, N_min, N_max, C_rate, S_rate, N_rate):
    p = p.copy()

    bounds = [[C_min, C_max],[S_min, S_max],[N_min, N_max]]
    
    C_rate = int(C_rate*(C_max - C_min))
    S_rate = int(S_rate*(S_max - S_min))
    N_rate = int(N_rate*(N_max - N_min))
    
    rates = [C_rate, S_rate, N_rate]
    do_mutation = random.randint(0, 1)
    
    if do_mutation < probability:
    
        i_property = random.randint(0, 2)
        mini, maxi = bounds[i_property]
        rate = rates[i_property]
        
        if p[i_property] - rate < mini:
            p[i_property] += rate
        elif p[i_property] + rate > maxi:
            p[i_property] -= rate
        else:
            p[i_property] += rate*(1-2*random.randint(0, 1))
    return p

plant = empty_plant()

def algo(plant, len_pop, n_iter):
    # Range for the optimization
    C_min = 1
    C_max = 1e6
    S_min = 1
    S_max = 1e5
    N_min = 1
    N_max = 100
    C_rate = 0.1
    S_rate = 0.1
    N_rate = 0.3

    # initial_pop = create_pop(len_pop)
    initial_pop = initialization(plant, len_pop, C_min, C_max, S_min, S_max, N_min, N_max)
    new_gene = initial_pop    
    
    LCOHs = []
    
    new_child = int(len_pop*95/100)
    nb_child_from_best = 5
    old_child = len_pop - new_child
    
    # Probability defintions
    p_crossover = 0.95
    p_mutation = 0.75
    
    for i in range(n_iter):
        
        # Sort step
        sort_pop, sort_pop_LCOH, best_LCOH = sort_selection(plant, new_gene, 4, C_min, C_max, S_min, S_max, N_min, N_max)
        best_config = sort_pop[0]
        LCOHs.append(best_LCOH)
        # breakpoint()
        childs = []
        for j in range(new_child):
            # Parent selection step
            if j < nb_child_from_best:
                p1 = sort_pop_LCOH[0]
            else:
                p1 = parent_selection_1(sort_pop_LCOH)
            p2 = parent_selection_1(sort_pop_LCOH)
                
            # Crossover
            # child = crossover_1(p1, p2, p_crossover)
            # child = crossover_2(p1, p2, 0.5, p_crossover)
            # child = crossover_3(p1, p2, p_crossover)
            child = crossover_4(p1, p2, p_crossover)
            
            # Mutation
            mut = mutation_1(child, p_mutation, C_min, C_max, S_min, S_max, N_min, N_max, C_rate, S_rate, N_rate)
            
            # New child
            childs.append(mut)
        
        new_children = []
        for j in range(old_child):
            old_children = sort_pop[:old_child]
            child = mutation_1(old_children[j], p_mutation, C_min, C_max, S_min, S_max, N_min, N_max, C_rate, S_rate, N_rate)
            new_children.append(child)
            
        new_gene = childs + new_children
        print(f"{i+1}/{n_iter} iterations. Best LCOH : {best_LCOH}, best config {best_config}.")
        # print(f"{i+1}/{n_iter} iterations.")
    # Sort step
    sort_pop, sort_pop_LCOH, best_LCOH = sort_selection(plant, new_gene, 1, C_min, C_max, S_min, S_max, N_min, N_max)
    LCOHs.append(best_LCOH)
        
    best_config = sort_pop[0]
    
    return best_config, best_LCOH, LCOHs, new_gene
    
def algo_of_algo(n):
    plant = empty_plant()
    
    configs = []
    LCOHs = []
    LCOHs_eachs = []
    
    for i in range(n):
        best_config, best_LCOH, LCOHs_each, new_gene = algo(plant, 100, 50)
        configs.append(best_config)
        LCOHs.append(best_LCOH)
        LCOHs_eachs.append(LCOHs_each)
        print(f'Algo number {i+1}/{n} completed')
    
    Cs, Ss, Ns = [], [], []
    val = [Cs, Ss, Ns]
    for i in range(3):
        val[i] = [all_config[i] for all_config in configs]
    
    fig, axis = plt.subplots(2, 2)
    
    axis[0, 0].boxplot(val[0])
    axis[0, 0].set_title("Capacity [kW]")
    
    axis[0, 1].boxplot(val[1])
    axis[0, 1].set_title("Storage [m3]")
    
    axis[1, 0].boxplot(val[2])
    axis[1, 0].set_title("Number of trucks")
    
    axis[1, 1].boxplot(LCOHs)
    axis[1, 1].set_title("LCOH [€/kWh]")
    
    return configs, LCOHs, val