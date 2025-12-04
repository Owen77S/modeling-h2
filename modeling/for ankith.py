import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt

class H2plant():
    def __init__(self):
        
        self.duration = 8760 #Time step of the modelisation
        
        self.data = {
            "WP" : [0]*self.duration, #List containing yield power from wind plant [kW]
            "NP" : [0]*self.duration, #List containing yield power from nuclear power plant [kW]
            "excess_power" : [0]*self.duration, #List containing excess power (risk of congestion) [kW]
            "supply_power"  : [0]*self.duration #List containing supplied power to the electrolyzer [kW]
            } 
        
        self.economics = {
            "install_fee" : 1800, #Install fee of the electrolyzer [€/kW]
            "OPEX_PEM" : 54, #OPEX of the PEM [€/kW/year]
            "water_price" : 3e-3, #water price [€/kg]
            "water_consumption" : 9, #water consumption to produce a kg of H2 [kG H2O/kg H2]
            "price_H2" : 2.7 #hydrogen price [€/kg]
            } 
        
        self.res = {
            "mass_H2" : [0]*self.duration, #List containing mass of hydrogen produced per hour [kg]
            "H2" : [0]*self.duration, #List containing hydrogen produced per hour [m^3]
            "H2_compressed" : [0]*self.duration, #List containing hydrogen produced per hour, after compression [m^3]
            "stored" : [0]*self.duration, #List containing hydrogen stored in the storage, after selling it, per hour [m^3]
            "wasted" : [0]*self.duration, #List containing hydrogen that can't be stored per hour [m^3]
            "sold" : [0]*self.duration, #List containing hydrogen sold per hour [m^3]
            'when_sold' : [], #list containing all the time steps at which we sell hydrogen
            'when_unavailable' : []}  #list containing all the time steps at which trucks are unavailable
        
        self.KPI = {
            "LCOH" : 0, #LCOH [€]
            "H2" : 0, #Total hydrogen produced over a year [kg]
            "H2_sold" : 0, #Total hydrogen effectively used over a year [kg]
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
            "NM3_to_kg" : 101325*2e-3/(8.314*273.15),
            "correlation_to_reality" : 2/3.49 #so that a 120kW electrolyser produce 2 kg of H2 per hour and not 3.4 as estimates our model.
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
        To initialize the plant with a WP/NP list.

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
    
    def change_nuclear_capacity(self, NC):
        """
        To change the power capacity of the nuclear plant. 
        Has to be used before power_manager() to be accounted !

        Parameters
        ----------
        NC : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        tmp = self.data["NP"].copy()
        self.data["NP"] = [NC*np/1450 for np in tmp]
    
    def power_manager(self):
        """
        Get the excess electricity. !!! Need to have a nuclear and wind
        yield production already initialized !!! (with get_data_from_excel)
        
        -------
        None.

        """
        if self.data['WP'] == [0]*self.duration:
            print("The yield wind power has not been initialized !")
            return       
    
        if self.data['NP'] == [0]*self.duration:
            print("The yield nuclear power has not been initialized !")
            return
        for t in range(self.duration):
            excess_electricity = self.data["WP"][t] + self.data["NP"][t] - self.param['grid_limit']
            if  excess_electricity > 0:
                #We produce to much energy : congestion
                self.data["excess_power"][t] = excess_electricity
            else:
                #We don't produce to much energy : no congestion
                self.data["excess_power"][t] = 0
    
    def n_F(self, electrolyzer_capacity, supply_power):
        """
        To compute the faradic efficiency.

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
        the year by the electrolyzer. !!! The electrolyzer capacity should be defined !!!       

        Parameters
        ----------
        None.

        Returns
        -------
        None.

        """
        if self.var['electrolyzer_capacity'] == 0:
            print("The electrolyzer capacity has not been defined !")
            return
        
        #Auxiliaries power requirement : 3% of the power supply
        n_aux = 1 - 0.03
        for t in range(self.duration):
            if self.data["excess_power"][t] >= self.var["electrolyzer_capacity"]:
                #The supply power can't excess the electrolyzer's capacity
                self.data["supply_power"][t] = self.var["electrolyzer_capacity"]
            else:
                self.data["supply_power"][t] = self.data["excess_power"][t]
            
            mass_H2 = n_aux*self.n_F(self.var["electrolyzer_capacity"], self.data["supply_power"][t])*self.data["supply_power"][t]/self.param['LHV_kg']
            mass_H2 = mass_H2*self.param['correlation_to_reality'] #explanation in the definition of the parameter
            self.res["mass_H2"][t] = mass_H2
            # Ideal gas law to convert the mass of H2 into a volume
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
        
        # In the first hour : we store the H2 produced. If it's higher than the capacity the remaining 
        # is wasted (we assume that we don't sell any hydrogen during the first hour)
        self.res['stored'][0] = min(self.res['H2_compressed'][0], self.var["storage_capacity"])

        for t in range(1, self.duration):
            
            # --------------------------------- Storage part 
            
            #Initialisation
            share_wasted = 0
            share_stored = 0
            quantity_sold = 0
            
            availability_in_storage = self.var["storage_capacity"] - self.res["stored"][t-1]

            # Dispatch strategy for the storage
            if self.res["H2_compressed"][t] > availability_in_storage:
                share_wasted = self.res["H2_compressed"][t] - availability_in_storage
                share_stored = availability_in_storage               
            
            else:
                share_stored = self.res['H2_compressed'][t]
                share_wasted = 0
            
            # Storage
            self.res["wasted"][t] = share_wasted  
            self.res["stored"][t] = self.res["stored"][t-1] + share_stored
            
            # --------------------------------- Selling part

            # If the state of charge is higher than the threshold for selling and there are trucks available : we sell
            if self.res["stored"][t] >= self.var["storage_capacity"]*self.var['threshold'] and trucks_available > 0:

                # We get the number of trucks that can be used, the quantity that will be sold,
                # and update the nb of trucks available and the storage capacity
                number_of_trucks_used = min(self.res['stored'][t]//self.param['truck_capacity'], trucks_available)
                trucks_available -= number_of_trucks_used
                quantity_sold = number_of_trucks_used*self.param['truck_capacity']
                list_of_unavailabilities.append([number_of_trucks_used, self.param["unavailable_hours"]])
                self.res["sold"][t] = quantity_sold
                self.res["stored"][t] = self.res["stored"][t] - quantity_sold

                self.res['when_sold'].append(t)
            
            # Else : we do nothing
            elif trucks_available == 0:
                self.res['when_unavailable'].append(t)
            
            # --------------------------------- Trucks manager
            
            # If the list is not empty, then there are trucks unavailable
            if len(list_of_unavailabilities) > 0:
                for i in range(len(list_of_unavailabilities)):
                    # If the first fleet of trucks come in one hour, then they are available in the next hour
                    if list_of_unavailabilities[0][1] == 1:
                        available_next_hour = 1
                    # We reduce their unavailability by one hour
                    list_of_unavailabilities[i][1] -= 1

            # If a fleet of truck is available at this hour (available_next_hour = 1), we update the
            # different variables.
            if available_next_hour == 1:
                trucks_available += list_of_unavailabilities[0][0]
                list_of_unavailabilities.pop(0)
                available_next_hour = 0
    
    def get_KPI(self):
        """
        To update KPIs

        Returns
        -------
        None.

        """
        years = 30
        kg_to_m3 =  1*self.param["correlation_to_reality"]*self.gas_model["R"]*(self.gas_model["T_op"]+273.15)/(self.gas_model["M"]*self.gas_model["P_op"]*10**5)
        
        self.KPI['H2'] = sum(self.res["mass_H2"])
        total_sold_decompressed = sum(self.res["sold"])/(self.gas_model["P_op"]/self.gas_model["P_out"])**(1/self.gas_model["n"])
        self.KPI["H2_sold"] = (self.gas_model["M"]*self.gas_model["P_op"]*10**5)*total_sold_decompressed/(self.gas_model["R"]*(self.gas_model["T_op"]+273.15))

        
        storage_in_kg = ((self.gas_model['P_out']*1e5)*self.var['storage_capacity']*self.gas_model['M'])/(self.gas_model['R']*(self.gas_model['T_op']+273.15))
        CAPEX_PEM = self.economics['install_fee']*self.var['electrolyzer_capacity']
        CAPEX_storage = 490*storage_in_kg
        CAPEX_selling = 93296 + self.var["number_of_trucks"]*610000
        CAPEX = CAPEX_PEM + CAPEX_storage + CAPEX_selling
        
        OPEX_PEM = self.economics['OPEX_PEM']*self.var['electrolyzer_capacity'] 
        water_price = self.economics['water_price']*self.economics["water_consumption"]*self.KPI['H2']
        OPEX_compressor = 4665
        OPEX_storage = 9.385398411955556*self.var['storage_capacity']*kg_to_m3
        OPEX_selling = 30500*self.var['number_of_trucks']
        OPEX = OPEX_PEM + water_price + OPEX_compressor + OPEX_selling + OPEX_storage
        
        replacement_PEM = (years//7.6)*630*self.var['electrolyzer_capacity']

        replacement_storage = (years//25)*470*self.var['storage_capacity']/kg_to_m3
        
        replacement = replacement_PEM + replacement_storage
        
        WACC = 5/100
        eps = 0.3/100
        num = CAPEX + replacement + sum([OPEX/(1+WACC)**n for n in range(1, years+1)])
        
        energy_in_H2 = self.KPI['H2_sold']*self.param['LHV_kg'] #kWh
        den = energy_in_H2*sum([(1-eps)**(n-1)/(1+WACC)**n for n in range(1, years+1)])
        
        self.KPI['LCOH'] = num/den

        self.KPI["wasted_power"] = 1 - sum(self.data["supply_power"])/sum(self.data["excess_power"])
        q = 1-eps
        decrease = (1-q**years)/(1-q)
        self.KPI["benefit"] = self.param["H2_price"]*self.KPI['H2_sold']*decrease
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
        
        res = self.KPI['LCOH']  

        return res   
    
    def objective2(self, C, S, N, T):
        """
        To compute the second objective function.

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
        
        penalty = self.KPI['wasted_hydrogen']*2 + self.KPI['wasted_power']*2
        
        res = self.KPI['LCOH'] + penalty

        return res, penalty
    
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
        verif_WP : int.
            0 : if the wasted power constraint is verified
            1 ! otherwise.
            
        verif_WH : boolean.
            0 : if the wasted hydrogen constraint is verified
            1 otherwise.

        """
        verif_WP = 0 if self.KPI["wasted_power"] < 0.7 else 1
        verif_WH = 0 if self.KPI["wasted_hydrogen"] < 0.7 else 1
        
        return verif_WP, verif_WH
        
    
    def shuffle(self, what):
        """
        Shuffle the yield power from the wind and the nuclear power plant.
        For sensitivity analysis.
        
        Parameters
        ----------
        what : str.
            "WP" : if we want to shuffle the wind power list.
            "NP" : if we want to shuffle the nuclear power list.
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
        """
        To set the electrolyzer capacity.

        Parameters
        ----------
        C : int.
            Electrolyzer capacity [kW].

        Returns
        -------
        None.

        """
        self.var["electrolyzer_capacity"] = C
    
    def set_storage_capacity(self, C):
        """
        To set the storage capacity.

        Parameters
        ----------
        C : int.
            Storage capacity [m3].

        Returns
        -------
        None.

        """
        self.var["storage_capacity"] = C
        
    def set_number_of_trucks(self, N):
        """
        To set the number of trucks.

        Parameters
        ----------
        N : int
            Number of trucks.

        Returns
        -------
        None.

        """
        self.var["number_of_trucks"] = N
        
    def set_threshold(self, T):
        """
        To set the threshold.

        Parameters
        ----------
        T : float.
            Threshold between 0 and 1.

        Returns
        -------
        None.

        """
        if T < 0 or T > 1:
            print("Wrong threshold value ! Must be between 0 and 1.")
            return
        self.var['threshold'] = T
    
    def clear(self):
        """
        To reinitialize the data of the plant. The function doesn't reinitialize the wind and nuclear
        power data.

        Returns
        -------
        None.

        """
        self.data["supply_power"] = [0]*self.duration 
        
        self.res = {
            "mass_H2" : [0]*self.duration, #List containing mass of hydrogen produced per hour [kg]
            "H2" : [0]*self.duration, #List containing hydrogen produced per hour [m^3]
            "H2_compressed" : [0]*self.duration, #List containing hydrogen produced per hour, after compression [m^3]
            "stored" : [0]*self.duration, #List containing hydrogen stored in the storage, after selling it, per hour [m^3]
            "wasted" : [0]*self.duration, #List containing hydrogen that can't be stored per hour [m^3]
            "sold" : [0]*self.duration, #List containing hydrogen sold per hour [m^3]
            'when_sold' : [], #list containing all the time steps at which we sell hydrogen
            'when_unavailable' : []}
        
        self.KPI = {
            "LCOH" : 0, #LCOH [€]
            "H2" : 0, #Total hydrogen produced over a year [kg]
            "H2_sold" : 0, #Total hydrogen effectively used over a year [kg]
            "wasted_power" : 0, #Share of the excess power wasted due to an undersized electrolyzer [kW]
            "benefit" : 0, #Benefit from selling hydrogen [€]
            "wasted_hydrogen" : 0, #%of the total hydrogen that couldn't be stored (when storage full) [%]
            "%time_storage_full" : 0 #% of time when the storage is full [%]
            }



# --------------- To initialize

def empty_plant(nb=104):
    """
    To create a plant with the wind and power data initialized with excel.

    Parameters
    ----------
            
    nb : int
        The number of wind turbines.

    Returns
    -------
    plant : TYPE
        DESCRIPTION.

    """
    plant = H2plant()
    plant.get_data_from_excel(nb)
    plant.power_manager()
    return plant


def define_plant(C, S, N, T=1, nb=104):
    """
    To get the KPIs from a plant defined with the 
    specific variables.

    Parameters
    ----------
    C : int
        Electrolyzer capacity [kW].
    S : int.
        Storage capacity [m3].
    N : int.
        Number of trucks.
    T : float between 0 and 1.
        Threshold.
    nb : int
        The number of wind turbines

    Returns
    -------
    plant : H2plant
        Plant with updated KPIs.

    """
    plant = H2plant()
    plant.get_data_from_excel(nb)
    plant.power_manager()
    
    plant.set_electrolyzer_capacity(C)
    plant.set_storage_capacity(S)
    plant.set_number_of_trucks(N)
    plant.set_threshold(T)
    
    plant.electrolyzer_production()
    plant.hydrogen_management()
    plant.get_KPI()
    
    return plant 

# --------------- Wind/Nuclear plant simulation

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
    # TO BE VERIFIED
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
    # TO BE COMPLETED
    CF = 0.92
    
# --------------- For plots

def show_production(plant):
    """
    To show the primary, excess and supplied power, of a plant.

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
    axis[2].stackplot(temps, [WP, NP], labels= ["Wind power", "Nuclear power"])
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
    
    plt.setp(axis[0], xlabel = "Time [hour]")
    plt.setp(axis[0], ylabel = "Volume of hydrogen produced/stored [m3]")
    
    axis[0].set_title('Hydrogen produced and stored throughout the year.')
    
    axis[0].legend()
    
    tmp = plant.res['H2_compressed']
    tmp2 = plant.res["wasted"]
    axis[1].plot(t, [sum(tmp[:T]) for T in t], label='Amount of hydrogen produced')
    txt = f'Amount of hydrogen wasted (Total wasted : {int(100*sum(tmp2)/sum(tmp))}%)'
    axis[1].plot(t, [sum(tmp2[:T]) for T in t], label=txt)
    
    plt.setp(axis[1], xlabel = 'Time [hour]')
    plt.setp(axis[1], ylabel = "Total volume hydrogen produced/wasted[m3]")
    
    axis[1].set_title('Hydrogen produced and stored throughout the year')
    axis[1].legend()
    
    figure.show()
     
def close():
    """
    To close all the plots.

    Returns
    -------
    None.

    """
    plt.close("all")

