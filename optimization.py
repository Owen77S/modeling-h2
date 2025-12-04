ximport pandas as pd
import numpy as np  
import matplotlib.pyplot as plt
import random

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

# --------------- GENETIC ALGORTIHM

def create_pop(len_population, C_min, C_max, S_min, S_max, N_min, N_max):
    """
    To create a random population. 
    Each member of the population is represented as a list containing three properties :
        - its electrolyzer capacity,
        - its storage capacity,
        - its number of trucks :
    
    member = [C_member, S_member, N_member].
    
    C_min, C_max, S_min, S_max, N_min, N_max are all int and represent the minimum/maximum value 
    that can take the electrolyzer capacity (C), the storage capacity (S), and
    the number of trucks (N) of each member of the population.

    Parameters
    ----------
    len_population : int.
        The length of the population (whcih is the length the of the list).
    C_min, C_max, S_min, S_max, N_min, N_max are all int and represent the minimum/maximum value 
    that can take the electrolyzer capacity (C), the storage capacity (S), and
    the number of trucks (N) of each member of the population.

    Returns
    -------
    population : list.
        The list containing the population.
        population = [member1, member2, ..., membern]
                   = [[C1, S1, N1], [C2, S2, N2], ..., [Cn, Sn, Nn]]

    """
    population = []
    
    for i in range(len_population):
        population.append([random.randint(C_min, C_max), random.randint(S_min, S_max), random.randint(N_min, N_max)])
    
    return population

def initialization(plant, len_population, C_min, C_max, S_min, S_max, N_min, N_max, printed):
    """
    Create a population satisfying the constraints.

    Parameters
    ----------
    plant : H2plant.
        A H2plant. It's more convenient to keep the same plant for the whole optimisation process.
    len_population, C_min, C_max, S_min, S_max, N_min, N_max are defined in the create_pop function.
    
    printed : int.
        If printed = 0 : it won't print the progress of the function (recommended while doing a multiprocess).
        Otherwise : it will.
    Returns
    -------
    population : list.
        A population, verifying the constraints.

    """
    population = []
    cpt = 0
    while len(population) != len_population:
        pop = [random.randint(C_min, C_max), random.randint(S_min, S_max), random.randint(N_min, N_max)]
        _ = plant.objective(*pop, 1)
        verif_WP, verif_WH = plant.constraints()
        
        if verif_WP + verif_WH == 0:
            population.append(pop)
        cpt += 1
        if printed != 0:
            print(f"Taille de la liste : {len(population)} / Nombre d'essais {cpt}")
    
    if printed != 0:
        print("Initialisation completed")
    return population

def improver(plant, member_LCOH, C_min, C_max, S_min, S_max, N_min, N_max):
    """
    Used to improve the member objective-wise.

    Parameters
    ----------
    plant : TYPE
        DESCRIPTION.
    member_LCOH : list
        A list containing a member of the population and its associated LCOH.
        member_LCOH = [[C_member, S_member, N_member], LCOH_associated]
    C_min, C_max, S_min, S_max, N_min, N_max are all int and represent the minimum/maximum value 
    that can take the electrolyzer capacity (C), the storage capacity (S), and
    the number of trucks (N) of each member of the population.

    Returns
    -------
    best_LCOH : float
        The LCOH of the improved member.
    best_config : list
        The improved member.

    """
    element = member_LCOH.copy()
    
    # Represents the exchange rate of each property between two
    # members during the improvement process
    C_rate = int(C_max/100)
    S_rate = int(S_max/100)
    N_rate = 3
    
    rates = [C_rate, S_rate, N_rate]
    
    configu = element[0]
    
    # The list will contain all the member-LCOH couples
    new_elements = [element]
    
    # For each property, we are creating two new members by changing the value of the property i as follows : 
    # Property i of new_member_1 = initial value property i + rate property i
    # Property i of new_member_2 = initial value property i - rate property i
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
    
    # Contains the list of all the LCOH
    LCOHs = [pop_LCOH[1] for pop_LCOH in new_elements]
    # Contains the list of all the members
    pops = [pop_LCOH[0] for pop_LCOH in new_elements]
    
    # We get the index of the member with the lowest LCOH
    best_LCOH = min(LCOHs)
    i_min = LCOHs.index(best_LCOH)
    best_config = pops[i_min]
    
    return best_LCOH, best_config
    
def sort_selection(plant, population, nb_improvements, C_min, C_max, S_min, S_max, N_min, N_max):
    """
    Select the members verifying the constraints in a population, and then sorting 
    the nb_improvments best ones in ascending order by LCOH.

    Parameters
    ----------
    plant : H2plant
        An H2 plant.
    population : list
        A population
    nb_improvements : int
        number of members that will be improved (suggestion : maximum 2 to make the calcul faster)
    C_min, C_max, S_min, S_max, N_min, N_max are all int and represent the minimum/maximum value 
    that can take the electrolyzer capacity (C), the storage capacity (S), and
    the number of trucks (N) of each member of the population.

    Returns
    -------
    pops : list
        The sorted population.
    pops_LCOHs : list
        The sorted population with the LCOH associated to each member.
        pops_LCOHs = [[member1, LCOH1], [member2, LCOH2],...,[membern, LCOHn]]
    LCOHs[0] : float
        The lowest LCOH of the whole population.

    """
    # We initialize the res list
    res = []
    
    # We compute the LCOH for each member and we verify 
    # if each member verify the constraints or not
    for pop in population:
        C, S, N = pop
        LCOH_associated = plant.objective(C, S, N, 1)
        verif_WP, verif_WH = plant.constraints()
        
        # ------------- CONSTRAINTS INTEGRATION 
        # If the one of the constraints is not verified, the member is not added to the new population.
        if verif_WP + verif_WH == 0:
            tmp = [pop, LCOH_associated]
            res.append(tmp)  
    
    LCOHs = [pop_LCOH[1] for pop_LCOH in res]
    pops = [pop_LCOH[0] for pop_LCOH in res]

    # Improvement of the best one

    best_LCOH = min(LCOHs)
    i_min = LCOHs.index(best_LCOH)
    
    best_couple = [pops[i_min].copy(), best_LCOH]
    
    LCOH_improved, couple_improved = improver(plant, best_couple, C_min, C_max, S_min, S_max, N_min, N_max)
    
    pops_LCOHs = res.copy()
    pops_LCOHs[0] = [couple_improved.copy(), LCOH_improved]
    
    LCOHs[0] = LCOH_improved
    pops[0] = couple_improved.copy()

    return pops, pops_LCOHs, LCOHs[0]

def parent_selection_alea(population):
    """
    To select a random member from a population.
    Used for the parent selection in the genetic algortihm.

    Parameters
    ----------
    population : list
        A population.

    Returns
    -------
    random.choice(population) : list
        A random member.

    """
    return random.choice(population)

def crossover_1(p1_LCOH, p2_LCOH, probability):
    """
    Function crossover 1.

    Parameters
    ----------
    p1_LCOH : list
        Member, considered as parent 1.
    p2_LCOH : list
        Member, considered as parent 2.
    probability : float between 0 and 1
        Probability that the crossover occurs.

    Returns
    -------
    child : list.
        A member, considered as the child from the two parents.

    """
    p_best, p_worst = sorted([p1_LCOH, p2_LCOH], key=lambda x: x[1])
    p_best, p_worst = p_best[0].copy(), p_worst[0].copy()
   
    child = p_worst.copy()
    do_crossover = random.uniform(0,1)
    
    if do_crossover < probability:
        i_property = random.randint(0, 2)
        child[i_property] = p_best[i_property]
        child[2] = int(child[2])
    
    return child

def crossover_2(p1_LCOH, p2_LCOH, alpha, probability):
    """
    Function crossover 2.

    Parameters
    ----------
    p1_LCOH : list
        Member, considered as parent 1.
    p2_LCOH : list
        Member, considered as parent 2.
    alpha : float between 0 and 1
        Value for the linear regression
    probability : float between 0 and 1
        Probability that the crossover occurs.

    Returns
    -------
    child : list.
        A member, considered as the child from the two parents.

    """
    do_crossover = random.uniform(0,1)
    if do_crossover < probability:
        p1 = p1_LCOH[0]
        p2 = p2_LCOH[0]
        child = []
        i_property = random.randint(0, 2)
        
        for i in range(3):
            
            if i == i_property:
                child.append(p1[i]*alpha + p2[i]*(1-alpha))
            else:
                select = random.randint(0, 1)
                child.append(p1[i]*select + p2[i]*(1-select))
        child[2] = int(child[2])
    else:
        p_best, p_worst = sorted([p1_LCOH, p2_LCOH], key=lambda x: x[1])
        p_best, p_worst = p_best[0], p_worst[0]
        child = p_best.copy()
        
    return child

def crossover_3(p1_LCOH, p2_LCOH, probability):
    """
    Function crossover 3.

    Parameters
    ----------
    p1_LCOH : list
        Member, considered as parent 1.
    p2_LCOH : list
        Member, considered as parent 2.
    probability : float between 0 and 1
        Probability that the crossover occurs.

    -------
    child : list.
        A member, considered as the child from the two parents.

    """

    do_crossover = random.uniform(0,1)
    if do_crossover < probability:
        p1 = p1_LCOH[0]
        p2 = p2_LCOH[0]
        child = []
        for i in range(3):
            alpha = random.uniform(0, 1)
            child.append(p1[i]*alpha + p2[i]*(1-alpha))
        child[2] = int(child[2])
    else:
        p_best, p_worst = sorted([p1_LCOH, p2_LCOH], key=lambda x: x[1])
        p_best, p_worst = p_best[0], p_worst[0]
        child = p_best.copy()
    
    return child

def crossover_4(p1_LCOH, p2_LCOH, probability):
    """
    Function crossover 4.

    Parameters
    ----------
    p1_LCOH : list
        Member, considered as parent 1.
    p2_LCOH : list
        Member, considered as parent 2.
    probability : float between 0 and 1
        Probability that the crossover occurs.
        
    Returns
    -------
    child : list.
        A member, considered as the child from the two parents.

    """
    do_crossover = random.uniform(0,1)
    if do_crossover < probability:
        child = []
        for i in range(3):
            child.append((p1_LCOH[0][i]+p2_LCOH[0][i])/2)
        child[2] = int(child[2])
    else:
        alea = random.randint(0, 1)
        child = [p1_LCOH[0].copy(), p2_LCOH[0].copy()][alea]
    
    return child

def mutation_1(member, probability, multiplier ,C_min, C_max, S_min, S_max, N_min, N_max, C_rate, S_rate, N_rate):
    """
    Function mutation 1

    Parameters
    ----------
    member : list
        The member we want to mute.
    probability : float between 0 and 1.
        The probability of doing the mutation.
    The other arguments are defined as previously.

    Returns
    -------
    p : list.
        The muted member.

    """
    p = member.copy()

    bounds = [[C_min, C_max],[S_min, S_max],[N_min, N_max]]
    
    C_rate = C_rate*(C_max - C_min)
    S_rate = S_rate*(S_max - S_min)
    N_rate = int(N_rate*(N_max - N_min))
    
    rates = [C_rate, S_rate, N_rate]
    do_mutation = random.uniform(0, 1)
    
    if do_mutation < probability:
        i_property = random.randint(0, 2)
        mini, maxi = bounds[i_property]
        rate = rates[i_property]*random.randint(1, multiplier)
        
        if p[i_property] - rate < mini:
            p[i_property] += rate/multiplier
        elif p[i_property] + rate > maxi:
            p[i_property] -= rate/multiplier
        else:
            p[i_property] += rate*(1-2*random.randint(0, 1))
    return p

def mutation_2(member, probability, multiplier ,C_min, C_max, S_min, S_max, N_min, N_max, C_rate, S_rate, N_rate):
    """
    Function mutation 2.

    The same arguments and outputs as mutation_1

    """
    p = member.copy()

    bounds = [[C_min, C_max],[S_min, S_max],[N_min, N_max]]
    
    C_rate = C_rate*(C_max - C_min)
    S_rate = S_rate*(S_max - S_min)
    N_rate = int(N_rate*(N_max - N_min))
    
    rates = [C_rate, S_rate, N_rate]
    do_mutation = random.uniform(0, 1)
    
    if do_mutation < probability:
        i_property = random.randint(0, 2)
        for i in range(3):
            if i != i_property:
                mini, maxi = bounds[i]
                rate = rates[i]*random.randint(1, multiplier)
                
                if p[i] - rate < mini:
                    p[i] += rate/multiplier
                elif p[i] + rate > maxi:
                    p[i] -= rate/multiplier
                else:
                    p[i] += rate*(1-2*random.randint(0, 1))
                
    return p

def mutation_3(p, probability ,C_min, C_max, S_min, S_max, N_min, N_max):
    """
    Function mutation 3 (recommanded)

    Parameters
    ----------
    The same arguments definition as the other functions, and the same output as the preivous
    mutation functions.

    """
    p = p.copy()

    bounds = [[C_min, C_max],[S_min, S_max],[N_min, N_max]]
        
    do_mutation = random.uniform(0, 1)
    
    if do_mutation < probability:
        nb_property = random.randint(1, 2)
        
        i_property = random.randint(0, 2)
        min_max = bounds[i_property]
        p[i_property] = random.randint(min_max[0], min_max[1])
        

        i_property_2 = i_property
        while i_property_2 == i_property:
            i_property_2 = random.randint(0, 2)
        min_max = bounds[i_property_2]
        p[i_property_2] = random.randint(min_max[0], min_max[1])
    return p


def crazy_mutation(elem, len_pop, probability, multiplier ,C_min, C_max, S_min, S_max, N_min, N_max, C_rate, S_rate, N_rate):
    """
    Function crazy mutation 1. Used to force a whole population to mute, to enhance diversity.

    Parameters
    ----------
    The same arg

    Returns
    -------
    new_gene : TYPE
        DESCRIPTION.

    """
    el = elem.copy()
    new_gene = [el]
    for i in range(len_pop - 1):
        mut = mutation_1(el, probability, multiplier ,C_min, C_max, S_min, S_max, N_min, N_max, C_rate, S_rate, N_rate)
        new_gene.append(mut)
    
    return new_gene

def crazy_mutation_2(plant, elem, len_pop, probability, multiplier ,C_min, C_max, S_min, S_max, N_min, N_max, C_rate, S_rate, N_rate, no_print):
    """
    Function crazy mutation 2. Used to force a whole population to mute, to enhance diversity.

    Parameters
    ----------
    The same arg

    Returns
    -------
    new_gene : TYPE
        DESCRIPTION.

    """
    el = elem.copy()
    new_gene = [el]
    nb_mut = int(len_pop/2)
    nb_init = len_pop - nb_mut - 1
    
    for i in range(nb_mut):
        mut = mutation_1(el, probability, multiplier ,C_min, C_max, S_min, S_max, N_min, N_max, C_rate, S_rate, N_rate)
        new_gene.append(mut)
    
    i = initialization(plant, nb_init, C_min, C_max, S_min, S_max, N_min, N_max, no_print)
    new_gene += i
    
    return new_gene

def crazy_mutation_3(plant, elem, len_pop,C_min, C_max, S_min, S_max, N_min, N_max):
    """
    Function crazy mutation 3. Used to force a whole population to mute, to enhance diversity.

    Parameters
    ----------
    The same arg

    Returns
    -------
    new_gene : TYPE
        DESCRIPTION.

    """
    new_gene = [elem]
    for i in range(len_pop-1):
        new_gene.append(mutation_3(elem, 10, C_min, C_max, S_min, S_max, N_min, N_max))
    
    return new_gene

def algo(plant, len_pop, n_iter, C_min, C_max, S_min, S_max, N_min, N_max, printed=0):
    """
    The genetic algorithm.

    Parameters
    ----------
    plant : H2 plant.
        A H2 plant. Suggestion : use the same H2 plant as the one used for sort_selection and initialization.
    len_pop : int.
        The length of the population (suggestion : around 40).
    n_iter : int.
        The number of iterations before providing the results (suggestion : around 50).

    C_min, C_max, S_min, S_max, N_min, N_max are all int and represent the minimum/maximum value 
    that can take the electrolyzer capacity (C), the storage capacity (S), and
    the number of trucks (N) of each member of the population.
    printed : int. Default value = 0
        If printed = 0 : it won't print the progress of the function (recommended while doing a multiprocess).
        Otherwise : it will.

    Returns
    -------
    best_config : list.
        The best member.
    best_LCOH : float
        The LCOH associated with the best member.
    LCOHs : list
        The list of LCOHs of all the members.
    new_gene : list
        The list of all the members of the last iteration.

    """
    # When do we start crazy mutation
    threshold_same_LCOH = 2 
    rank_digits_that_changes = 4
    precision = 10**(-(rank_digits_that_changes-1))

    # Initialisation
    initial_pop = initialization(plant, len_pop, C_min, C_max, S_min, S_max, N_min, N_max, printed)
    new_gene = initial_pop    
    LCOHs = []
    
    # Definition of a new generation
    
    from_best = 1 # The number of best from the previous generation that stays in the new generation
    mutation_of_from_best = 2 # The number of mutation of the from_best 
    child_per_best = 2 # The number of children from the best
    from_init = 10 # From a new initialization
    others = len_pop - from_best*(1 + mutation_of_from_best + child_per_best) - from_init # The others : childs from the remaining

    # Probability defintions
    p_crossover = 0.95
    p_mutation = 0.75
    
    # Compteur for crazy mutation
    cpt_mut = 0
    
    for k in range(n_iter):
        # Sort + selection + improver step
        sort_pop, sort_pop_LCOH, best_LCOH = sort_selection(plant, new_gene, 4, C_min, C_max, S_min, S_max, N_min, N_max)
        best_config = sort_pop[0]
        LCOHs.append(best_LCOH)
        
        best_pop_LCOH = sort_pop_LCOH[:from_best]
        new_gene = []
        
        # If there are converge issues, we do a crazy mutation
        last_LCOH = LCOHs[-threshold_same_LCOH:]
        
        # We count how many times the LCOH has not varied more than a specific value (precision)
        nb_time_same_LCOH = sum([LCOH - best_LCOH < precision for LCOH in last_LCOH])
        
        # If the LCOH hasn't varied much for a long period of time, we do crazy mutation
        if k >= threshold_same_LCOH and nb_time_same_LCOH == threshold_same_LCOH:
                cpt_mut += 1
                new_gene = crazy_mutation_3(plant, sort_pop[0], len_pop, C_min, C_max, S_min, S_max, N_min, N_max)
                # new_gene = crazy_mutation(sort_pop[0], len_pop, 100, 10 ,C_min, C_max, S_min, S_max, N_min, N_max, C_rate, S_rate, N_rate)
                # new_gene = crazy_mutation_2(plant, sort_pop[0], len_pop, 100, 15 ,C_min, C_max, S_min, S_max, N_min, N_max, C_rate, S_rate, N_rate, 0)
                if cpt_mut == 20:
                    # Too many crazy mutations, we can't find any better solution
                    print("Fin prématuré")
                    return best_config, best_LCOH, LCOHs, new_gene
                if printed != 0:
                    print(f"{k+1}/{n_iter} iterations. CRAZY MUTATION Best LCOH : {best_LCOH}, best config {best_config}.")

        
        # Else, we continue the classic generation creation
        else:            
            cpt_mut = 0
            # We add the best ONE in the new_gene
            new_gene.append(sort_pop[0])
            
            # We add mutations from THE best
            p = new_gene[0]
            for j in range(mutation_of_from_best):
                muted = mutation_3(p, 10 ,C_min, C_max, S_min, S_max, N_min, N_max)
                # muted = mutation_2(p, 2, 10, C_min, C_max, S_min, S_max, N_min, N_max, 0.05, 0.05, 0.05)
                new_gene.append(muted)
            
            # # We add the best ones        
            # for i in range(from_best):
            #     new_gene.append(sort_pop[i])
            
            # # print("We add the best ones ")
            # # print(new_gene)
            # 
            
            # # We add mutations from best
            # # tmp = new_gene.copy()
            # for i in range(from_best):
            #     # p = tmp[i]
            #     p = new_gene[i]
            #     for j in range(mutation_of_from_best):
            #         muted = mutation_3(p, 10 ,C_min, C_max, S_min, S_max, N_min, N_max)
            #         # muted = mutation_2(p, 2, 10, C_min, C_max, S_min, S_max, N_min, N_max, 0.05, 0.05, 0.05)
            #         new_gene.append(muted)

            # print("We add mutations from best")
            # print(new_gene)
            
            
            # We add children from best
            # tmp = new_gene.copy()
            # for i in range(from_best):
            p1 = best_pop_LCOH[0]
            for j in range(child_per_best):
                p2 = parent_selection_alea(sort_pop_LCOH)
                
                # child = crossover_1(p1, p2, 10)
                # child = crossover_2(p1, p2, 0.7, 10)
                # child = crossover_3(p1, p2, 10)
                child = crossover_4(p1, p2, 10)
                new_gene.append(child)
            
            # We add random members
            pop = initialization(plant, from_init, C_min, C_max, S_min, S_max, S_min, S_max, 0)
            new_gene += pop   
            
            # We add the others
            for i in range(others):
                p1 = parent_selection_alea(sort_pop_LCOH)
                p2 = parent_selection_alea(sort_pop_LCOH)
            
                # Crossover
                # child = crossover_1(p1, p2, p_crossover)
                # child = crossover_2(p1, p2, 0.7, p_crossover)
                # child = crossover_3(p1, p2, p_crossover)
                child = crossover_4(p1, p2, p_crossover)
                
                # Mutation
                # mut = mutation_2(child, p_mutation, 10, C_min, C_max, S_min, S_max, N_min, N_max, C_rate, S_rate, N_rate)
                mut = mutation_3(child, p_mutation ,C_min, C_max, S_min, S_max, N_min, N_max)
                # New child
                new_gene.append(mut)
                
                # print("We add the others")
                # print(new_gene)
                
            if printed != 0:
                print(f"{k+1}/{n_iter} iterations. Best LCOH : {best_LCOH}, best config {best_config}.")
            
            # print(f"{i+1}/{n_iter} iterations.")
    # Sort step
    sort_pop, sort_pop_LCOH, best_LCOH = sort_selection(plant, new_gene, 1, C_min, C_max, S_min, S_max, N_min, N_max)
    LCOHs.append(best_LCOH)
        
    best_config = sort_pop[0]
    
    return best_config, best_LCOH, LCOHs, new_gene
 
def optimization(plant, n_best_design, n_iter, len_pop, C_min, C_max, S_min, S_max, N_min, N_max, printed=1):
    """
    

    Parameters
    ----------
  
    n_best_design : int.
        Number of runs of 'algo'.
    The other arguments are the same as the one defined in the algo function.

    Returns
    -------
    [Cs, Ss, Ns, LCOHs] : list
        A list containing a list of the best designs, and their associated LCOH
        Cs = [optimised C of the run 1, optimised C of the run 2, ...]
        Ss = [optimised S of the run 1, optimised S of the run 2, ...]
        Ns = [optimised N of the run 1, optimised N of the run 2, ...]
        LCOHs = [LCOH of the optimised design from the run 1, LCOH of the optimised design from the run 2, ...]
    [Cs[min_i], Ss[min_i], Ns[min_i], min(LCOHs)] : list
        The electrolyzer capacity, storage capacity, number of trucks of the best design among all, and its LCOH associated

    """
    Cs, Ns, Ss = [], [], []
    LCOHs = []
    
    for i in range(n_best_design):
        
        best_config, best_LCOH, useless, useless2 = algo(plant, len_pop, n_iter, C_min, C_max, S_min, S_max, N_min, N_max, 0)
        
        best_C, best_S, best_N = best_config
        
        Cs.append(best_C)
        Ss.append(best_S)
        Ns.append(best_N)
        LCOHs.append(best_LCOH)
        if printed != 0:
            print(f"Algo {i+1}/{n_best_design} completed")
    
    min_i = 0
    for i in range(len(LCOHs)):
        if LCOHs[i] == min(LCOHs):
            min_i = i
            break
    
    return [Cs, Ss, Ns, LCOHs], [Cs[min_i], Ss[min_i], Ns[min_i], min(LCOHs)]

def sensitivity(val_min, val_max, nb, what):    
    """
    To create a "sensitivity analysis" on the different variables of the model.
    It's not a proper sensitivity analysis, because it's not applied on
    a parameter of the model.

    Parameters
    ----------
    val_min : int.
        Minimum value for the analysis.
    val_max : int
        Maximum value for the analysis.
    nb : int
        Number of points used for the analysis.
    what : str
        If what = 'C' : analysis done on the electrolyzer capacity.
        If what = 'S' : analysis done on the storage capacity.
        If what = 'N' : analysis done on the number of trucks.
        
    Returns
    -------
    None.

    """
    r = np.linspace(val_min, val_max, nb, dtype=int)

    LCOHs = []
    wasted_H2s = []
    wasted_Ps = []
    
    cpt = 0
    
    for val in r:
        
        plant = empty_plant()   
        plant.set_electrolyzer_capacity(49161)
        plant.set_storage_capacity(326)
        plant.set_number_of_trucks(11)
        
        if what == 'C':
            plant.set_electrolyzer_capacity(val)
        elif what == 'S':
            plant.set_storage_capacity(val)
        elif what == 'N':
            plant.set_number_of_trucks(val)
        cpt += 1
        print(f"{cpt}/{len(r)} iterations.")
        
        plant.electrolyzer_production()
        plant.hydrogen_management()
        plant.get_KPI()
        
        LCOHs.append(plant.KPI['LCOH'])
        wasted_H2s.append(plant.KPI['wasted_hydrogen']*100)
        wasted_Ps.append(plant.KPI['wasted_power']*100)
        
    fig, ax = plt.subplots()
    
    ax.plot(r, LCOHs)
    
    ax.set_ylabel("LCOH [EUR/kWh]")
    if what == 'C':
        ax.set_xlabel("Electrolyzer capacity [kW]")
        t = "electrolyzer capacity"
    elif what == 'S':
        ax.set_xlabel("Storage capacity [m3]")
        t = "storage capacity"
    elif what == 'N':
        ax.set_xlabel("Number of tube trailers")
        t = "number of tube trailers"
    
    fig.suptitle(f"Imact of the {t} on the LCOH") 
    fig.show()
    
    
    fig, ax = plt.subplots()
    
    ax.plot(r, wasted_H2s)
    
    ax.set_ylabel("Wasted hydrogen [%]")
    if what == 'C':
        ax.set_xlabel("Electrolyzer capacity [kW]")
        t = "electrolyzer capacity"
    elif what == 'S':
        ax.set_xlabel("Storage capacity [m3]")
        t = "storage capacity"
    elif what == 'N':
        ax.set_xlabel("Number of tube trailers")
        t = "number of tube trailers"
    
    fig.suptitle(f"Imact of the {t} on the share of hydrogen wasted") 
    fig.show()
    
    
    fig, ax = plt.subplots()
    
    ax.plot(r, wasted_Ps)
    
    ax.set_ylabel("Wasted power [%]")
    if what == 'C':
        ax.set_xlabel("Electrolyzer capacity [kW]")
        t = "electrolyzer capacity"
    elif what == 'S':
        ax.set_xlabel("Storage capacity [m3]")
        t = "storage capacity"
    elif what == 'N':
        ax.set_xlabel("Number of tube trailers")
        t = "number of tube trailers"
    
    fig.suptitle(f"Imact of the {t} on the share of power wasted") 
    fig.show()
    
