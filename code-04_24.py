import pyomo.environ as pyomo
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

class H2plant():
    
    def __init__(self):
        self.duration = 8760
        self.data = {"WP" : [0]*self.duration,
                     "NP" : [0]*self.duration,
                     "excess_power" : [0]*self.duration,
                     "supply_power"  : [0]*self.duration}
        self.economics = {"install_fee" : 1800, #€/kW
                          "OPEX_PEM" : 54, #€/kW/year
                          "water_price" : 3e-3, #€/kg
                          "water_consumption" : 9, #kG H2O/kg H2
                          "price_H2" : 2.7} #€/kg
        self.res = {"H2" : [0]*self.duration,
                    "total_H2" : 0,
                    "LCOE" : 0,
                    "wasted_power" : 0}
        self.param = {
            "eta_F_characteristic" : 0.36697247706,
            "grid_limit" : 1319414, #kW
            "HHV" : 33.3 #kWh/kg
            }
        
        
    
    def get_data_from_excel(self, path_excel = "data_2.xlsx"):
        '''
        Get the parameters (wind and nuclear power plant production) from Excel
        Update self.data
        Parameters
        ----------
        path_excel : str, optional
            The excel file path. The default is "data_2.xlsx".

        Returns
        -------
        None.

        '''
        
        wind_nuclear = pd.read_excel("data_2.xlsx",
                                     usecols = 'A:B')
        
        self.data["WP"] = wind_nuclear["Wind power [kW]"].tolist()
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
            if  excess_electricity >= 0:
                #We produce to much energy : congestion
                self.data["excess_power"][t] = excess_electricity
            else:
                #We don't produce to much energy : no congestion
                self.data["excess_power"][t] = 0
    
    def n_F(self, capacity, supply_power):
        """
        To compute faradic efficiency.

        Parameters
        ----------
        capacity : int
            The capacity of the electrolyzer [kW].
        supply_power : int
            The supply power to the electrolyzer [kW].

        Returns
        -------
        n_F
            The faradic efficiency [%].

        """
        return 1 - np.exp(-(supply_power/capacity)/self.param["eta_F_characteristic"])
    
    def electrolyzer_production(self, capacity):
        """
        Determine the hourly hydrogen production and the total hydrogen produced throughout
        the year by the electrolyzer of a specific capacity.        

        Parameters
        ----------
        capacity : int
            The capacity of the electrolyzer [kW].

        Returns
        -------
        None.

        """
        #Auxiliaries power requirement : 3% of the power supply
        n_aux = 1 - 0.03
        for t in range(self.duration):
            if self.data["excess_power"][t] >= capacity:
                #The supply power can't excess the electrolyser's capacity
                self.data["supply_power"][t] = capacity
            else:
                self.data["supply_power"][t] = self.data["excess_power"][t]
            
            self.res["H2"][t] = n_aux*self.n_F(capacity, self.data["supply_power"][t])*self.data["supply_power"][t]
        
        self.res["total_H2"] = sum(self.res["H2"])
        self.res['wasted_power'] = sum(self.data["supply_power"])/sum(self.data["excess_power"])
        
    def LCOE(self, capacity):
        CAPEX = self.economics['install_fee']*capacity
        OPEX = self.economics['OPEX_PEM']*capacity*1 + self.economics['water_price']*self.economics["water_consumption"]*self.res['total_H2']     
        energy_in_H2 = self.res['total_H2']*self.param['HHV'] #kWh
        self.res['LCOE'] = (CAPEX+OPEX)/energy_in_H2
    
    def shuffle(self, what):
        """
        Shuffle the yield power from the wind and the nuclear power plant.
        For sensitivity analysis.
        
        Parameters
        ----------
        whatt : str. Default value : "all".
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
    
    def speed(self):
        self.get_data_from_excel()
        self.power_manager()
        self.electrolyzer_production()

#STUDY PART 1

def optimisation(mini, maxi, incr):
    
    plant = H2plant()
    plant.get_data_from_excel()
    plant.power_manager()
    total_H2_list = []
    total_LCOE = []
    
    capacities = range(mini, maxi, incr)
    cpt = 0
    
    for capacity in capacities:
        plant.electrolyzer_production(capacity)
        plant.LCOE(capacity)
        total_H2_list.append(plant.res["total_H2"])
        total_LCOE.append(plant.res['LCOE'])
        
        cpt += 1
        if cpt%50 == 0:
            print(100*cpt/len(capacities), "% completed.")
            
    plt.figure()
    plt.xlabel("Capacity [kW]")
    plt.ylabel("Total hydrogen produced [kg]")
    plt.plot(capacities, total_H2_list)
    
    plt.figure()
    plt.xlabel("Capacity [kW]")
    plt.ylabel("LCOE [€/kWh]")
    plt.plot(capacities, total_LCOE)
    
    plt.figure()
    plt.ylabel("LCOE [€/kWh]")
    plt.xlabel("Total hydrogen produced [kg]")
    plt.plot( total_H2_list,total_LCOE,'+')



#STUDY PART 2

def sensitivity_analysis(H2P, nb_shuffle, capacity, grid_limit = 1400+1455460):
    """
    Sensitivity analysis of the form of the yield power of nuclear + wind

    Parameters
    ----------
    H2P : H2_plant
        The H2_plant instance that you use.
    nb_shuffle : int
        The number of shuffling you want to do for the sensitivity analysis.
    grid_limit : int, optional.
        The grid limit as defined previously. The default value is  1400+1455460.
    capacity : int
        The electrolyzer capacity has defined previously.

    Returns
    -------
    design_capacity : list
        The value for the design capacity for each shuffled sample of wind + nuclear yield power.

    """
    if grid_limit != H2P.param["grid_limit"]:
        #Then the grid limit have to change
        H2P.change_grid_limit(grid_limit)

    design_capacity = [0]*nb_shuffle
    for i in range(nb_shuffle):
        H2P.shuffle("NP")
        H2P.shuffle("WP")
        H2P.power_manager(grid_limit)
        H2P.electrolyzer_production(capacity)
        design_capacity[i] = H2P["total_H2"]
    
    return design_capacity

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

def show_data():
    plant = H2plant()
    plant.get_data_from_excel()
    plant.power_manager()
    WP = plant.data["WP"]
    NP = plant.data["NP"]
    
    figure, axis = plt.subplots(2, 1)
    axis[0].plot(WP)
    axis[0].set_title("Wind power [kW]")
    
    temps = range(1, plant.duration+1)
    tmp = [WP[t-1] + NP[t-1] for t in temps]
    axis[1].plot(temps, tmp)
    axis[1].plot(temps, [plant.param['grid_limit']]*plant.duration)
    axis[1].set_title("Wind power + Nuclear power [kW]")
    
    plt.show()

def close():
    plt.close("all")