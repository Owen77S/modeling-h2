import optimization as m
import matplotlib.pyplot as plt

C = 53630
S = 118
N = 8
y = 40
p = m.define_plant(C, S, N)

p.KPI['H2'] = sum(p.res["mass_H2"])
total_sold_decompressed = sum(p.res["sold"])/(p.gas_model["P_op"]/p.gas_model["P_out"])**(1/p.gas_model["n"])
p.KPI["H2_sold"] = (p.gas_model["M"]*p.gas_model["P_op"]*10**5)*total_sold_decompressed/(p.gas_model["R"]*(p.gas_model["T_op"]+273.15))


storage_in_kg = ((p.gas_model['P_out']*1e5)*p.var['storage_capacity']*p.gas_model['M'])/(p.gas_model['R']*(p.gas_model['T_op']+273.15))
CAPEX_PEM = p.economics['install_fee']*p.var['electrolyzer_capacity']
CAPEX_storage = 490*storage_in_kg
CAPEX_selling = 93296 + p.var["number_of_trucks"]*610000
CAPEX = CAPEX_PEM + CAPEX_storage + CAPEX_selling

OPEX_PEM = p.economics['OPEX_PEM']*p.var['electrolyzer_capacity'] 
water_price = p.economics['water_price']*p.economics["water_consumption"]*p.KPI['H2']
OPEX_compressor = 4665
OPEX_selling = 30500*p.var['number_of_trucks']
OPEX = OPEX_PEM + water_price + OPEX_compressor + OPEX_selling

energy_in_H2 = p.KPI['H2_sold']*p.param['LHV_kg']

years = range(1, y+1)
LCOH1 = [p.KPI['LCOH']]

for year in years:
    if year == 1:
        c= 0
    else: 
        price = CAPEX + OPEX*year
        prod = energy_in_H2*year
        LCOH1.append(price/prod)
    
fig, ax = plt.subplots()

ax.plot(years, LCOH1,label='First without WACC, eps')


# TEST 2

p2 = m.define_plant(C, S, N)

LCOH2 = [p2.KPI['LCOH']]
years = range(1, y+1)

for year in years:

    if year == 1:
        c = 0
    else:
        p2.KPI['H2'] = sum(p2.res["mass_H2"])
        total_sold_decompressed = sum(p2.res["sold"])/(p2.gas_model["P_op"]/p2.gas_model["P_out"])**(1/p2.gas_model["n"])
        p2.KPI["H2_sold"] = (p2.gas_model["M"]*p2.gas_model["P_op"]*10**5)*total_sold_decompressed/(p2.gas_model["R"]*(p2.gas_model["T_op"]+273.15))
        
        
        storage_in_kg = ((p2.gas_model['P_out']*1e5)*p2.var['storage_capacity']*p2.gas_model['M'])/(p2.gas_model['R']*(p2.gas_model['T_op']+273.15))
        CAPEX_PEM = p2.economics['install_fee']*p2.var['electrolyzer_capacity']
        CAPEX_storage = 490*storage_in_kg
        CAPEX_selling = 93296 + p2.var["number_of_trucks"]*610000
        CAPEX = CAPEX_PEM + CAPEX_storage + CAPEX_selling
        
        OPEX_PEM = p2.economics['OPEX_PEM']*p2.var['electrolyzer_capacity'] 
        water_price = p2.economics['water_price']*p2.economics["water_consumption"]*p2.KPI['H2']
        OPEX_compressor = 4665
        OPEX_selling = 30500*p2.var['number_of_trucks']
        OPEX = OPEX_PEM + water_price + OPEX_compressor + OPEX_selling
        
        replacement_PEM = (year//7.6)*630*p2.var['electrolyzer_capacity']
        kg_to_m3 =  1*p2.param["correlation_to_reality"]*p2.gas_model["R"]*(p2.gas_model["T_op"]+273.15)/(p2.gas_model["M"]*p2.gas_model["P_op"]*10**5)
        replacement_storage = (year//25)*470*p2.var['storage_capacity']/kg_to_m3
        
        replacement = replacement_PEM + replacement_storage
        
        energy_in_H2 = p2.KPI['H2_sold']*p2.param['LHV_kg'] #kWh
        
        LCOH2.append((CAPEX+OPEX*year+replacement)/(energy_in_H2*year))



ax.plot(years, LCOH2, label="Second without WACC/esp")

p3 = m.define_plant(C, S, N)
LCOH3 = [p3.KPI['LCOH']]

for year in years:    
    if year==1:
        c = 0
    else:
        p3.KPI['H2'] = sum(p3.res["mass_H2"])
        total_sold_decompressed = sum(p3.res["sold"])/(p3.gas_model["P_op"]/p3.gas_model["P_out"])**(1/p3.gas_model["n"])
        p3.KPI["H2_sold"] = (p3.gas_model["M"]*p3.gas_model["P_op"]*10**5)*total_sold_decompressed/(p3.gas_model["R"]*(p3.gas_model["T_op"]+273.15))
        
        
        storage_in_kg = ((p3.gas_model['P_out']*1e5)*p3.var['storage_capacity']*p3.gas_model['M'])/(p3.gas_model['R']*(p3.gas_model['T_op']+273.15))
        CAPEX_PEM = p3.economics['install_fee']*p3.var['electrolyzer_capacity']
        CAPEX_storage = 490*storage_in_kg
        CAPEX_selling = 93296 + p3.var["number_of_trucks"]*610000
        CAPEX = CAPEX_PEM + CAPEX_storage + CAPEX_selling
        
        OPEX_PEM = p3.economics['OPEX_PEM']*p3.var['electrolyzer_capacity'] 
        water_price = p3.economics['water_price']*p3.economics["water_consumption"]*p3.KPI['H2']
        OPEX_compressor = 4665
        OPEX_selling = 30500*p3.var['number_of_trucks']
        OPEX = OPEX_PEM + water_price + OPEX_compressor + OPEX_selling
        
        replacement_PEM = (year//7.6)*630*p3.var['electrolyzer_capacity']
        kg_to_m3 =  1*p3.param["correlation_to_reality"]*p3.gas_model["R"]*(p3.gas_model["T_op"]+273.15)/(p3.gas_model["M"]*p3.gas_model["P_op"]*10**5)
        replacement_storage = (year//25)*470*p3.var['storage_capacity']/kg_to_m3
        
        replacement = replacement_PEM + replacement_storage
        
        WACC = 7.5/100
        eps = 0.3/100
        num = CAPEX + replacement + sum([OPEX/(1+WACC)**n for n in range(1, year+1)])
        
        energy_in_H2 = p3.KPI['H2_sold']*p3.param['LHV_kg'] #kWh
        den = energy_in_H2*sum([(1-eps)**n/(1+WACC)**n for n in range(1, year+1)])
        
        p3.KPI['LCOH'] = num/den
        LCOH3.append((p3.KPI['LCOH']))
    
ax.plot(years, LCOH3, label='Second with WACC')

ax.legend()


fig.show()