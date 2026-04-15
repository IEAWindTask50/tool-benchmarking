import time
import numpy as np
import pandas as pd
import os

from hydesign.assembly.hpp_assembly import hpp_model
from hydesign.examples import examples_filepath

sites = pd.read_csv('overview.csv', index_col=0)
example = 0

ex_site = sites.iloc[int(example),:]

print('Selected example site:')
print('---------------------------------------------------')
print(ex_site.T)

longitude = ex_site['longitude']
latitude = ex_site['latitude']
altitude = ex_site['altitude']
input_ts_fn = os.path.join(os.getcwd(), ex_site['input_ts_fn'])
sim_pars_fn = os.path.join(os.getcwd(), ex_site['sim_pars_fn'])

hpp = hpp_model(
        latitude=latitude,
        longitude=longitude,
        altitude=altitude,
        num_batteries = 10,
        work_dir = './',
        sim_pars_fn = sim_pars_fn,
        input_ts_fn = input_ts_fn,
)

inputs = dict(
clearance =	60,
sp =	287.6391934,
p_rated =	10,
Nwt	=	25,
wind_MW_per_km2 =	4.782274215,
solar_MW =	223.2191223,
surface_tilt =	50,
surface_azimuth = 210,
DC_AC_ratio	=1.295598428,
b_P =75.27402422,
b_E_h=	4.451612903,
cost_of_battery_P_fluct_in_peak_price_ratio	=0.580645161,)

start = time.time()
outs = hpp.evaluate(**inputs)
hpp.print_design([v for k,v in inputs.items()], outs)
end = time.time()
print('exec. time [min]:', (end - start)/60 )
print(hpp.prob['NPV_over_CAPEX'])

from hopp.simulation import HoppInterface

hi = HoppInterface(os.path.join(os.getcwd(), "hopp_input/08-wind-solar-france.yaml"))

hi.system.wind._financial_model.FinancialParameters.real_discount_rate = 6
hi.system.pv._financial_model.FinancialParameters.real_discount_rate = 6
hi.system.battery._financial_model.FinancialParameters.real_discount_rate = 6
hi.system.grid._financial_model.FinancialParameters.real_discount_rate = 6

hi.system.wind._financial_model.FinancialParameters.inflation_rate = 2
hi.system.pv._financial_model.FinancialParameters.inflation_rate = 2
hi.system.battery._financial_model.FinancialParameters.inflation_rate = 2
hi.system.grid._financial_model.FinancialParameters.inflation_rate = 2

# Federal tax rate 21%, set state tax rate to 4% (total 25% tax rate)
hi.system.wind._financial_model.FinancialParameters.state_tax_rate = [4]*25
hi.system.pv._financial_model.FinancialParameters.state_tax_rate = [4]*25
hi.system.battery._financial_model.FinancialParameters.state_tax_rate = [4]*25
hi.system.grid._financial_model.FinancialParameters.state_tax_rate = [4]*25

hi.simulate(25)

hybrid_plant = hi.system

annual_energies = hybrid_plant.annual_energies
npvs = hybrid_plant.net_present_values
cf = hybrid_plant.capacity_factors

wind_installed_cost = hybrid_plant.wind.total_installed_cost
solar_installed_cost = hybrid_plant.pv.total_installed_cost
battery_installed_cost = hybrid_plant.battery.total_installed_cost
hybrid_installed_cost = hybrid_plant.grid.total_installed_cost

print("Wind Installed Cost [M$]: {}".format(wind_installed_cost/1e6))
print("Solar Installed Cost [M$]: {}".format(solar_installed_cost/1e6))
print("Battery Installed Cost [M$]: {}".format(battery_installed_cost/1e6))
print("Hybrid Installed Cost [M$]: {}\n".format(hybrid_installed_cost/1e6))

print("Wind NPV [M$]: {}".format(hybrid_plant.net_present_values.wind/1e6))
print("Solar NPV [M$]: {}".format(hybrid_plant.net_present_values.pv/1e6))
print("Hybrid NPV [M$]: {}\n".format(hybrid_plant.net_present_values.hybrid/1e6))

print("Annual Energies [kW]",annual_energies)
print("Capacity Factors",cf)
print("NPV [$]", npvs)

print("LCOE [cents/kWh]", hybrid_plant.lcoe_real,"\n")

print("Hybrid LCOE [$/MWh]", hybrid_plant.lcoe_real['hybrid']*10)

import json

with open("singleowner.json", 'w') as f:
    dat = hybrid_plant.grid._financial_model.export()
    d = dict()
    for k, v in dat.items():
        d.update(v)
    json.dump(d, f)

    euro_per_dollar = 1.0
df = pd.DataFrame({'HOPP': [wind_installed_cost/1e6*euro_per_dollar,
                            solar_installed_cost/1e6*euro_per_dollar,
                            battery_installed_cost/1e6*euro_per_dollar,
                            hybrid_installed_cost/1e6*euro_per_dollar,
                            hybrid_plant.net_present_values.hybrid/1e6*euro_per_dollar,
                           ],
                   'HyDesign': [float(hpp.prob['wpp_cost.CAPEX_w']/1e6),
                                float(hpp.prob['pvp_cost.CAPEX_s']/1e6),
                                float(hpp.prob['battery_cost.CAPEX_b']/1e6),
                                float(hpp.prob['CAPEX']/1e6),
                                float(hpp.prob['NPV']/1e6),
                                ] },
                  index=['CAPEX Wind [M€]',
                         'CAPEX Solar [M€]',
                         'CAPEX Battery [M€]',
                         'CAPEX Hybrid [M€]',
                         'NPV Hybrid [M€]',
                        ])
df
hpp.prob.model.list_outputs()

