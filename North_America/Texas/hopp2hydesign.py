# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 11:51:29 2025

@author: mikf
"""

import yaml
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

base_dir = os.path.dirname(os.path.abspath(__file__))

folder = 'hopp_input'
file = '34.22_-102.75_windtoolkit_2013_60min_100m_120m.srw'
path = os.path.join(base_dir, folder, file)
df1 = pd.read_csv(path, skiprows=[0,1,3,4])

folder = 'hopp_input'
file = '34.22_-102.75_psmv3_60_2013.csv'
path = os.path.join(base_dir, folder, file)
df12 = pd.read_csv(path, skiprows=[0,1])

folder = 'hopp_input'
file = 'texas-price.csv'
path = os.path.join(base_dir, folder, file)
df13 = pd.read_csv(path, skiprows=[])

folder = 'hydesign_input'
file = 'hpp_pars.yml'
modified_file = 'hpp_pars_converted.yml'
path = os.path.join(base_dir, folder, file)
path2 = os.path.join(base_dir, folder, modified_file)
with open(path) as file:
    data2 = yaml.load(file, Loader=yaml.FullLoader)

folder = 'hydesign_input'
file = 'input_ts_template.csv'
path = os.path.join(base_dir, folder, file)
df2 = pd.read_csv(path, index_col=0)
df3 = pd.DataFrame({'WS_100': df1['Speed'].values,
                    'WD_100': df1['Direction'].values,
                    'WS_120': df1['Speed.1'].values,
                    'WD_120': df1['Direction.1'].values,
                    'temp_air_1': df12['Temperature'].values,
                    'ghi': df12['GHI'].values,
                    'dni': df12['DNI'].values,
                    'dhi': df12['DHI'].values,
                    'Price': df13['$/MWh'].values + 1e-9,
                    }, index=df2.index)
file = 'input_ts_texas.csv'
path = os.path.join(base_dir, folder, file)
df3['temp_air_1'] = df3['temp_air_1'] + 273.15 #HOPP has degree celsius
df3.to_csv(path)

folder = 'hopp_input'
file = 'hopp_config.yaml'
path = os.path.join(base_dir, folder, file)
with open(path) as file:
    data3 = yaml.load(file, Loader=yaml.FullLoader)

folder = 'hopp_input'
file = 'lbw_6MW.yaml'
path = os.path.join(base_dir, folder, file)
with open(path) as file:
    data = yaml.load(file, Loader=yaml.FullLoader)
rotor_diameter = data['rotor_diameter']
hub_height = data['hub_height']

p_rated = data3['technologies']['wind']['turbine_rating_kw'] / 10 ** 3
# p_rated = turbine_rating = 6  # MW why is this commented out in the input file?
# area = np.pi * rotor_diameter ** 2 / 4
# sp = p_rated * 10 ** 6 / area
# clearance = hub_height - rotor_diameter / 2

# data2['latitude'] = data3['site']['data']['lat']
# data2['longitude'] = data3['site']['data']['lon']
# data2['altitude'] = data3['site']['data']['elev']
# data2['year'] = str(data3['site']['data']['year'])

data2['p_rated'] = p_rated
data2['p_rated_ref'] = p_rated
data2['d'] = rotor_diameter
data2['d_ref'] = rotor_diameter
data2['hh'] = hub_height
data2['hh_ref'] = hub_height
# data2['clearance'] = clearance
# data2['sp'] = sp

# data2['Nwt'] = data3['technologies']['wind']['num_turbines']
# data2['solar_MW'] = data3['technologies']['pv']['system_capacity_kw'] / 10 ** 3
# data2['b_P'] = data3['technologies']['battery']['system_capacity_kw'] / 10 ** 3
# data2['b_E_h'] = data3['technologies']['battery']['system_capacity_kwh'] / data3['technologies']['battery']['system_capacity_kw']
# data2['battery_depth_of_discharge'] = 1 - data3['technologies']['battery']['minimum_SOC'] / 100

data2['G_MW'] = data3['technologies']['grid']['interconnect_kw'] / 10 ** 3

data2['wind_turbine_cost'] = data3['config']['cost_info']['wind_installed_cost_mw']
data2['wind_civil_works_cost'] = 0 #covered in the wind turbine cost
data2['wind_fixed_onm_cost'] = data3['config']['cost_info']['wind_om_per_kw'] * 1000
data2['wind_variable_onm_cost'] = 0 #hopp covers all the wind o&m cost in one variable

data2['solar_PV_cost'] = data3['config']['cost_info']['solar_installed_cost_mw']
data2['solar_hardware_installation_cost'] = 0 #covered in the pv cost
data2['solar_inverter_cost'] = 0 #covered in the pv cost
data2['solar_fixed_onm_cost'] = data3['config']['cost_info']['pv_om_per_kw'] * 1000

data2['battery_energy_cost'] = data3['config']['cost_info']['storage_installed_cost_mwh']
data2['battery_power_cost'] = data3['config']['cost_info']['storage_installed_cost_mw']
data2['battery_BOP_installation_commissioning_cost'] = 0 #covered in the storage power cost
data2['battery_control_system_cost'] = 0 #covered in the storage power cost
data2['battery_energy_onm_cost'] = data3['config']['cost_info']['battery_om_per_kw'] * 1000

data2['battery_depth_of_discharge'] = 1 - data3['technologies']['battery']['minimum_SOC'] / 100

###
### THE PARAMETERS BELOW ARE IN THE HYDESIGN INPUT LIST BUT NOT IN THE HOPP ONE
###

data2['N_life'] # I cannot see it in the hopp_input.yml file

data2['wpp_efficiency'] = 0.95 # Check if the wind power generation in both the HOPP and HyDesign evaluations match, and change the efficiency if needed.
# data2['wind_deg_yr'] # I cannot see it in the hopp_input.yml file
# data2['wind_deg'] = 0 # To make it consistent with the HOPP.yaml file
data2['share_WT_deg_types'] = 0 # To make it consistent with the HOPP.yaml file

data2['land_use_per_solar_MW'] = 0 # Covered in the wind turbine land use

# data2['pv_deg_yr'] # I cannot see it in the hopp_input.yml file
# data2['pv_deg'] = 0 # To make it consistent with the HOPP.yaml file

# data2['battery_charge_efficiency'] # I cannot see it in the hopp_input.yml file 
data2['battery_price_reduction_per_year'] = 0 # Not covered in HOPP
# data2['min_LoH'] # I cannot see it in the hopp_input.yml file

data2['n_full_power_hours_expected_per_day_at_peak_price'] = 0 # Not covered in HOPP
data2['peak_hr_quantile'] = 0 # Not covered in HOPP

data2['hpp_BOS_soft_cost'] = 0 # I cannot see it in the hopp_input.yml file 
# data2['hpp_grid_connection_cost']# I cannot see it in the hopp_input.yml file
data2['land_cost'] = 0 # I cannot see it in the hopp_input.yml file

with open(path2, 'w') as file:
    yaml.dump(data2, file, sort_keys=False)


### DHI PLOT ####
plt.figure(figsize=(14, 5))
plt.plot(df3['dhi'].values, label='HyDesign DHI')
plt.plot(df12['DHI'].values, label='HOPP DHI')
plt.xlabel('time (h)')
plt.ylabel('DHI [W/m²]')
plt.title('DHI - HyDesign vs HOPP')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

ax1.plot(df3['dhi'].values, color='orange')
ax1.set_title('DHI - HyDesign')
ax1.set_ylabel('DHI [W/m²]')
ax1.grid(True)

ax2.plot(df12['DHI'].values, color='blue')
ax2.set_title('DHI - HOPP')
ax2.set_xlabel('time (h)')
ax2.set_ylabel('DHI [W/m²]')
ax2.grid(True)

plt.tight_layout()
plt.show()

### DNI PLOT ####
plt.figure(figsize=(14, 5))
plt.plot(df3['dni'].values, label='HyDesign DNI')
plt.plot(df12['DNI'].values, label='HOPP DNI')
plt.xlabel('time (h)')
plt.ylabel('DNI [W/m²]')
plt.title('DNI - HyDesign vs HOPP')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

ax1.plot(df3['dni'].values, color='orange')
ax1.set_title('DNI - HyDesign')
ax1.set_ylabel('DNI [W/m²]')
ax1.grid(True)

ax2.plot(df12['DNI'].values, color='blue')
ax2.set_title('DNI - HOPP')
ax2.set_xlabel('time (h)')
ax2.set_ylabel('DNI [W/m²]')
ax2.grid(True)

plt.tight_layout()
plt.show()

### GHI PLOT ####
plt.figure(figsize=(14, 5))
plt.plot(df3['ghi'].values, label='HyDesign GHI')
plt.plot(df12['GHI'].values, label='HOPP GHI')
plt.xlabel('time (h)')
plt.ylabel('GHI [W/m²]')
plt.title('GHI - HyDesign vs HOPP')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

ax1.plot(df3['ghi'].values, color='orange')
ax1.set_title('GHI - HyDesign')
ax1.set_ylabel('GHI [W/m²]')
ax1.grid(True)

ax2.plot(df12['GHI'].values, color='blue')
ax2.set_title('GHI - HOPP')
ax2.set_xlabel('time (h)')
ax2.set_ylabel('GHI [W/m²]')
ax2.grid(True)

plt.tight_layout()
plt.show()