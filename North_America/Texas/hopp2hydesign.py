# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 11:51:29 2025

@author: mikf
"""

import yaml
import os
import numpy as np
import pandas as pd


folder = 'hopp_input'
file = '34.22_-102.75_windtoolkit_2013_60min_100m_120m.srw'
path = os.path.join(folder, file)
df1 = pd.read_csv(path, skiprows=[0,1,3,4])

folder = 'hopp_input'
file = '34.22_-102.75_psmv3_60_2013.csv'
path = os.path.join(folder, file)
df12 = pd.read_csv(path, skiprows=[0,1])

folder = 'hopp_input'
file = 'texas-price.csv'
path = os.path.join(folder, file)
df13 = pd.read_csv(path, skiprows=[])

folder = 'hydesign_input'
file = 'hpp_pars.yml'
path = os.path.join(folder, file)
with open(path) as file:
    data2 = yaml.load(file, Loader=yaml.FullLoader)

folder = 'hydesign_input'
file = 'input_ts_template.csv'
path = os.path.join(folder, file)
df2 = pd.read_csv(path, index_col=0)
df3 = pd.DataFrame({'WS_100': df1['Speed'].values,
                    'WS_120': df1['Speed.1'].values,
                    'temp_air_1': df12['Temperature'].values,
                    'ghi': df12['GHI'].values,
                    'dni': df12['DNI'].values,
                    'dhi': df12['DHI'].values,
                    'Price': df13['$/MWh'].values,
                    }, index=df2.index)
file = 'input_ts_texas.csv'
path = os.path.join(folder, file)
df3.to_csv(path)

folder = 'hopp_input'
file = 'hopp_config.yaml'
path = os.path.join(folder, file)
with open(path) as file:
    data3 = yaml.load(file, Loader=yaml.FullLoader)

folder = 'hopp_input'
file = 'lbw_6MW.yaml'
path = os.path.join(folder, file)
with open(path) as file:
    data = yaml.load(file, Loader=yaml.FullLoader)
rotor_diameter = data['rotor_diameter']
hub_height = data['hub_height']

p_rated = data3['technologies']['wind']['turbine_rating_kw'] / 10 ** 3
# p_rated = turbine_rating = 6  # MW why is this commented out in the input file?
area = np.pi * rotor_diameter ** 2 / 4
sp = p_rated * 10 ** 6 / area
clearance = hub_height - rotor_diameter / 2

data2['latitude'] = data3['site']['data']['lat']
data2['longitude'] = data3['site']['data']['lon']
data2['altitude'] = data3['site']['data']['eev']
data2['year'] = str(data3['site']['data']['year'])

data2['p_rated'] = data2['p_rated_ref'] = p_rated
data2['rotor_diameter'] = data2['d_ref'] = rotor_diameter
data2['hub_height'] = data2['hh_ref'] = hub_height
data2['clearance'] = clearance
data2['sp'] = sp

data2['Nwt'] = data3['technologies']['wind']['num_turbines']
data2['solar_MW'] = data3['technologies']['pv']['system_capacity_kw'] / 10 ** 3
data2['b_P'] = data3['technologies']['battery']['system_capacity_kw'] / 10 ** 3
data2['b_E_h'] = data3['technologies']['battery']['system_capacity_kwh'] / data3['technologies']['battery']['system_capacity_kw']
data2['battery_depth_of_discharge'] = 1 - data3['technologies']['battery']['minimum_SOC'] / 100

data2['G_MW'] = data3['technologies']['grid']['interconnect_kw'] / 10 ** 3







