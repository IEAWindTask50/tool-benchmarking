# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd

#%% WIND

hopp_ref_wind_path = r"/Users/kbrunik/github/tool-benchmarking/Europe/hopp_input/france.srw"
#"C:\Sandbox\Repo\NREL\HOPP\resource_files\wind\35.2018863_-101.945027_windtoolkit_2012_60min_80m_100m.srw"
hopp_ref_df = pd.read_csv(hopp_ref_wind_path, skiprows=[0,1,3,4])

hopp_path = r"/Users/kbrunik/github/tool-benchmarking/Europe/hopp_input/france-2.yaml" #"C:\Sandbox\Repo\NREL\HOPP\resource_files\wind\france.srw"

hyd_wind_path = r"/Users/kbrunik/github/tool-benchmarking/Europe/hyd_input/input_ts_France_good_wind.csv"
#"C:\Sandbox\Repo\TOPFARM\hydesign\hydesign\examples\Europe\GWA2\input_ts_France_good_wind.csv"
hyd_df = pd.read_csv(hyd_wind_path)
hyd_df = hyd_df.truncate(after=8759)

hyd_2_hopp_wind_map = {'temp_air_1': 'Temperature',
                       'WS_50': 'Speed',
                      'WD_50': 'Direction',
                      'temp_air_1': 'Temperature',
                      'WS_100': 'Speed.1', 
                      'WD_100': 'Direction.1',
                      'temp_air_1': 'Temperature',
                      'WS_150': 'Speed.2', 
                      'WD_150': 'Direction.2',
                      'temp_air_1': 'Temperature',
                      'WS_200': 'Speed.3', 
                      'WD_200': 'Direction.3',
                      } ## The pressures are not in HyDeisgn

hopp_df = hopp_ref_df.copy()
for k, v in hyd_2_hopp_wind_map.items():
    hopp_df[v] = hyd_df[k]
    
hopp_df['Temperature'] -=273.15  # Hydesign have Kelvin
hopp_df.to_csv(hopp_path, sep=',', index=False) ## Note there is a few lines in the top of the file that needs to be added

#%% SOLAR

hopp_ref_solar_path = r"C:\Sandbox\Repo\NREL\HOPP\resource_files\solar\35.2018863_-101.945027_psmv3_60_2012.csv"
hopp_ref_solar_df = pd.read_csv(hopp_ref_solar_path, skiprows=[0,1])

hopp_solar_path = r"C:\Sandbox\Repo\NREL\HOPP\resource_files\solar\france.csv"

hyd_2_hopp_solar_map = {'ghi': 'GHI',
                        'dhi': 'DHI',
                        'dni': 'DNI',
                        'WS_100': 'Wind Speed', 
                        'temp_air_1': 'Temperature', 
                        # 'Solar Zenith Angle',   Why is this not in HyDesign
                        # 'Pressure',
                       # 'Dew Point'
                       }

hopp_solar_df = hopp_ref_solar_df.copy()
for k, v in hyd_2_hopp_solar_map.items():
    hopp_solar_df[v] = hyd_df[k]
    
hopp_solar_df['Temperature'] -= 273.15  # Hydesign have Kelvin
hopp_solar_df.to_csv(hopp_solar_path, sep=',', index=False) ## Note there is a few lines in the top of the file that needs to be added
