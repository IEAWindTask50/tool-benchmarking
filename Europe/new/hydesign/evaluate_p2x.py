import os
import time
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from hydesign.assembly.hpp_assembly_P2X import hpp_model_P2X as hpp_model
from hydesign.examples import examples_filepath

if __name__ == '__main__':
    examples_sites = pd.read_csv(f'{examples_filepath}examples_sites.csv', index_col=0, sep=';')

    name = 'France_good_wind'
    ex_site = examples_sites.loc[examples_sites.name == name]

    longitude = ex_site['longitude'].values[0]
    latitude = ex_site['latitude'].values[0]
    altitude = ex_site['altitude'].values[0]


    input_ts_fn = examples_filepath+ex_site['input_ts_fn'].values[0]

    input_ts = pd.read_csv(input_ts_fn, index_col=0, parse_dates=True)

    required_cols = [col for col in input_ts.columns if 'WD' not in col]
    input_ts = input_ts.loc[:,required_cols]

    sim_pars_fn = examples_filepath+ex_site['sim_pars_fn'].values[0]

    with open(sim_pars_fn) as file:
        sim_pars = yaml.load(file, Loader=yaml.FullLoader)
    H2_demand_fn = examples_filepath+ex_site['H2_demand_col'].values[0]

    H2_demand_ts = pd.read_csv(H2_demand_fn, index_col=0, parse_dates=True)
    

    hpp = hpp_model(
            latitude=latitude,
            longitude=longitude,
            altitude=altitude,
            num_batteries = 3,
            work_dir = './',
            sim_pars_fn = sim_pars_fn,
            input_ts_fn = input_ts_fn,
            H2_demand_fn = H2_demand_fn, 
    )

    start = time.time()

    rotor_diameter_m = 220
    hub_height_m = 130
    wt_rated_power_MW = 10
    surface_tilt_deg = 50
    surface_azimuth_deg = 230
    DC_AC_ratio = 1.5

    Nwt = 30
    wind_MW_per_km2 = 7
    solar_MW = 150
    b_P = 20
    b_E_h  = 4
    cost_of_batt_degr = 8
    clearance = hub_height_m - rotor_diameter_m / 2
    sp = 4 * wt_rated_power_MW * 10 ** 6 / np.pi / rotor_diameter_m ** 2

    ptg_MW = 130
    HSS_kg = 0

    x = [# Wind plant design
        clearance, sp, wt_rated_power_MW, Nwt, wind_MW_per_km2,
        # PV plant design
        solar_MW,  surface_tilt_deg, surface_azimuth_deg, DC_AC_ratio,
        # Energy storage & EMS price constrains
        b_P, b_E_h, cost_of_batt_degr,
        #P2X
        ptg_MW, HSS_kg ]

    x = [35.0, 350.0, 10.0, 39.0, 7.0, 250.0, 50.0, 230.0, 1.8060101395322055, 389.0, 4.0, 18.594850115004615]

    outs = hpp.evaluate(*x)

    hpp.print_design()

    end = time.time()
    print(f'exec. time [min]:', (end - start)/60 )    
    

    import os

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(BASE_DIR, 'evaluation_p2x')
    
    hpp.evaluation_in_csv(data_path)
