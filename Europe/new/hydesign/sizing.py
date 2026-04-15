import os
import time
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from hydesign.assembly.hpp_assembly import hpp_model
from hydesign.examples import examples_filepath

from hydesign.assembly.hpp_assembly import hpp_model
from hydesign.Parallel_EGO import EfficientGlobalOptimizationDriver
from hydesign.examples import examples_filepath
import pandas as pd
import os

if __name__ == '__main__':


    n_procs = os.cpu_count()
    if n_procs > 2:
        n_procs -= 1
        n_doe = int(3 * n_procs)
    else:
        n_procs -= 0
        n_doe = int(8 * n_procs)
    print(n_doe, n_procs)

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
    inputs = {
        'name': ex_site['name'],
        'longitude': longitude,
        'latitude': latitude,
        'altitude': altitude,
        'input_ts_fn': input_ts_fn,
        'sim_pars_fn': sim_pars_fn,
        'H2_demand_fn': H2_demand_fn,

        'opt_var': "NPV_over_CAPEX",
        'num_batteries': 10,
        'n_procs': n_procs,
        'n_doe': n_doe,
        'n_clusters': n_procs,
        'n_seed': 0,
        'max_iter': 10,
        'final_design_fn': 'hydesign_design_0.csv',
        'npred': 3e4,
        'tol': 1e-6,
        'min_conv_iter': 3,
        'work_dir': './',
        'hpp_model': hpp_model,
    'variables': {
        'clearance [m]':
            # {'var_type':'design',
            # 'limits':[10, 60],
            # 'types':'int'
            # },
            {'var_type':'fixed',
              'value': 35
              },
        'sp [W/m2]':
            # {'var_type':'design',
            # 'limits':[200, 360],
            # 'types':'int'
            # },
            {'var_type':'fixed',
              'value': 350
              },

        'p_rated [MW]':
            # {'var_type':'design',
            # 'limits':[1, 10],
            # 'types':'int'
            # },
            {'var_type':'fixed',
             'value': 10
            },
        'Nwt':
            # {'var_type':'design',
            # 'limits':[0, 50],
            # 'types':'int'
            # },
            {'var_type':'fixed',
              'value': 31
              },
        'wind_MW_per_km2 [MW/km2]':
            # {'var_type':'design',
            # 'limits':[5, 9],
            # 'types':'float'
            # },
            {'var_type':'fixed',
              'value': 7
              },
        'solar_MW [MW]':
            # {'var_type':'design',
            #   'limits':[0, 500],
            #   'types':'int'
            #   },
            {'var_type':'fixed',
            'value': 166
            },
        'surface_tilt [deg]':
            {'var_type':'design',
              'limits':[-90, 180],
              'types':'float'
              },
            # {'var_type':'fixed',
            # 'value': 25
            # },
        'surface_azimuth [deg]':
            {'var_type':'design',
              'limits':[0, 360],
              'types':'float'
              },
            # {'var_type':'fixed',
            # 'value': 180
            # },
        'DC_AC_ratio':
            # {'var_type':'design',
            #   'limits':[1, 2.0],
            #   'types':'float'
            #   },
            {'var_type':'fixed',
            'value':1.5,
            },
        'b_P [MW]':
            # {'var_type':'design',
            #   'limits':[0, 100],
            #   'types':'int'
            #   },
            {'var_type':'fixed',
            'value': 50
            },
        'b_E_h [h]':
            # {'var_type':'design',
            #   'limits':[1, 10],
            #   'types':'int'
            #   },
            {'var_type':'fixed',
            'value': 4
            },
        'cost_of_battery_P_fluct_in_peak_price_ratio':
            # {'var_type':'design',
            #   'limits':[0, 20],
            #   'types':'float'
            #   },
            {'var_type':'fixed',
            'value': 8},
        }}
    EGOD = EfficientGlobalOptimizationDriver(**inputs)
    EGOD.run()
