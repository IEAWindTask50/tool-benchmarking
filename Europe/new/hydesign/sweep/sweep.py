import os
import time
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from hydesign.assembly.hpp_assembly import hpp_model
from hydesign.examples import examples_filepath
import xarray as xr

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

    rotor_diameter_m = 220
    hub_height_m = 130
    wt_rated_power_MW = 10
    surface_tilt_deg = 50
    surface_azimuth_deg = 230
    DC_AC_ratio = 1.5


    hpp = hpp_model(
            latitude=latitude,
            longitude=longitude,
            altitude=altitude,
            rotor_diameter_m = rotor_diameter_m,
            hub_height_m = hub_height_m,
            wt_rated_power_MW = wt_rated_power_MW,
            surface_tilt_deg = surface_tilt_deg,
            surface_azimuth_deg = surface_azimuth_deg,
            DC_AC_ratio = DC_AC_ratio,
            num_batteries = 5,
            work_dir = './',
            sim_pars_fn = sim_pars_fn,
            input_ts_fn = input_ts_fn,
    )

    start = time.time()

    Nwt = 30
    wind_MW_per_km2 = 7
    solar_MW = 150
    b_P = 20
    b_E_h  = 4
    cost_of_batt_degr = 8
    clearance = hub_height_m - rotor_diameter_m / 2
    sp = 4 * wt_rated_power_MW * 10 ** 6 / np.pi / rotor_diameter_m ** 2

    surface_tilt_degs = np.linspace(0,90, 10)
    surface_azimuth_degs = np.linspace(0,360, 20)
    output_dir = os.path.dirname(__file__)
    output_nc = os.path.join(output_dir, 'sweep_results.nc')
    output_csv = os.path.join(output_dir, 'sweep_results.csv')
    # checkpoint_every = 10

    res = xr.Dataset(
        data_vars={
            'NPV_over_CAPEX': (['surface_tilt_deg', 'surface_azimuth_deg'], np.full((10,20), np.nan)),
            'AEP': (['surface_tilt_deg', 'surface_azimuth_deg'], np.full((10,20), np.nan)),
            'LCOE': (['surface_tilt_deg', 'surface_azimuth_deg'], np.full((10,20), np.nan)),
            'Revenues': (['surface_tilt_deg', 'surface_azimuth_deg'], np.full((10,20), np.nan)),
            'CAPEX': (['surface_tilt_deg', 'surface_azimuth_deg'], np.full((10,20), np.nan))
        },
        coords={
            'surface_tilt_deg': surface_tilt_degs,
            'surface_azimuth_deg': surface_azimuth_degs
        }
    )
    eval_count = 0
    for surface_tilt_deg in surface_tilt_degs:
        for surface_azimuth_deg in surface_azimuth_degs:
            x = [# Wind plant design
                clearance, sp, wt_rated_power_MW, Nwt, wind_MW_per_km2,
                # PV plant design
                solar_MW,  surface_tilt_deg, surface_azimuth_deg, DC_AC_ratio,
                # Energy storage & EMS price constrains
                b_P, b_E_h, cost_of_batt_degr]
            outs = hpp.evaluate(*x)
            if surface_tilt_deg == surface_tilt_degs[0] and surface_azimuth_deg == surface_azimuth_degs[0]:
                try:
                    hpp.print_design(x, outs)
                except TypeError:
                    hpp.print_design()

            outs_by_name = dict(zip(hpp.list_out_vars, np.asarray(outs).ravel()))
            idx = dict(surface_tilt_deg=surface_tilt_deg, surface_azimuth_deg=surface_azimuth_deg)
            res['NPV_over_CAPEX'].loc[idx] = outs_by_name['NPV_over_CAPEX']
            res['AEP'].loc[idx] = outs_by_name['AEP [GWh]']
            res['LCOE'].loc[idx] = outs_by_name['LCOE [Euro/MWh]']
            res['Revenues'].loc[idx] = outs_by_name['Revenues [MEuro]']
            res['CAPEX'].loc[idx] = outs_by_name['CAPEX [MEuro]']

                # eval_count += 1
                # if eval_count % checkpoint_every == 0:
                # res.to_netcdf(output_nc)

    res.to_netcdf(output_nc)
    res.to_dataframe().reset_index().to_csv(output_csv, index=False)

    print(res)
    print(f'Saved NetCDF results to: {output_nc}')
    print(f'Saved CSV results to: {output_csv}')
    end = time.time()
    # x = [# Wind plant design
    #     clearance, sp, wt_rated_power_MW, Nwt, wind_MW_per_km2,
    #     # PV plant design
    #     solar_MW,  surface_tilt_deg, surface_azimuth_deg, DC_AC_ratio,
    #     # Energy storage & EMS price constrains
    #     b_P, b_E_h, cost_of_batt_degr]
    # # x=[35.0, 350.0, 10.0, 17.0, 7.0, 
    # #    227.0, 9.907972374352239, 135.42752579370625, 1.5, 
    # #    10.0, 4.0, 17.38679959753782]
    # outs = hpp.evaluate(*x)

    # hpp.print_design(x, outs)

    # end = time.time()
    print(f'exec. time [min]:', (end - start)/60 )
    # hpp.evaluation_in_csv('evaluation.csv')