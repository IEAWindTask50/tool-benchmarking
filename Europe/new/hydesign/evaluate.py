import os
import time
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from hydesign.assembly.hpp_assembly import hpp_model
from hydesign.examples import examples_filepath


def _extract_hourly_series(prob, var_name, n_hours):
    values = np.asarray(prob.get_val(var_name)).reshape(-1)
    if values.size == n_hours:
        return values
    if values.size > n_hours:
        return values[:n_hours]
    raise ValueError(f"Variable {var_name} has {values.size} points, expected at least {n_hours}")


def _extract_degraded_year(prob, var_name, n_hours, year_index):
    values = np.asarray(prob.get_val(var_name)).reshape(-1)
    total_years = values.size // n_hours
    if total_years <= 0:
        raise ValueError(f"Variable {var_name} has incompatible length {values.size} for n_hours={n_hours}")

    # If requested year is unavailable, use the last simulated year.
    idx = min(year_index, total_years - 1)
    start = idx * n_hours
    end = start + n_hours
    return values[start:end], idx

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

    # Benchmark parity option: remove shared CAPEX terms so total CAPEX aligns to
    # wind+solar+battery component costs for cross-tool comparison.
    sim_pars['hpp_BOS_soft_cost'] = 0.0
    sim_pars['hpp_grid_connection_cost'] = 0.0
    sim_pars['land_cost'] = 0.0

    local_sim_pars_fn = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'hpp_pars_shared0.yml')
    with open(local_sim_pars_fn, 'w', encoding='utf-8') as file:
        yaml.safe_dump(sim_pars, file, sort_keys=False)

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
                sim_pars_fn = local_sim_pars_fn,
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

    x = [# Wind plant design
        clearance, sp, wt_rated_power_MW, Nwt, wind_MW_per_km2,
        # PV plant design
        solar_MW,  surface_tilt_deg, surface_azimuth_deg, DC_AC_ratio,
        # Energy storage & EMS price constrains
        b_P, b_E_h, cost_of_batt_degr]
    # x=[35.0, 350.0, 10.0, 17.0, 7.0, 
    #    227.0, 9.907972374352239, 135.42752579370625, 1.5, 
    #    10.0, 4.0, 17.38679959753782]
    outs = hpp.evaluate(*x)

    hpp.print_design(x, outs)

    end = time.time()
    print(f'exec. time [min]:', (end - start)/60 )

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(BASE_DIR, 'evaluation')

    hpp.evaluation_in_csv(data_path)

    prob = hpp.prob
    wind_ts_mw = np.asarray(prob.get_val('wind_t')).reshape(-1)
    solar_ts_mw = np.asarray(prob.get_val('solar_t')).reshape(-1)
    n_hours = wind_ts_mw.size
    wind_deg24_ts_mw, wind_deg_year_used = _extract_degraded_year(prob, 'wpp_with_degradation.wind_t_ext_deg', n_hours, 24)
    solar_deg24_ts_mw, solar_deg_year_used = _extract_degraded_year(prob, 'pvp_with_degradation.solar_t_ext_deg', n_hours, 24)

    ts_df = pd.DataFrame(
        {
            'hour': np.arange(n_hours, dtype=int),
            'wind_power_undegraded_mw': wind_ts_mw,
            'wind_power_year24_degraded_mw': wind_deg24_ts_mw,
            'solar_power_undegraded_mw': solar_ts_mw,
            'solar_power_year24_degraded_mw': solar_deg24_ts_mw,
            'battery_power_mw': _extract_hourly_series(prob, 'ems.b_t', n_hours),
            'curtailment_power_mw': _extract_hourly_series(prob, 'ems.hpp_curt_t', n_hours),
        }
    )
    ts_path = os.path.join(BASE_DIR, 'france_hydesign_timeseries.csv')
    ts_df.to_csv(ts_path, index=False)
    print(f'Saved: {ts_path}')
    print(f'Wind degradation year used: {wind_deg_year_used}')
    print(f'Solar degradation year used: {solar_deg_year_used}')
