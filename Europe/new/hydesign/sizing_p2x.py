def _set_numeric_thread_env(default_threads='1'):
    import os

    for env_name in [
        'OMP_NUM_THREADS',
        'OPENBLAS_NUM_THREADS',
        'MKL_NUM_THREADS',
        'NUMEXPR_NUM_THREADS',
        'VECLIB_MAXIMUM_THREADS',
        'BLIS_NUM_THREADS',
    ]:
        os.environ.setdefault(env_name, default_threads)


def _get_env_int(name, default):
    import os

    value = os.getenv(name)
    return default if value in (None, '') else int(value)


def _get_env_float(name, default):
    import os

    value = os.getenv(name)
    return default if value in (None, '') else float(value)


def _enable_serial_parallel_ego(parallel_ego_module, np):
    if getattr(parallel_ego_module.ParallelEvaluator, '_serial_mode_patched', False):
        return

    run_ydoe_orig = parallel_ego_module.ParallelEvaluator.run_ydoe
    run_both_orig = parallel_ego_module.ParallelEvaluator.run_both
    run_xopt_iter_orig = parallel_ego_module.ParallelEvaluator.run_xopt_iter

    def run_ydoe(self, fun, x, **kwargs):
        if self.n_procs == 1:
            return np.array([fun((x[[i], :], kwargs)) for i in range(x.shape[0])]).reshape(-1, 1)
        return run_ydoe_orig(self, fun, x, **kwargs)

    def run_both(self, fun, i, **kwargs):
        if self.n_procs == 1:
            return [fun(((i * 100) + kwargs['n_seed'], kwargs))]
        return run_both_orig(self, fun, i, **kwargs)

    def run_xopt_iter(self, fun, x, **kwargs):
        if self.n_procs == 1:
            return np.vstack([fun((x[[ii], :], kwargs)) for ii in range(x.shape[0])])
        return run_xopt_iter_orig(self, fun, x, **kwargs)

    parallel_ego_module.ParallelEvaluator.run_ydoe = run_ydoe
    parallel_ego_module.ParallelEvaluator.run_both = run_both
    parallel_ego_module.ParallelEvaluator.run_xopt_iter = run_xopt_iter
    parallel_ego_module.ParallelEvaluator._serial_mode_patched = True


if __name__ == '__main__':
    _set_numeric_thread_env()

    import os
    import time
    import yaml
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    from hydesign.assembly.hpp_assembly_P2X import hpp_model_P2X as hpp_model
    from hydesign.examples import examples_filepath

    import hydesign.Parallel_EGO as parallel_ego_module
    from hydesign.Parallel_EGO import EfficientGlobalOptimizationDriver
    from hydesign.examples import examples_filepath
    import pandas as pd
    import os
    cpu_count = os.cpu_count() or 1
    if os.name == 'nt':
        default_n_procs = 1
    else:
        default_n_procs = cpu_count - 1 if cpu_count > 2 else cpu_count
    n_procs = _get_env_int('HYDESIGN_N_PROCS', max(1, default_n_procs))
    default_n_doe = int(3 * n_procs) if n_procs > 1 else 8
    n_doe = _get_env_int('HYDESIGN_N_DOE', default_n_doe)
    n_clusters = _get_env_int('HYDESIGN_N_CLUSTERS', n_procs)
    max_iter = _get_env_int('HYDESIGN_MAX_ITER', 20)
    npred = _get_env_float('HYDESIGN_NPRED', 3e4)

    if n_procs == 1:
        _enable_serial_parallel_ego(parallel_ego_module, np)

    print(
        (
            'Launching sizing_p2x with '
            f'cpu_count={cpu_count}, n_procs={n_procs}, n_doe={n_doe}, '
            f'n_clusters={n_clusters}, max_iter={max_iter}, npred={int(npred)}. '
            'The first progress checkpoint is printed only after the initial DOE finishes.'
        ),
        flush=True,
    )
    if os.name == 'nt' and n_procs == 1:
        print(
            'Windows default is serial evaluation for P2X because concurrent worker processes '
            'have shown intermittent failures with valid designs. Set HYDESIGN_N_PROCS to opt back into parallel mode.',
            flush=True,
        )

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
        'name': ex_site['name'].values[0],
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
        'n_clusters': n_clusters,
        'n_seed': 0,
        'max_iter': max_iter,
        'final_design_fn': 'hydesign_design_0_p2x.csv',
        'npred': npred,
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
            {'var_type':'design',
            'limits':[0, 50],
            'types':'int'
            },
            # {'var_type':'fixed',
            #   'value': 30
            #   },
        'wind_MW_per_km2 [MW/km2]':
            # {'var_type':'design',
            # 'limits':[5, 9],
            # 'types':'float'
            # },
            {'var_type':'fixed',
              'value': 7
              },
        'solar_MW [MW]':
            {'var_type':'design',
              'limits':[0, 500],
              'types':'int'
              },
            # {'var_type':'fixed',
            # 'value': 150
            # },
        'surface_tilt [deg]':
            # {'var_type':'design',
            #   'limits':[-90, 180],
            #   'types':'float'
            #   },
            {'var_type':'fixed',
            'value': 50.0,
            },
        'surface_azimuth [deg]':
            # {'var_type':'design',
            #   'limits':[0, 360],
            #   'types':'float'
            #   },
            {'var_type':'fixed',
            'value': 230.0,
            },
        'DC_AC_ratio':
            {'var_type':'design',
              'limits':[1, 2.0],
              'types':'float'
              },
            # {'var_type':'fixed',
            # 'value':1.5,
            # },
        'b_P [MW]':
            {'var_type':'design',
              'limits':[0, 500],
              'types':'int'
              },
            # {'var_type':'fixed',
            # 'value': 50
            # },
        'b_E_h [h]':
            # {'var_type':'design',
            #   'limits':[1, 10],
            #   'types':'int'
            #   },
            {'var_type':'fixed',
            'value': 4
            },
        'cost_of_battery_P_fluct_in_peak_price_ratio':
            {'var_type':'design',
              'limits':[0, 20],
              'types':'float'
              },
            # {'var_type':'fixed',
            # 'value': 8},
        'ptg_MW [MW]':
            {'var_type':'design',
            'limits':[1, 500],
            'types':'int'
            },        
            }}
    print('Creating EfficientGlobalOptimizationDriver...', flush=True)
    EGOD = EfficientGlobalOptimizationDriver(**inputs)
    print('Starting optimization run...', flush=True)
    EGOD.run()
