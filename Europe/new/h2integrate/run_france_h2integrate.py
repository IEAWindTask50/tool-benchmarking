from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from h2integrate.core.h2integrate_model import H2IntegrateModel

try:
    from prepare_shared_weather import prepare_openmeteo_like_resource_files
except ModuleNotFoundError:
    from benchmark.h2integrate.prepare_shared_weather import prepare_openmeteo_like_resource_files


def _safe_get(model: H2IntegrateModel, var_name: str, units: str | None = None):
    try:
        if units is None:
            return model.prob.get_val(var_name)
        return model.prob.get_val(var_name, units=units)
    except Exception:
        return None


def _sum_timeseries_hours(values: np.ndarray | None) -> float | None:
    if values is None:
        return None
    arr = np.asarray(values).reshape(-1)
    return float(arr.sum())


def _compute_discharge_from_soc(soc_ts: np.ndarray, battery_capacity_kw: float | None = None) -> np.ndarray:
    """Compute discharge power from SOC timeseries.
    
    Discharge (kW) ≈ -capacity * dSOC/dt = -capacity * (SOC[t+1] - SOC[t]) / 1hour
    Positive discharge = battery discharging (supply to grid)
    """
    soc = np.asarray(soc_ts).reshape(-1)
    
    # Compute SOC differences (hourly)
    soc_diff = np.diff(soc, prepend=soc[0])  # Assume first hour has zero change
    
    # If battery capacity is provided, scale by it; otherwise assume SOC is already normalized 0-1
    if battery_capacity_kw is not None:
        # discharge_kw = -soc_diff * battery_capacity (negative diff = discharge)
        discharge_kw = -soc_diff * battery_capacity_kw
    else:
        # Assume soc_diff is already in power units or we return normalized
        discharge_kw = -soc_diff
    
    return discharge_kw


def _first_available(model: H2IntegrateModel, candidates: list[tuple[str, str | None]]) -> float | None:
    for var_name, units in candidates:
        val = _safe_get(model, var_name, units=units)
        if val is None:
            continue
        try:
            return float(np.asarray(val).reshape(-1)[0])
        except Exception:
            continue
    return None


def _first_available_timeseries(model: H2IntegrateModel, candidates: list[tuple[str, str | None]]) -> np.ndarray | None:
    for var_name, units in candidates:
        values = _safe_get(model, var_name, units=units)
        if values is None:
            continue
        return np.asarray(values).reshape(-1)
    return None


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    prepare_openmeteo_like_resource_files(base_dir)
    top_level_config = base_dir / "france_hybrid.yaml"

    model = H2IntegrateModel(top_level_config)
    model.run()

    wind_ts_kw = _safe_get(model, "wind.electricity_out", units="kW")
    solar_ts_kw = _safe_get(model, "solar.electricity_out", units="kW")
    batt_net_ts_kw = _first_available_timeseries(
        model,
        [
            ("battery.electricity_set_point", "kW"),
            ("plant.battery.DemandOpenLoopStorageController.electricity_set_point", "kW"),
            ("battery.electricity_out", "kW"),
            ("plant.battery.SimpleGenericStorage.electricity_out", "kW"),
        ],
    )
    
    # Get battery SOC and capacity to compute actual discharge (better than rated capacity constant)
    batt_soc_ts = _first_available_timeseries(
        model,
        [
            ("plant.battery.DemandOpenLoopStorageController.electricity_soc", None),
            ("battery.state_of_charge", None),
            ("battery.SOC", None),
        ],
    )
    batt_capacity_kw = _first_available(
        model,
        [
            ("plant.battery.DemandOpenLoopStorageController.rated_electricity_production", "kW"),
            ("plant.battery.SimpleGenericStorage.rated_electricity_production", "kW"),
            ("battery.rated_power", "kW"),
        ],
    )
    
    # Compute actual discharge from SOC changes if available
    batt_discharge_ts_kw: np.ndarray | None = None
    if batt_soc_ts is not None:
        batt_discharge_ts_kw = _compute_discharge_from_soc(batt_soc_ts, batt_capacity_kw)
    else:
        # Fallback to rated power (likely constant) if no SOC available
        batt_discharge_ts_kw = _safe_get(model, "battery.electricity_out", units="kW")
    
    wind_deg24_ts_kw = _safe_get(model, "wind.electricity_out_year24_degraded", units="kW")
    solar_deg24_ts_kw = _safe_get(model, "solar.electricity_out_year24_degraded", units="kW")
    curtail_ts_kw = _first_available_timeseries(
        model,
        [
            ("battery.electricity_unused_commodity", "kW"),
            ("battery.unused_electricity_out", "kW"),
            ("combiner.electricity_unused_commodity", "kW"),
        ],
    )

    summary = {
        "case": "france_h2integrate",
        "site_latitude": 48.744116,
        "site_longitude": -0.864258,
        "wind_energy_gwh": None if wind_ts_kw is None else _sum_timeseries_hours(wind_ts_kw) / 1e6,
        "solar_energy_gwh": None if solar_ts_kw is None else _sum_timeseries_hours(solar_ts_kw) / 1e6,
        "battery_output_gwh": None if batt_discharge_ts_kw is None else _sum_timeseries_hours(batt_discharge_ts_kw) / 1e6,
        "wind_capex_million": None,
        "solar_capex_million": None,
        "battery_capex_million": None,
        "wind_opex_million_per_year": None,
        "solar_opex_million_per_year": None,
        "battery_opex_million_per_year": None,
        "npv_million": None,
        "lcoe_eur_per_mwh": None,
        "irr": None,
        "total_curtailment_gwh": None,
    }

    wind_capex = _safe_get(model, "wind.CapEx", units="USD")
    solar_capex = _safe_get(model, "solar.CapEx", units="USD")
    battery_capex = _safe_get(model, "battery.CapEx", units="USD")
    wind_opex = _safe_get(model, "wind.OpEx", units="USD/yr")
    solar_opex = _safe_get(model, "solar.OpEx", units="USD/yr")
    battery_opex = _safe_get(model, "battery.OpEx", units="USD/year")

    if wind_capex is not None:
        summary["wind_capex_million"] = float(np.asarray(wind_capex).reshape(-1)[0]) / 1e6
    if solar_capex is not None:
        summary["solar_capex_million"] = float(np.asarray(solar_capex).reshape(-1)[0]) / 1e6
    if battery_capex is not None:
        summary["battery_capex_million"] = float(np.asarray(battery_capex).reshape(-1)[0]) / 1e6
    if wind_opex is not None:
        summary["wind_opex_million_per_year"] = float(np.asarray(wind_opex).reshape(-1)[0]) / 1e6
    if solar_opex is not None:
        summary["solar_opex_million_per_year"] = float(np.asarray(solar_opex).reshape(-1)[0]) / 1e6
    if battery_opex is not None:
        summary["battery_opex_million_per_year"] = float(np.asarray(battery_opex).reshape(-1)[0]) / 1e6

    npv_val = _first_available(
        model,
        [
            ("plant.finance_subgroup_hybrid.electricity_finance_npv.NPV_electricity__npv", "USD"),
            ("finance_subgroup_hybrid.NPV", "USD"),
            ("finance_subgroup_hybrid.npv", "USD"),
            ("finance_subgroup_renewables.NPV", "USD"),
            ("finance_subgroup_electricity.NPV", "USD"),
            ("plant.finance_subgroup_hybrid.electricity_finance_npv.NPV_electricity", "USD"),
            ("plant.finance_subgroup_renewables.electricity_finance_npv.NPV_electricity", "USD"),
            ("plant.finance_subgroup_battery.electricity_finance_npv.NPV_electricity", "USD"),
        ],
    )
    if npv_val is not None:
        summary["npv_million"] = npv_val / 1e6

    lcoe_val = _first_available(
        model,
        [
            ("plant.finance_subgroup_hybrid.electricity_finance_lco.LCOE_lco", None),
            ("plant.finance_subgroup_hybrid.electricity_finance_lco.LCOE_electricity", None),
            ("finance_subgroup_hybrid.LCOE", "USD/(MW*h)"),
            ("finance_subgroup_renewables.LCOE", "USD/(MW*h)"),
            ("finance_subgroup_electricity.LCOE", "USD/(MW*h)"),
            ("plant.finance_subgroup_hybrid.electricity_finance_lco.LCOE_electricity", "USD/(MW*h)"),
            ("plant.finance_subgroup_renewables.electricity_finance_lco.LCOE_electricity", "USD/(MW*h)"),
        ],
    )
    if lcoe_val is not None:
        # ProFastLCO electricity output is typically USD/kWh. Convert to EUR/MWh-equivalent.
        summary["lcoe_eur_per_mwh"] = lcoe_val * 1000.0 if lcoe_val < 10 else lcoe_val

    irr_val = _first_available(
        model,
        [
            ("plant.finance_subgroup_hybrid.electricity_finance_lco.irr_electricity__lco", None),
            ("plant.finance_subgroup_hybrid.electricity_finance_lco.irr_electricity_lco", None),
            ("plant.finance_subgroup_hybrid.electricity_finance_lco.irr_electricity", None),
            ("finance_subgroup_hybrid.IRR", None),
            ("finance_subgroup_renewables.IRR", None),
            ("finance_subgroup_electricity.IRR", None),
            ("plant.finance_subgroup_hybrid.electricity_finance_npv.IRR_electricity", None),
            ("plant.finance_subgroup_renewables.electricity_finance_npv.IRR_electricity", None),
        ],
    )
    if irr_val is not None:
        summary["irr"] = irr_val

    if curtail_ts_kw is not None:
        summary["total_curtailment_gwh"] = _sum_timeseries_hours(curtail_ts_kw) / 1e6

    summary_df = pd.DataFrame([summary])
    summary_path = base_dir / "france_h2integrate_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    ts_df = pd.DataFrame(
        {
            "hour": np.arange(len(np.asarray(wind_ts_kw).reshape(-1)), dtype=int) if wind_ts_kw is not None else None,
            "wind_electricity_kW": None if wind_ts_kw is None else np.asarray(wind_ts_kw).reshape(-1),
            "wind_electricity_year24_degraded_kW": None if wind_deg24_ts_kw is None else np.asarray(wind_deg24_ts_kw).reshape(-1),
            "solar_electricity_kW": None if solar_ts_kw is None else np.asarray(solar_ts_kw).reshape(-1),
            "solar_electricity_year24_degraded_kW": None if solar_deg24_ts_kw is None else np.asarray(solar_deg24_ts_kw).reshape(-1),
            "battery_net_power_kW": None if batt_net_ts_kw is None else np.asarray(batt_net_ts_kw).reshape(-1),
            "battery_electricity_out_kW": None if batt_discharge_ts_kw is None else np.asarray(batt_discharge_ts_kw).reshape(-1),
            "battery_soc_fraction": None if batt_soc_ts is None else np.asarray(batt_soc_ts).reshape(-1),
            "curtailment_kW": None if curtail_ts_kw is None else curtail_ts_kw,
        }
    )
    ts_path = base_dir / "france_h2integrate_timeseries.csv"
    ts_df.to_csv(ts_path, index=False)

    print("Saved:", summary_path)
    print("Saved:", ts_path)


if __name__ == "__main__":
    main()
