from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _detect_best_lag_hours(hydesign: np.ndarray, h2integrate: np.ndarray, max_lag_hours: int = 72) -> tuple[int, float, float]:
    best: tuple[float, int, float, float] | None = None
    for lag in range(-max_lag_hours, max_lag_hours + 1):
        if lag >= 0:
            a = hydesign[lag:]
            b = h2integrate[: len(h2integrate) - lag]
        else:
            a = hydesign[: len(hydesign) + lag]
            b = h2integrate[-lag:]

        if len(a) < 200:
            continue

        rmse = float(np.sqrt(np.mean((a - b) ** 2)))
        corr = np.corrcoef(a, b)[0, 1] if (np.std(a) > 1e-9 and np.std(b) > 1e-9) else np.nan
        score = (corr if np.isfinite(corr) else -999.0) - 1e-4 * rmse

        if best is None or score > best[0]:
            best = (score, lag, float(corr) if np.isfinite(corr) else float("nan"), rmse)

    if best is None:
        return 0, float("nan"), float("nan")
    _, lag_hours, corr, rmse = best
    return lag_hours, corr, rmse


def _apply_lag_alignment(merged: pd.DataFrame, lag_hours: int) -> pd.DataFrame:
    if lag_hours == 0:
        return merged.copy()

    df = merged.copy()
    hydro_cols = [
        "wind_power_mw_hydesign",
        "solar_power_mw_hydesign",
        "battery_power_mw_hydesign",
        "curtailment_power_mw_hydesign",
    ]
    for col in hydro_cols:
        # Positive lag means HyDesign appears later; shift it earlier in time.
        df[col] = df[col].shift(-lag_hours)

    return df.dropna().reset_index(drop=True)


def _load_hydesign_timeseries(workspace_root: Path) -> pd.DataFrame:
    src = workspace_root / "benchmark" / "hydesign" / "france_hydesign_timeseries.csv"
    if not src.exists():
        raise FileNotFoundError(f"Could not find HyDesign time-series CSV at {src}")

    df = pd.read_csv(src)
    required_cols = [
        "hour",
        "wind_power_undegraded_mw",
        "wind_power_year24_degraded_mw",
        "solar_power_undegraded_mw",
        "solar_power_year24_degraded_mw",
        "battery_power_mw",
        "curtailment_power_mw",
    ]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing HyDesign time-series columns: {missing}")

    renamed = df.rename(
        columns={
            "wind_power_undegraded_mw": "wind_power_undegraded_mw",
            "wind_power_year24_degraded_mw": "wind_power_year24_degraded_mw",
            "solar_power_undegraded_mw": "solar_power_undegraded_mw",
            "solar_power_year24_degraded_mw": "solar_power_year24_degraded_mw",
        }
    ).copy()

    return renamed.loc[:, required_cols].copy()


def _load_h2integrate_timeseries(workspace_root: Path) -> pd.DataFrame:
    src = workspace_root / "benchmark" / "h2integrate" / "france_h2integrate_timeseries.csv"
    if not src.exists():
        raise FileNotFoundError(f"Could not find h2integrate time-series CSV at {src}")

    df = pd.read_csv(src)
    required_cols = ["hour", "wind_electricity_kW", "solar_electricity_kW", "curtailment_kW"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing h2integrate time-series columns: {missing}")

    # Prefer h2integrate net battery power (dispatch signal) for comparison.
    # Fallback to exported battery electricity output if net power is unavailable.
    if "battery_net_power_kW" in df.columns:
        battery_discharge_kw = pd.to_numeric(df["battery_net_power_kW"], errors="coerce")
        battery_source = "battery_net_power_kW"
    elif "battery_electricity_out_kW" in df.columns:
        battery_discharge_kw = pd.to_numeric(df["battery_electricity_out_kW"], errors="coerce")
        battery_source = "battery_electricity_out_kW"
    else:
        raise ValueError(
            "Missing h2integrate battery power columns: expected one of "
            "['battery_electricity_out_kW', 'battery_net_power_kW']"
        )

    renamed = df.rename(
        columns={
            "wind_electricity_kW": "wind_power_undegraded_mw",
            "solar_electricity_kW": "solar_power_undegraded_mw",
            "curtailment_kW": "curtailment_power_mw",
        }
    ).copy()

    renamed["battery_power_mw"] = battery_discharge_kw

    if "wind_electricity_year24_degraded_kW" in renamed.columns:
        renamed["wind_power_year24_degraded_mw"] = pd.to_numeric(
            renamed["wind_electricity_year24_degraded_kW"], errors="coerce"
        )
    else:
        renamed["wind_power_year24_degraded_mw"] = np.nan

    if "solar_electricity_year24_degraded_kW" in renamed.columns:
        renamed["solar_power_year24_degraded_mw"] = pd.to_numeric(
            renamed["solar_electricity_year24_degraded_kW"], errors="coerce"
        )
    else:
        renamed["solar_power_year24_degraded_mw"] = np.nan

    renamed["battery_source"] = battery_source

    for col in [
        "wind_power_undegraded_mw",
        "wind_power_year24_degraded_mw",
        "solar_power_undegraded_mw",
        "solar_power_year24_degraded_mw",
        "battery_power_mw",
        "curtailment_power_mw",
    ]:
        renamed[col] = renamed[col].astype(float) / 1000.0

    return renamed.loc[:, [
        "hour",
        "wind_power_undegraded_mw",
        "wind_power_year24_degraded_mw",
        "solar_power_undegraded_mw",
        "solar_power_year24_degraded_mw",
        "battery_power_mw",
        "curtailment_power_mw",
        "battery_source",
    ]]


def _pick_focus_window(df: pd.DataFrame, window_hours: int = 24 * 14) -> tuple[int, int]:
    if len(df) <= window_hours:
        return 0, len(df)

    # Prefer a window where h2integrate battery is visibly dynamic.
    h2_battery = df["battery_power_mw_h2integrate"].astype(float)
    rolling_battery_std = h2_battery.rolling(window=window_hours, min_periods=window_hours).std()
    if np.isfinite(rolling_battery_std.to_numpy(dtype=float)).any():
        best_std = float(np.nanmax(rolling_battery_std.to_numpy(dtype=float)))
        if best_std > 1e-9:
            end_idx = int(rolling_battery_std.idxmax()) + 1
            start_idx = max(0, end_idx - window_hours)
            return start_idx, min(len(df), start_idx + window_hours)

    # Fallback: highest overall system activity.
    activity = (
        df["wind_power_undegraded_mw_hydesign"].abs()
        + df["solar_power_undegraded_mw_hydesign"].abs()
        + df["battery_power_mw_hydesign"].abs()
        + df["curtailment_power_mw_hydesign"].abs()
        + df["wind_power_undegraded_mw_h2integrate"].abs()
        + df["solar_power_undegraded_mw_h2integrate"].abs()
        + df["battery_power_mw_h2integrate"].abs()
        + df["curtailment_power_mw_h2integrate"].abs()
    )
    rolling_activity = activity.rolling(window=window_hours, min_periods=window_hours).sum()
    end_idx = int(rolling_activity.idxmax()) + 1
    start_idx = max(0, end_idx - window_hours)
    return start_idx, min(len(df), start_idx + window_hours)


def _plot_operation_series(merged: pd.DataFrame, out_path: Path, title: str) -> None:
    fig, axes = plt.subplots(4, 1, figsize=(16, 12), sharex=True)

    # Wind: include HyDesign undegraded and year-24 degraded; h2integrate degraded if available.
    ax = axes[0]
    ax.plot(merged["hour"], merged["wind_power_undegraded_mw_hydesign"], label="HyDesign wind undegraded", color=(0.06, 0.38, 0.55), linewidth=1.2)
    ax.plot(merged["hour"], merged["wind_power_year24_degraded_mw_hydesign"], label="HyDesign wind year-24 degraded", color=(0.06, 0.38, 0.55), linewidth=1.0, linestyle="--")
    ax.plot(merged["hour"], merged["wind_power_undegraded_mw_h2integrate"], label="h2integrate wind", color="black", linewidth=0.9, alpha=0.85)
    h2_wind_deg_col = "wind_power_year24_degraded_mw_h2integrate"
    if h2_wind_deg_col in merged.columns and np.isfinite(merged[h2_wind_deg_col].to_numpy(dtype=float)).any():
        ax.plot(merged["hour"], merged[h2_wind_deg_col], label="h2integrate wind year-24 degraded", color="black", linewidth=0.9, linestyle="--", alpha=0.85)
    ax.set_ylabel("Wind\n[MW]")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right")

    # Solar: include HyDesign undegraded and year-24 degraded; h2integrate degraded if available.
    ax = axes[1]
    ax.plot(merged["hour"], merged["solar_power_undegraded_mw_hydesign"], label="HyDesign solar undegraded", color=(0.90, 0.58, 0.16), linewidth=1.2)
    ax.plot(merged["hour"], merged["solar_power_year24_degraded_mw_hydesign"], label="HyDesign solar year-24 degraded", color=(0.90, 0.58, 0.16), linewidth=1.0, linestyle="--")
    ax.plot(merged["hour"], merged["solar_power_undegraded_mw_h2integrate"], label="h2integrate solar", color="black", linewidth=0.9, alpha=0.85)
    h2_solar_deg_col = "solar_power_year24_degraded_mw_h2integrate"
    if h2_solar_deg_col in merged.columns and np.isfinite(merged[h2_solar_deg_col].to_numpy(dtype=float)).any():
        ax.plot(merged["hour"], merged[h2_solar_deg_col], label="h2integrate solar year-24 degraded", color="black", linewidth=0.9, linestyle="--", alpha=0.85)
    ax.set_ylabel("Solar\n[MW]")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right")

    # Battery: HyDesign net battery power vs h2integrate battery power.
    ax = axes[2]
    ax.plot(merged["hour"], merged["battery_power_mw_hydesign"], label="HyDesign battery net power", color=(0.18, 0.52, 0.31), linewidth=1.2)
    ax.plot(merged["hour"], merged["battery_power_mw_h2integrate"], label="h2integrate battery power", color="black", linewidth=0.9, alpha=0.85)
    ax.axhline(0, color="gray", linewidth=0.5, linestyle=":", alpha=0.7)
    ax.set_ylabel("Battery\n[MW]")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right")

    # Curtailment
    ax = axes[3]
    ax.plot(merged["hour"], merged["curtailment_power_mw_hydesign"], label="HyDesign", color=(0.72, 0.25, 0.20), linewidth=1.2)
    ax.plot(merged["hour"], merged["curtailment_power_mw_h2integrate"], label="h2integrate", color="black", linewidth=0.9, alpha=0.85)
    ax.set_ylabel("Curtailment\n[MW]")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right")

    axes[-1].set_xlabel("Simulation Hour")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    comparison_dir = Path(__file__).resolve().parent
    workspace_root = comparison_dir.parent.parent

    hydesign_df = _load_hydesign_timeseries(workspace_root)
    h2_df = _load_h2integrate_timeseries(workspace_root)

    merged = hydesign_df.merge(h2_df, on="hour", suffixes=("_hydesign", "_h2integrate"), how="inner")
    if merged.empty:
        raise ValueError("No overlapping hourly samples found between HyDesign and h2integrate")

    lag_hours, lag_corr, lag_rmse = _detect_best_lag_hours(
        merged["solar_power_undegraded_mw_hydesign"].to_numpy(dtype=float),
        merged["solar_power_undegraded_mw_h2integrate"].to_numpy(dtype=float),
    )

    aligned = _apply_lag_alignment(merged, lag_hours)
    if aligned.empty:
        raise ValueError("Lag alignment removed all rows; check time-series lengths and lag settings")

    merged_path = comparison_dir / "hydesign_vs_h2integrate_operation_timeseries.csv"
    merged.to_csv(merged_path, index=False)
    aligned_path = comparison_dir / "hydesign_vs_h2integrate_operation_timeseries_lag_aligned.csv"
    aligned.to_csv(aligned_path, index=False)

    diagnostics_path = comparison_dir / "operation_timeseries_lag_diagnostics.txt"
    battery_source = "unknown"
    if "battery_source" in h2_df.columns and not h2_df["battery_source"].isna().all():
        battery_source = str(h2_df["battery_source"].iloc[0])
    h2_has_degraded_ts = (
        "wind_power_year24_degraded_mw_h2integrate" in merged.columns
        and np.isfinite(merged["wind_power_year24_degraded_mw_h2integrate"].to_numpy(dtype=float)).any()
    ) or (
        "solar_power_year24_degraded_mw_h2integrate" in merged.columns
        and np.isfinite(merged["solar_power_year24_degraded_mw_h2integrate"].to_numpy(dtype=float)).any()
    )
    diagnostics_path.write_text(
        "\n".join(
            [
                "Lag diagnostics based on solar time series:",
                f"best_lag_hours={lag_hours}",
                f"correlation_at_best_lag={lag_corr}",
                f"rmse_at_best_lag_mw={lag_rmse}",
                f"h2integrate_year24_degradation_timeseries_available={h2_has_degraded_ts}",
                f"h2integrate_battery_series_used={battery_source}",
                f"h2integrate_battery_power_source={battery_source}",
                f"h2integrate_battery_power_min_mw={float(merged['battery_power_mw_h2integrate'].min()):.2f}",
                f"h2integrate_battery_power_max_mw={float(merged['battery_power_mw_h2integrate'].max()):.2f}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    # _plot_operation_series(
    #     merged,
    #     comparison_dir / "hydesign_vs_h2integrate_operation_timeseries_full_year_raw.png",
    #     "HyDesign vs h2integrate: HPP Operation Time Series (Full Year, Raw)",
    # )

    # _plot_operation_series(
    #     aligned,
    #     comparison_dir / "hydesign_vs_h2integrate_operation_timeseries_full_year_lag_aligned_v2.png",
    #     f"HyDesign vs h2integrate: HPP Operation Time Series (Full Year, Lag-Aligned: {lag_hours:+d}h)",
    # )
    # Keep legacy filename in sync with the corrected aligned output.
    # _plot_operation_series(
    #     aligned,
    #     comparison_dir / "hydesign_vs_h2integrate_operation_timeseries_full_year.png",
    #     f"HyDesign vs h2integrate: HPP Operation Time Series (Full Year, Lag-Aligned: {lag_hours:+d}h)",
    # )

    start_idx, end_idx = _pick_focus_window(aligned)
    focus = aligned.iloc[start_idx:end_idx].copy()
    # _plot_operation_series(
    #     focus,
    #     comparison_dir / "hydesign_vs_h2integrate_operation_timeseries_focus_window_lag_aligned_v2.png",
    #     f"HyDesign vs h2integrate: HPP Operation Time Series (Focused 14-Day Window, Lag-Aligned: {lag_hours:+d}h)",
    # )
    # # Keep legacy filename in sync with the corrected aligned output.
    _plot_operation_series(
        focus,
        comparison_dir / "hydesign_vs_h2integrate_operation_timeseries_focus_window.png",
        f"HyDesign vs h2integrate: HPP Operation Time Series (Focused 14-Day Window, Lag-Aligned: {lag_hours:+d}h)",
    )

    # print("Saved:", merged_path)
    # print("Saved:", aligned_path)
    # print("Saved:", diagnostics_path)
    # print("Saved:", comparison_dir / "hydesign_vs_h2integrate_operation_timeseries_full_year_raw.png")
    # print("Saved:", comparison_dir / "hydesign_vs_h2integrate_operation_timeseries_full_year.png")
    # print("Saved:", comparison_dir / "hydesign_vs_h2integrate_operation_timeseries_full_year_lag_aligned_v2.png")
    # print("Saved:", comparison_dir / "hydesign_vs_h2integrate_operation_timeseries_focus_window.png")
    # print("Saved:", comparison_dir / "hydesign_vs_h2integrate_operation_timeseries_focus_window_lag_aligned_v2.png")


if __name__ == "__main__":
    main()