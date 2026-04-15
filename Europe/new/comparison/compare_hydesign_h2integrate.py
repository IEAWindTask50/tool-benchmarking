from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _format_sig3(value: float) -> str:
    if pd.isna(value):
        return "--"
    return f"{float(value):.3g}"


def _write_latex_absolute_metrics_table(workspace_root: Path, comparison_df: pd.DataFrame) -> Path:
    metric_labels = {
        "total_generation_gwh": "Total generation",
        "wind_capex_million": "Wind CAPEX",
        "solar_capex_million": "Solar CAPEX",
        "battery_capex_million": "Battery CAPEX",
        "total_capex_million": "Total CAPEX",
        "wind_opex_million_per_year": "Wind OPEX",
        "solar_opex_million_per_year": "Solar OPEX",
        "total_opex_million_per_year": "Total OPEX",
        "npv_million": "NPV",
        "lcoe_eur_per_mwh": "LCOE",
        "irr": "IRR",
        "total_curtailment_gwh": "Total curtailment",
    }
    metric_units = {
        "total_generation_gwh": "GWh",
        "wind_capex_million": "MEUR",
        "solar_capex_million": "MEUR",
        "battery_capex_million": "MEUR",
        "total_capex_million": "MEUR",
        "wind_opex_million_per_year": "MEUR/yr",
        "solar_opex_million_per_year": "MEUR/yr",
        "total_opex_million_per_year": "MEUR/yr",
        "npv_million": "MEUR",
        "lcoe_eur_per_mwh": "EUR/MWh",
        "irr": "-",
        "total_curtailment_gwh": "GWh",
    }

    lines = [
        "% Absolute (raw) values, not normalized. Values shown with 3 significant digits.",
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{HyDesign vs h2integrate: absolute metric comparison}",
        "\\label{tab:hydesign_h2integrate_absolute_metrics}",
        "\\begin{tabular}{l l r r}",
        "\\hline",
        r"Metric & Unit & HyDesign & h2integrate \\",
        "\\hline",
    ]

    for row in comparison_df.itertuples(index=False):
        metric_key = str(row.metric)
        label = metric_labels.get(metric_key, metric_key)
        unit = metric_units.get(metric_key, "-")
        hydesign_val = _format_sig3(float(row.hydesign))
        h2_val = _format_sig3(float(row.h2integrate))
        lines.append(f"{label} & {unit} & {hydesign_val} & {h2_val} \\\\")

    lines.extend([
        "\\hline",
        "\\end{tabular}",
        "\\end{table}",
        "",
    ])

    paper_dir = workspace_root / "paper"
    paper_dir.mkdir(parents=True, exist_ok=True)
    out_path = paper_dir / "hydesign_vs_h2integrate_absolute_metrics_table.tex"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def _read_hydesign_kv_csv(csv_path: Path) -> dict[str, float | str]:
    df = pd.read_csv(csv_path, header=None)
    if df.shape[1] < 2:
        raise ValueError(f"Unexpected HyDesign CSV format in {csv_path}")

    values: dict[str, float | str] = {}
    for _, row in df.iterrows():
        key = str(row.iloc[0]).strip()
        val = row.iloc[1]
        if key == "" or key.lower() == "nan":
            continue
        try:
            values[key] = float(val)
        except (TypeError, ValueError):
            values[key] = str(val)
    return values


def _read_hydesign_table_csv(csv_path: Path) -> dict[str, float | str]:
    """Read HyDesign output where metrics are columns and one row holds values."""
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"HyDesign table CSV is empty: {csv_path}")

    row = df.iloc[0]
    values: dict[str, float | str] = {}
    for col in df.columns:
        key = str(col).strip()
        if key == "" or key.lower() == "unnamed: 0":
            continue
        val = row[col]
        try:
            values[key] = float(val)
        except (TypeError, ValueError):
            values[key] = str(val)
    return values


def _find_hydesign_source(workspace_root: Path) -> Path:
    benchmark_root = workspace_root / "benchmark"
    candidates = [
        benchmark_root / "hydesign" / "evaluation.csv",
        benchmark_root / "hydesign" / "evaluation.csv.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Could not find a HyDesign evaluation CSV inside benchmark/hydesign. "
        "Expected one of: benchmark/hydesign/evaluation.csv.csv or benchmark/hydesign/evaluation.csv"
    )


def _extract_hydesign_metrics(workspace_root: Path) -> tuple[dict[str, float | str], Path]:
    src = _find_hydesign_source(workspace_root)
    # New benchmark outputs are table-like; older outputs are key-value style.
    # Try table parser first, then fall back to key-value parser.
    try:
        kv = _read_hydesign_table_csv(src)
    except Exception:
        kv = _read_hydesign_kv_csv(src)

    metrics = {
        "total_generation_gwh": kv.get("AEP [GWh]", kv.get("Mean Annual Electricity Sold [GWh]", np.nan)),
        "wind_capex_million": kv.get("Wind CAPEX [MEuro]", np.nan),
        "solar_capex_million": kv.get("PV CAPEX [MEuro]", np.nan),
        "battery_capex_million": kv.get("Batt CAPEX [MEuro]", np.nan),
        "total_capex_million": kv.get("CAPEX [MEuro]", np.nan),
        "wind_opex_million_per_year": kv.get("Wind OPEX [MEuro]", np.nan),
        "solar_opex_million_per_year": kv.get("PV OPEX [MEuro]", np.nan),
        # "battery_opex_million_per_year": kv.get("Batt OPEX [MEuro]", np.nan),
        "total_opex_million_per_year": kv.get("OPEX [MEuro]", np.nan),
        "npv_million": kv.get("NPV [MEuro]", np.nan),
        "lcoe_eur_per_mwh": kv.get("LCOE [Euro/MWh]", np.nan),
        "irr": kv.get("IRR", np.nan),
        "total_curtailment_gwh": kv.get("Total curtailment [GWh]", np.nan),
    }

    if pd.isna(metrics["total_capex_million"]):
        metrics["total_capex_million"] = np.nansum(
            [
                metrics["wind_capex_million"],
                metrics["solar_capex_million"],
                metrics["battery_capex_million"],
            ]
        )

    if pd.isna(metrics["total_opex_million_per_year"]):
        metrics["total_opex_million_per_year"] = np.nansum(
            [
                metrics["wind_opex_million_per_year"],
                metrics["solar_opex_million_per_year"],
                # metrics["battery_opex_million_per_year"],
            ]
        )

    return metrics, src


def _extract_h2integrate_metrics(workspace_root: Path) -> tuple[dict[str, float | str], Path]:
    src = workspace_root / "benchmark" / "h2integrate" / "france_h2integrate_summary.csv"
    if not src.exists():
        raise FileNotFoundError(f"Could not find h2integrate summary CSV at {src}")

    df = pd.read_csv(src)
    if df.empty:
        raise ValueError(f"h2integrate summary CSV is empty: {src}")

    row = df.iloc[0]
    wind_gen = float(row.get("wind_energy_gwh", np.nan))
    solar_gen = float(row.get("solar_energy_gwh", np.nan))

    metrics = {
        "total_generation_gwh": wind_gen + solar_gen,
        "wind_capex_million": float(row.get("wind_capex_million", np.nan)),
        "solar_capex_million": float(row.get("solar_capex_million", np.nan)),
        "battery_capex_million": float(row.get("battery_capex_million", np.nan)),
        "total_capex_million": np.nan,
        "wind_opex_million_per_year": float(row.get("wind_opex_million_per_year", np.nan)),
        "solar_opex_million_per_year": float(row.get("solar_opex_million_per_year", np.nan)),
        # "battery_opex_million_per_year": float(row.get("battery_opex_million_per_year", np.nan)),
        "total_opex_million_per_year": np.nan,
        "npv_million": float(row.get("npv_million", np.nan)),
        "lcoe_eur_per_mwh": float(row.get("lcoe_eur_per_mwh", np.nan)),
        "irr": float(row.get("irr", np.nan)),
        "total_curtailment_gwh": float(row.get("total_curtailment_gwh", np.nan)),
    }

    metrics["total_capex_million"] = np.nansum(
        [
            metrics["wind_capex_million"],
            metrics["solar_capex_million"],
            metrics["battery_capex_million"],
        ]
    )
    metrics["total_opex_million_per_year"] = np.nansum(
        [
            metrics["wind_opex_million_per_year"],
            metrics["solar_opex_million_per_year"],
            # metrics["battery_opex_million_per_year"],
        ]
    )

    return metrics, src


def _plot_side_by_side(
    out_dir: Path,
    hydesign_metrics: dict[str, float | str],
    h2_metrics: dict[str, float | str],
) -> None:
    metric_keys = [
        "total_generation_gwh",
        "wind_capex_million",
        "solar_capex_million",
        "battery_capex_million",
        "total_capex_million",
        "wind_opex_million_per_year",
        "solar_opex_million_per_year",
        # "battery_opex_million_per_year",
        "total_opex_million_per_year",
        "npv_million",
        "lcoe_eur_per_mwh",
        "irr",
        "total_curtailment_gwh",
    ]
    metric_labels = [
        "Generation",
        "Wind CAPEX",
        "Solar CAPEX",
        "Battery CAPEX",
        "Total CAPEX",
        "Wind OPEX",
        "Solar OPEX",
        # "Battery OPEX",
        "Total OPEX",
        "NPV",
        "LCOE",
        "IRR",
        "Curtailment",
    ]

    hydesign_vals = np.array([float(hydesign_metrics.get(k, np.nan)) for k in metric_keys], dtype=float)
    h2_vals = np.array([float(h2_metrics.get(k, np.nan)) for k in metric_keys], dtype=float)

    means = np.nanmean(np.vstack([hydesign_vals, h2_vals]), axis=0)
    means = np.where(np.isclose(means, 0.0), np.nan, means)

    hydesign_norm = hydesign_vals / means
    h2_norm = h2_vals / means

    fig, ax = plt.subplots(figsize=(17.5, 6.5))
    x = np.arange(len(metric_keys))
    width = 0.38

    ax.bar(x - width / 2, hydesign_norm, width, label="HyDesign")
    ax.bar(x + width / 2, h2_norm, width, label="h2integrate")
    ax.axhline(1.0, color="black", linewidth=1.0, linestyle="--", alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, rotation=35, ha="right")
    ax.set_ylabel("Normalized Value (value / mean of both tools)")
    ax.set_title("HyDesign vs h2integrate: Normalized Metric Comparison")
    ax.legend()
    fig.tight_layout()

    out_png = out_dir / "hydesign_vs_h2integrate_side_by_side.png"
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def main() -> None:
    comparison_dir = Path(__file__).resolve().parent
    workspace_root = comparison_dir.parent.parent

    hydesign_metrics, hydesign_source = _extract_hydesign_metrics(workspace_root)
    h2_metrics, h2_source = _extract_h2integrate_metrics(workspace_root)

    comparison_df = pd.DataFrame(
        {
            "metric": [
                "total_generation_gwh",
                "wind_capex_million",
                "solar_capex_million",
                "battery_capex_million",
                "total_capex_million",
                "wind_opex_million_per_year",
                "solar_opex_million_per_year",
                # "battery_opex_million_per_year",
                "total_opex_million_per_year",
                "npv_million",
                "lcoe_eur_per_mwh",
                "irr",
                "total_curtailment_gwh",
            ],
            "hydesign": [
                hydesign_metrics.get("total_generation_gwh", np.nan),
                hydesign_metrics.get("wind_capex_million", np.nan),
                hydesign_metrics.get("solar_capex_million", np.nan),
                hydesign_metrics.get("battery_capex_million", np.nan),
                hydesign_metrics.get("total_capex_million", np.nan),
                hydesign_metrics.get("wind_opex_million_per_year", np.nan),
                hydesign_metrics.get("solar_opex_million_per_year", np.nan),
                # hydesign_metrics.get("battery_opex_million_per_year", np.nan),
                hydesign_metrics.get("total_opex_million_per_year", np.nan),
                hydesign_metrics.get("npv_million", np.nan),
                hydesign_metrics.get("lcoe_eur_per_mwh", np.nan),
                hydesign_metrics.get("irr", np.nan),
                hydesign_metrics.get("total_curtailment_gwh", np.nan),
            ],
            "h2integrate": [
                h2_metrics.get("total_generation_gwh", np.nan),
                h2_metrics.get("wind_capex_million", np.nan),
                h2_metrics.get("solar_capex_million", np.nan),
                h2_metrics.get("battery_capex_million", np.nan),
                h2_metrics.get("total_capex_million", np.nan),
                h2_metrics.get("wind_opex_million_per_year", np.nan),
                h2_metrics.get("solar_opex_million_per_year", np.nan),
                # h2_metrics.get("battery_opex_million_per_year", np.nan),
                h2_metrics.get("total_opex_million_per_year", np.nan),
                h2_metrics.get("npv_million", np.nan),
                h2_metrics.get("lcoe_eur_per_mwh", np.nan),
                h2_metrics.get("irr", np.nan),
                h2_metrics.get("total_curtailment_gwh", np.nan),
            ],
        }
    )
    comparison_df["h2_minus_hydesign"] = comparison_df["h2integrate"] - comparison_df["hydesign"]

    summary_path = comparison_dir / "hydesign_vs_h2integrate_summary.csv"
    comparison_df.to_csv(summary_path, index=False)
    latex_table_path = _write_latex_absolute_metrics_table(workspace_root, comparison_df)

    sources_path = comparison_dir / "comparison_sources.txt"
    sources_path.write_text(
        "\n".join(
            [
                f"HyDesign source: {hydesign_source}",
                f"h2integrate source: {h2_source}",
            ]
        ),
        encoding="utf-8",
    )

    _plot_side_by_side(comparison_dir, hydesign_metrics, h2_metrics)

    print(f"Saved: {summary_path}")
    print(f"Saved: {comparison_dir / 'hydesign_vs_h2integrate_side_by_side.png'}")
    print(f"Saved: {sources_path}")
    print(f"Saved: {latex_table_path}")


if __name__ == "__main__":
    main()
