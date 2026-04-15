import os

import matplotlib.pyplot as plt
import xarray as xr


def main() -> None:
    here = os.path.dirname(__file__)
    nc_path = os.path.join(here, "sweep_results.nc")
    out_path = os.path.join(here, "sweep_npv_over_capex_heatmap.png")

    ds = xr.open_dataset(nc_path)

    # Use dataset coordinates directly as axes.
    x = ds["surface_azimuth_deg"]
    y = ds["surface_tilt_deg"]
    z = ds["NPV_over_CAPEX"]

    fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    mesh = ax.pcolormesh(x, y, z, shading="auto", cmap="viridis")

    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label("NPV_over_CAPEX")

    ax.set_title("NPV_over_CAPEX Heatmap")
    ax.set_xlabel("surface_azimuth_deg")
    ax.set_ylabel("surface_tilt_deg")

    fig.savefig(out_path, dpi=200)
    print(f"Saved heatmap to: {out_path}")


if __name__ == "__main__":
    main()
