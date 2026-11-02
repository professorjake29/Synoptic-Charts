# -*- coding: utf-8 -*-
"""
Updated 04 Oct 2026: Wyoming-style station title; profile legend removed.

Created on Thu Sep 24 06:01:39 2026

@author: jakew
"""

from datetime import datetime, timedelta, timezone
from io import StringIO

from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import metpy.calc as mpcalc
import numpy as np
import pandas as pd
import requests
from metpy.plots import SkewT
from metpy.units import units
import os

os.chdir("/mnt/c/Users/jakewiley/Documents")

station_number = "71811" # Used by Wyoming
station_id = "CYZV"  # Used by IEM
# Wyoming-style station name and backup coordinates for the example station.
# Add an entry here if you switch to a different WMO station number.
station_details = {
    "71811": ("SEPT-ILES UA, QUE, CANADA", 50.22, -66.24),
}
url = "https://weather.uwyo.edu/wsgi/sounding"
days_to_search = 3  # Check up to three dates independently for each cycle.
mixed_layer_depth_hpa = 100  # Depth of the layer mixed to define the parcel.


def sounding_params(sounding_time, data_type):
    return {
        "datetime": sounding_time.strftime("%Y-%m-%d %H:%M:%S"),
        "id": station_number,
        "src": "UNKNOWN",
        "type": data_type,
    }


def latest_sounding_for_cycle(hour, now):
    """Find the most recent available 00Z or 12Z sounding."""
    candidate = now.replace(hour=hour, minute=0, second=0, microsecond=0)
    if candidate > now:
        candidate -= timedelta(days=1)

    for day_offset in range(days_to_search):
        sounding_time = candidate - timedelta(days=day_offset)
        response = requests.get(
            url, params=sounding_params(sounding_time, "TEXT:CSV"), timeout=45
        )
        if response.status_code == 400 and response.text.startswith(
            "Unable to retrieve the data"
        ):
            print(f"{sounding_time:%Y-%m-%d %HZ}: not available")
            continue

        response.raise_for_status()
        data = pd.read_csv(StringIO(response.text))
        if len(data) > 1:
            print(f"{sounding_time:%Y-%m-%d %HZ}: available")
            return sounding_time, data
        print(f"{sounding_time:%Y-%m-%d %HZ}: no usable profile")

    return None


def get_indices(sounding_time):
    """Retrieve indices and station coordinates for the selected sounding."""
    response = requests.get(
        url, params=sounding_params(sounding_time, "INDICES"), timeout=45
    )
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    heading = soup.find("h3", string="Sounding Indices")
    table = heading.find_next("table") if heading else None
    if table is None:
        raise RuntimeError(
            f"Could not find Wyoming's indices for {sounding_time:%Y-%m-%d %HZ}."
        )

    indices = []
    station_coordinates = {}
    for row in table.find_all("tr"):
        cells = [cell.get_text(" ", strip=True) for cell in row.find_all("td")]
        if len(cells) == 4:
            code, description, value, unit = cells
            if code in {"SLAT", "SLON"}:
                try:
                    station_coordinates[code] = float(value)
                except ValueError:
                    pass
            elif code != "SELV":
                indices.append((code, description, value, unit))

    if not indices:
        raise RuntimeError(
            f"Wyoming returned no indices for {sounding_time:%Y-%m-%d %HZ}."
        )
    return indices, station_coordinates


def get_pwat_climatology(sounding_time):
    """Retrieve IEM PWAT climatological mean and median for this date/hour."""
    climo_url = (
        "https://mesonet.agron.iastate.edu/plotting/auto/plot/212/"
        f"network:RAOB::station:{station_id}::year:{sounding_time:%Y}"
        f"::hour:{sounding_time:%H}::var:pwater_mm"
        "::level:500::_r:t::dpi:100.csv"
    )

    response = requests.get(climo_url, timeout=45)
    response.raise_for_status()

    # IEM's CSV uses two header rows.
    climo = pd.read_csv(StringIO(response.text), header=[0, 1])

    # Match month, day, and sounding hour; e.g., 092412 for Sep 24 at 12Z.
    date_key = sounding_time.strftime("%m%d%H")
    matching_row = climo.loc[
        climo.iloc[:, 0].astype(str).str.zfill(6) == date_key
    ]

    if matching_row.empty:
        raise RuntimeError(
            f"No IEM PWAT climatology found for {station_id} on {date_key}."
        )

    climo_row = matching_row.iloc[0]
    pwat_mean = float(climo_row[("pwater_mm", "mean")])
    pwat_median = float(climo_row[("pwater_mm", "50%")])
    return pwat_mean, pwat_median


def add_pressure_level_heights(skew, data):
    """Label standard pressure levels with heights from the sounding, if given."""
    standard_levels = np.array([1000, 925, 850, 700, 500, 400, 300, 250, 200, 150, 100])
    height_column = next(
        (
            column for column in data.columns
            if ("height" in column.lower() or column.lower() == "hght")
            and (column.lower().endswith("_m") or column.lower() == "hght")
        ),
        None,
    )

    observed = None
    if height_column is not None:
        observed = (
            data.assign(height_m=pd.to_numeric(data[height_column], errors="coerce"))
            .dropna(subset=["pressure_hPa", "height_m"])
            .sort_values("pressure_hPa")
            .drop_duplicates("pressure_hPa")
        )

    if observed is not None and len(observed) >= 2:
        # Interpolate observed heights in log pressure; skip levels beyond the ascent.
        pressures = observed["pressure_hPa"].to_numpy()
        heights = observed["height_m"].to_numpy()
        levels = standard_levels[
            (standard_levels >= pressures.min()) & (standard_levels <= pressures.max())
        ]
        level_heights = np.interp(np.log(levels), np.log(pressures), heights)
        heading = "Height (m)"
    else:
        # The CSV does not always carry heights; label the approximation honestly.
        levels = standard_levels
        level_heights = mpcalc.pressure_to_height_std(levels * units.hPa).to("m").magnitude
        heading = "Approx. height (m)"

    skew.ax.text(
        0.015, 0.95, heading, transform=skew.ax.transAxes,
        fontsize=8, color="0.35", ha="left", va="top",
    )
    for level, height in zip(levels, level_heights):
        skew.ax.text(
            0.015, level, f"{height:.0f}",
            transform=skew.ax.get_yaxis_transform(which="tick1"),
            fontsize=8, color="0.35", ha="left",
            va="top" if level == 100 else "center",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.65, "pad": 0.1},
            zorder=5,
        )


def plot_sounding(sounding_time, data, indices, pwat_climo=None,
                  station_coordinates=None):
    """Build one Skew-T with matching Wyoming indices and PWAT climatology."""
    # Prepare the observed profile.
    data = data.copy()
    columns = [
        "pressure_hPa",
        "temperature_C",
        "dew point temperature_C",
        "wind direction_degree",
        "wind speed_m/s",
    ]
    for column in columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    data = data[data["pressure_hPa"].between(100, 1050)]

    thermo = (
        data.dropna(
            subset=["pressure_hPa", "temperature_C", "dew point temperature_C"]
        )
        .sort_values("pressure_hPa", ascending=False)
        .drop_duplicates("pressure_hPa")
    )

    if thermo.empty:
        raise RuntimeError("No usable temperature/dewpoint levels were found.")

    pressure = thermo["pressure_hPa"].to_numpy() * units.hPa
    temperature = thermo["temperature_C"].to_numpy() * units.degC
    dewpoint = thermo["dew point temperature_C"].to_numpy() * units.degC

    # Mix potential temperature and water-vapor mixing ratio over the lowest layer.
    mixing_depth = mixed_layer_depth_hpa * units.hPa
    if pressure[0] - pressure[-1] < mixing_depth:
        raise RuntimeError(
            f"The profile does not cover the lowest {mixed_layer_depth_hpa:g} hPa "
            "needed for the mixed-layer parcel."
        )
    ml_pressure, ml_temperature, ml_dewpoint = mpcalc.mixed_parcel(
        pressure, temperature, dewpoint, depth=mixing_depth
    )

    # Lift the mixed parcel from the surface pressure.
    parcel_pressure = np.arange(ml_pressure.to("hPa").magnitude, 99, -5) * units.hPa
    parcel_temperature = mpcalc.parcel_profile(
        parcel_pressure, ml_temperature, ml_dewpoint
    ).to("degC")

    wind = (
        data.dropna(
            subset=["pressure_hPa", "wind direction_degree", "wind speed_m/s"]
        )
        .sort_values("pressure_hPa", ascending=False)
        .copy()
    )
    wind["pressure_bin"] = (wind["pressure_hPa"] / 50).round()
    wind = wind.drop_duplicates("pressure_bin")

    # Reserve the right side of the figure for the indices.
    fig = plt.figure(figsize=(14, 10))
    skew = SkewT(fig, rotation=45, rect=(0.06, 0.10, 0.61, 0.80))

    skew.ax.set_ylim(1050, 100)
    skew.ax.set_xlim(-40, 50)

    skew.plot_dry_adiabats(colors="red", alpha=0.55, linewidths=0.8)
    skew.plot_moist_adiabats(colors="blue", alpha=0.5, linewidths=0.8)

    # Use the same mixing ratios for both the green lines and their labels.
    mixing_ratios = np.array([0.2, 0.4, 1, 2, 4, 7, 10, 16, 24, 32])
    skew.plot_mixing_lines(
        mixing_ratio=mixing_ratios * units("g/kg"),
        pressure=np.linspace(600, 1050, 100) * units.hPa,
        colors="green", linestyles="dashed", alpha=0.7, linewidths=0.8,
    )
    label_pressure = 1015 * units.hPa
    for ratio in mixing_ratios:
        label_temperature = mpcalc.dewpoint(
            mpcalc.vapor_pressure(label_pressure, ratio * units("g/kg"))
        ).to("degC")
        skew.ax.text(
            label_temperature.magnitude, label_pressure.magnitude, f"{ratio:g}",
            color="green", fontsize=8, ha="center", va="center",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 0.1},
            zorder=5,
        )

    add_pressure_level_heights(skew, data)


    skew.plot(pressure, temperature, color="black", linewidth=2.2,
              label="Temperature")
    skew.plot(pressure, dewpoint, color="black", linewidth=1.6,
              label="Dewpoint")
    skew.plot(parcel_pressure, parcel_temperature, color="0.45",
              linewidth=2, linestyle="-", label=f"{mixed_layer_depth_hpa:g}-hPa mixed-layer parcel")

    if not wind.empty:
        speed = wind["wind speed_m/s"].to_numpy() * units("m/s")
        direction = wind["wind direction_degree"].to_numpy() * units.degrees
        u, v = mpcalc.wind_components(speed, direction)

        skew.plot_barbs(
            wind["pressure_hPa"].to_numpy() * units.hPa,
            u.to("knots"),
            v.to("knots"),
            xloc=1.07,
            length=6.5,
        )

    station_name, default_lat, default_lon = station_details.get(
        station_number, (station_id, None, None)
    )
    coordinates = station_coordinates or {}
    latitude = coordinates.get("SLAT", default_lat)
    longitude = coordinates.get("SLON", default_lon)

    fig.text(
        0.02, 0.985,
        f"Station {station_number} at {sounding_time:%H UTC %d %b %Y}",
        ha="left", va="top", fontsize=16,
    )
    fig.text(0.02, 0.951, station_name, ha="left", va="top", fontsize=11)
    if latitude is not None and longitude is not None:
        fig.text(
            0.02, 0.929,
            f"Latitude: {latitude:.3f} Longitude: {longitude:.3f}",
            ha="left", va="top", fontsize=11,
        )
    skew.ax.set_xlabel("Temperature (°C)")
    skew.ax.set_ylabel("Pressure (hPa)")

    # Display the indices in a separate panel.
    short_names = {
        "SHOW": "Showalter",
        "LFVT": "Virtual LI",
        "SWET": "SWEAT",
        "KINX": "K Index",
        "CTOT": "Cross Totals",
        "VTOT": "Vertical Totals",
        "TOTL": "Total Totals",
        "DCAPE": "DCAPE",
        "MUCAPE": "MU CAPE",
        "MUCIN": "MU CIN",
        "LCLP": "LCL pressure",
        "LCLT": "LCL temperature",
        "LCLZ": "LCL height",
        "CCLP": "CCL pressure",
        "CCLT": "CCL temperature",
        "CCLC": "Conv. temperature",
        "PWAT": "Precipitable water",
    }

    short_units = {
        "delta_degree_Celsius": "°C",
        "degree_Celsius": "°C",
        "joule / kilogram": "J/kg",
        "hectopascal": "hPa",
        "meter": "m",
        "millimeter": "mm",
        "dimensionless": "",
    }

    index_ax = fig.add_axes((0.76, 0.12, 0.22, 0.76))
    index_ax.axis("off")
    index_ax.text(
        0.02, 0.98, "Wyoming sounding indices",
        transform=index_ax.transAxes,
        fontsize=13, fontweight="bold", va="top",
    )

    row_spacing = min(0.045, 0.76 / max(1, len(indices)))
    if not indices:
        index_ax.text(
            0.02, 0.91, "Indices unavailable",
            transform=index_ax.transAxes, fontsize=9.5, va="top",
        )

    for i, (code, description, value, unit) in enumerate(indices):
        y = 0.91 - i * row_spacing
        label = short_names.get(code, code)
        formatted_unit = short_units.get(unit, unit)
        formatted_value = f"{value} {formatted_unit}".strip()

        index_ax.text(
            0.02, y, label,
            transform=index_ax.transAxes, fontsize=9.5, va="top",
        )
        index_ax.text(
            0.58, y, formatted_value,
            transform=index_ax.transAxes,
            fontsize=9.5, va="top", ha="left",
        )

    # Place IEM PWAT climatology below the Wyoming indices.
    climo_y = 0.91 - len(indices) * row_spacing - 0.015
    if pwat_climo is None:
        index_ax.text(
            0.02, climo_y, "IEM PWAT climatology unavailable",
            transform=index_ax.transAxes, fontsize=9.5, va="top",
        )
    else:
        pwat_mean, pwat_median = pwat_climo
        for label, value in (
            ("IEM PWAT mean", pwat_mean),
            ("IEM PWAT median", pwat_median),
        ):
            index_ax.text(
                0.02, climo_y, label,
                transform=index_ax.transAxes, fontsize=9.5, va="top",
            )
            index_ax.text(
                0.58, climo_y, f"{value:.1f} mm",
                transform=index_ax.transAxes,
                fontsize=9.5, va="top", ha="left",
            )
            climo_y -= 0.045

    filename = f"sounding_{station_number}_{sounding_time:%Y%m%d_%HZ}.png"
    fig.savefig(filename, dpi=200, bbox_inches="tight")
    print(f"Saved {filename}")


def main():
    now = datetime.now(timezone.utc)
    plotted = 0
    for hour in (0, 12):
        latest = latest_sounding_for_cycle(hour, now)
        if latest is None:
            print(f"No {hour:02d}Z sounding found in the last {days_to_search} dates.")
            continue

        sounding_time, data = latest
        try:
            indices, station_coordinates = get_indices(sounding_time)
        except (requests.RequestException, RuntimeError) as error:
            print(f"Indices unavailable for {sounding_time:%Y-%m-%d %HZ}: {error}")
            indices = []
            station_coordinates = {}

        try:
            pwat_climo = get_pwat_climatology(sounding_time)
            print(
                f"IEM PWAT climatology for {sounding_time:%Y-%m-%d %HZ}: "
                f"mean={pwat_climo[0]:.1f} mm, median={pwat_climo[1]:.1f} mm"
            )
        except (requests.RequestException, RuntimeError, KeyError, ValueError) as error:
            print(
                f"PWAT climatology unavailable for "
                f"{sounding_time:%Y-%m-%d %HZ}: {error}"
            )
            pwat_climo = None

        plot_sounding(sounding_time, data, indices, pwat_climo,
                      station_coordinates=station_coordinates)
        plotted += 1

    if plotted == 0:
        raise RuntimeError(
            f"No 00Z or 12Z sounding found for {station_number} "
            f"in the last {days_to_search} dates."
        )
    


if __name__ == "__main__":
    main()

import subprocess

print("All charts completed successfully.")
print("Uploading charts to GitHub...")

subprocess.run(
    ['/home/wiley/upload_synoptic_charts_revised_current.sh'],
    check=True
)

print("Charts successfully uploaded to GitHub.")
