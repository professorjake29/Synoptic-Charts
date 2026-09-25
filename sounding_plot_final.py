# -*- coding: utf-8 -*-
"""
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

station_number = "71867"
url = "https://weather.uwyo.edu/wsgi/sounding"
days_to_search = 3  # Check up to three dates independently for each cycle.


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
    """Retrieve indices for the same station and cycle as the profile."""
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
    for row in table.find_all("tr"):
        cells = [cell.get_text(" ", strip=True) for cell in row.find_all("td")]
        if len(cells) == 4:
            code, description, value, unit = cells
            if code not in {"SLAT", "SLON", "SELV"}:
                indices.append((code, description, value, unit))

    if not indices:
        raise RuntimeError(
            f"Wyoming returned no indices for {sounding_time:%Y-%m-%d %HZ}."
        )
    return indices


def plot_sounding(sounding_time, data, indices):
    """Build one Skew-T with its matching Wyoming indices."""
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

    surface = thermo.iloc[0]
    p_surface = surface["pressure_hPa"] * units.hPa
    t_surface = surface["temperature_C"] * units.degC
    td_surface = surface["dew point temperature_C"] * units.degC

    parcel_pressure = np.arange(p_surface.magnitude, 99, -5) * units.hPa
    parcel_temperature = mpcalc.parcel_profile(
        parcel_pressure, t_surface, td_surface
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

    skew.plot_dry_adiabats(colors="0.55", alpha=0.5, linewidths=0.8)
    skew.plot_moist_adiabats(colors="tab:blue", alpha=0.35, linewidths=0.8)
    skew.plot_mixing_lines(colors="tab:green", alpha=0.35, linewidths=0.8)

    pressure = thermo["pressure_hPa"].to_numpy() * units.hPa
    temperature = thermo["temperature_C"].to_numpy() * units.degC
    dewpoint = thermo["dew point temperature_C"].to_numpy() * units.degC

    skew.plot(pressure, temperature, color="red", linewidth=2,
              label="Temperature")
    skew.plot(pressure, dewpoint, color="green", linewidth=2,
              label="Dewpoint")
    skew.plot(parcel_pressure, parcel_temperature, color="black",
              linewidth=2, linestyle="--", label="Surface-based parcel")

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

    skew.ax.set_title(
        f"Observed Sounding — {sounding_time:%Y-%m-%d %HZ}"
    )
    skew.ax.set_xlabel("Temperature (°C)")
    skew.ax.set_ylabel("Pressure (hPa)")
    skew.ax.legend(loc="upper left")

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

    index_ax = fig.add_axes((0.77, 0.12, 0.21, 0.76))
    index_ax.axis("off")
    index_ax.text(
        0.02, 0.98, "Wyoming sounding indices",
        transform=index_ax.transAxes,
        fontsize=13, fontweight="bold", va="top",
    )

    row_spacing = min(0.048, 0.83 / max(1, len(indices)))
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
            indices = get_indices(sounding_time)
        except (requests.RequestException, RuntimeError) as error:
            print(f"Indices unavailable for {sounding_time:%Y-%m-%d %HZ}: {error}")
            indices = []
        plot_sounding(sounding_time, data, indices)
        plotted += 1

    if plotted == 0:
        raise RuntimeError(
            f"No 00Z or 12Z sounding found for {station_number} "
            f"in the last {days_to_search} dates."
        )
    plt.show()


if __name__ == "__main__":
    main()
