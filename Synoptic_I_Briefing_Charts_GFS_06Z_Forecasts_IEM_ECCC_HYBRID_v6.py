from datetime import datetime, timedelta
import matplotlib
matplotlib.use('Agg')
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import metpy.calc as mpcalc
from metpy.units import units
import numpy as np
from siphon.ncss import NCSS
import cartopy.util as cutil
import matplotlib.gridspec as gridspec
from metpy.plots import StationPlot
import pandas as pd
import matplotlib.patheffects as PathEffects
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.ticker import FuncFormatter
import os
import time
import requests
from io import StringIO


from pathlib import Path


# =============================================================================
# FORECAST CITY LOCATION
# =============================================================================
# Set these two values once for the city/location being forecast.
# Use decimal degrees: north/east positive, south/west negative.
#
# Examples:
#   30.69 N, 88.18 W  ->  30.69, -88.18
#
# Leave either value as None to disable the marker.
FORECAST_CITY_LAT = 50.22
FORECAST_CITY_LON = -66.24

# Marker appearance for all non-DIFAX charts.
FORECAST_CITY_MARKER = '*'
FORECAST_CITY_MARKER_SIZE = 220
FORECAST_CITY_MARKER_COLOR = 'gold'
FORECAST_CITY_MARKER_EDGE_COLOR = 'black'
FORECAST_CITY_MARKER_EDGE_WIDTH = 1.2


def plot_forecast_city_marker(ax):
    """
    Plot the user-specified forecast location on a Cartopy map.

    The marker is intentionally drawn at a high z-order so it remains visible
    above contours, filled fields, wind barbs, coastlines, and other map layers.
    DIFAX charts do not call this helper.
    """
    if FORECAST_CITY_LAT is None or FORECAST_CITY_LON is None:
        return

    if not (-90.0 <= FORECAST_CITY_LAT <= 90.0):
        raise ValueError("FORECAST_CITY_LAT must be between -90 and 90 degrees.")

    if not (-180.0 <= FORECAST_CITY_LON <= 180.0):
        raise ValueError("FORECAST_CITY_LON must be between -180 and 180 degrees.")

    ax.scatter(
        FORECAST_CITY_LON,
        FORECAST_CITY_LAT,
        s=FORECAST_CITY_MARKER_SIZE,
        marker=FORECAST_CITY_MARKER,
        facecolor=FORECAST_CITY_MARKER_COLOR,
        edgecolor=FORECAST_CITY_MARKER_EDGE_COLOR,
        linewidth=FORECAST_CITY_MARKER_EDGE_WIDTH,
        transform=ccrs.PlateCarree(),
        zorder=50,
        clip_on=True
    )





# WMO station numbers used by the University of Wyoming sounding service.
# Existing short station codes are retained elsewhere for readable logging.

IEM_RAOB_URL = "https://mesonet.agron.iastate.edu/json/raob.py"
IEM_RAOB_NETWORK_URL = "https://mesonet.agron.iastate.edu/geojson/network.py"

# Cache station metadata after the first request.
_IEM_RAOB_METADATA = None



# Explicit IEM RAOB aliases for Canadian upper-air stations used by this script.
# These are normalized to the legacy 3-letter station identifiers already used
# throughout the DIFAX station lists.
IEM_STATION_ALIASES = {
    "CAWE": "AWE",
    "CWMW": "WMW",
    "CWSE": "WSE",
    "CWVK": "WVK",
    "CYAH": "YAH",
    "CYJT": "YJT",
    "CYPL": "YPL",
    "CYQD": "YQD",
    "CYQI": "YQI",
    "CYSM": "YSM",
    "CYVP": "YVP",
    "CYVQ": "YVQ",
    "CYYE": "YYE",
    "CYYQ": "YYQ",
    "CYYR": "YYR",
    "CYYT": "YYT",
    "CYZT": "YZT",
    "CYZV": "YZV",
    "CZXS": "ZXS",
    "CWPL": "YPL",
    "CWQI": "YQI",
    "CYXS": "YXS",
    "CYMW": "YMW",
}

def normalize_iem_raob_station(station_id):
    """
    Normalize IEM RAOB station identifiers to the 3-letter identifiers
    already used by the DIFAX station lists.

    Canadian stations are handled first through an explicit alias table
    because their identifiers are not always a simple leading-C + 3-letter
    code transformation.
    """
    if station_id is None:
        return None

    sid = str(station_id).strip().upper()

    # Explicit Canadian/legacy aliases take priority.
    if sid in IEM_STATION_ALIASES:
        return IEM_STATION_ALIASES[sid]

    # Some IEM identifiers can carry a leading underscore.
    if sid.startswith("_") and len(sid) == 4:
        return sid[1:]

    # U.S. ICAO identifiers generally map cleanly by removing the leading K.
    if len(sid) == 4 and sid.startswith("K"):
        return sid[1:]

    return sid


def get_iem_raob_metadata(max_retries=3, wait_seconds=5):
    """Retrieve IEM RAOB station latitude/longitude metadata once."""
    global _IEM_RAOB_METADATA

    if _IEM_RAOB_METADATA is not None:
        return _IEM_RAOB_METADATA

    params = {
        "network": "RAOB",
        "only_online": "0",
    }

    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(
                IEM_RAOB_NETWORK_URL,
                params=params,
                timeout=30,
            )
            response.raise_for_status()
            payload = response.json()

            metadata = {}

            for feature in payload.get("features", []):
                props = feature.get("properties", {}) or {}
                geometry = feature.get("geometry", {}) or {}
                coords = geometry.get("coordinates", [])

                sid = (
                    feature.get("id")
                    or props.get("sid")
                    or props.get("station")
                    or props.get("id")
                )

                if sid is None or len(coords) < 2:
                    continue

                lon, lat = coords[0], coords[1]

                try:
                    lon = float(lon)
                    lat = float(lat)
                except (TypeError, ValueError):
                    continue

                if not (np.isfinite(lon) and np.isfinite(lat)):
                    continue

                normalized = normalize_iem_raob_station(sid)
                metadata[normalized] = {
                    "latitude": lat,
                    "longitude": lon,
                    "iem_station": str(sid).upper(),
                }

            if metadata:
                _IEM_RAOB_METADATA = metadata
                print(
                    f"IEM RAOB metadata loaded for {len(metadata)} station identifiers."
                )
                return metadata

            raise RuntimeError("IEM RAOB metadata response contained no usable stations.")

        except (requests.RequestException, ValueError, RuntimeError) as exc:
            print(
                f"IEM RAOB metadata request failed "
                f"(attempt {attempt}/{max_retries}): {exc}"
            )

            if attempt < max_retries:
                time.sleep(wait_seconds)

    raise RuntimeError(
        "Unable to retrieve IEM RAOB station metadata after repeated attempts."
    )


def get_iem_raob_level(valid_time, pressure_level, max_retries=3, wait_seconds=5):
    """
    Retrieve all IEM RAOB observations at one pressure level and UTC cycle.

    IEM returns wind speed in knots. u/v are converted to m/s here so the
    existing DIFAX StationPlot code can remain unchanged.
    """
    params = {
        "ts": valid_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "pressure": int(pressure_level),
    }

    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(
                IEM_RAOB_URL,
                params=params,
                timeout=45,
            )
            response.raise_for_status()
            payload = response.json()

            rows = []
            metadata = get_iem_raob_metadata()

            for sounding in payload.get("profiles", []):
                iem_station = sounding.get("station")
                station = normalize_iem_raob_station(iem_station)

                if station is None:
                    continue

                station_meta = metadata.get(station)
                if station_meta is None:
                    continue

                for level in sounding.get("profile", []):
                    pressure = level.get("pres")

                    if pressure is None:
                        continue

                    try:
                        pressure = float(pressure)
                    except (TypeError, ValueError):
                        continue

                    # The service was queried at a specific pressure level,
                    # but retain this guard for robustness.
                    if not np.isclose(pressure, float(pressure_level), atol=0.1):
                        continue

                    height = level.get("hght")
                    temperature = level.get("tmpc")
                    dewpoint = level.get("dwpc")
                    direction = level.get("drct")
                    speed_knots = level.get("sknt")

                    def numeric_or_nan(value):
                        try:
                            return float(value) if value is not None else np.nan
                        except (TypeError, ValueError):
                            return np.nan

                    height = numeric_or_nan(height)
                    temperature = numeric_or_nan(temperature)
                    dewpoint = numeric_or_nan(dewpoint)
                    direction = numeric_or_nan(direction)
                    speed_knots = numeric_or_nan(speed_knots)

                    u_wind = np.nan
                    v_wind = np.nan

                    if np.isfinite(direction) and np.isfinite(speed_knots):
                        u_component, v_component = mpcalc.wind_components(
                            speed_knots * units.knots,
                            direction * units.degrees,
                        )
                        u_wind = u_component.to("m/s").magnitude
                        v_wind = v_component.to("m/s").magnitude

                    rows.append(
                        {
                            "pressure": pressure,
                            "height": height,
                            "temperature": temperature,
                            "dewpoint": dewpoint,
                            "direction": direction,
                            "speed": speed_knots,
                            "u_wind": u_wind,
                            "v_wind": v_wind,
                            "station": station,
                            "time": valid_time,
                            "latitude": station_meta["latitude"],
                            "longitude": station_meta["longitude"],
                        }
                    )

            result = pd.DataFrame(rows)

            if result.empty:
                print(
                    f"No IEM RAOB observations found at {pressure_level} hPa "
                    f"for {valid_time:%Y-%m-%d %HZ}."
                )

            return result

        except (requests.RequestException, ValueError) as exc:
            print(
                f"IEM RAOB request failed for {pressure_level} hPa "
                f"at {valid_time:%Y-%m-%d %HZ} "
                f"(attempt {attempt}/{max_retries}): {exc}"
            )

            if attempt < max_retries:
                time.sleep(wait_seconds)

    print(
        f"Skipping IEM RAOB {pressure_level}-hPa data for "
        f"{valid_time:%Y-%m-%d %HZ} after {max_retries} failed attempts."
    )

    return pd.DataFrame(
        columns=[
            "pressure", "height", "temperature", "dewpoint",
            "direction", "speed", "u_wind", "v_wind",
            "station", "time", "latitude", "longitude"
        ]
    )


def get_iem_raob_difax_data(valid_time, pressure_levels=(500, 700, 850)):
    """
    Retrieve all pressure levels needed by the DIFAX charts.

    This makes one bulk IEM request per requested pressure level rather than
    one request per sounding station.
    """
    frames = []

    for pressure_level in pressure_levels:
        print(
            f"Retrieving IEM RAOB {pressure_level}-hPa observations "
            f"for {valid_time:%Y-%m-%d %HZ}..."
        )
        frame = get_iem_raob_level(valid_time, pressure_level)

        if not frame.empty:
            frames.append(frame)

    if not frames:
        return pd.DataFrame(
            columns=[
                "pressure", "height", "temperature", "dewpoint",
                "direction", "speed", "u_wind", "v_wind",
                "station", "time", "latitude", "longitude"
            ]
        )

    return pd.concat(frames, ignore_index=True)




ECCC_CANADIAN_STATIONS = {
    "AWE": "CAWE",
    "WMW": "CWMW",
    "WSE": "CWSE",
    "WVK": "CWVK",
    "YAH": "CYAH",
    "YJT": "CYJT",
    "YPL": "CYPL",
    "YQD": "CYQD",
    "YQI": "CYQI",
    "YSM": "CYSM",
    "YVP": "CYVP",
    "YVQ": "CYVQ",
    "YYE": "CYYE",
    "YYQ": "CYYQ",
    "YYR": "CYYR",
    "YYT": "CYYT",
    "YZT": "CYZT",
    "YZV": "CYZV",
    "ZXS": "CZXS",
}

_ECCC_RAOB_CACHE = {}


def _empty_difax_dataframe():
    return pd.DataFrame(
        columns=[
            "pressure", "height", "temperature", "dewpoint",
            "direction", "speed", "u_wind", "v_wind",
            "station", "time", "latitude", "longitude"
        ]
    )


def _eccc_obs_urls(valid_time, eccc_station):
    filename = f"ObsTephi_{valid_time:%H}_{eccc_station}.csv"
    dated = (
        f"https://dd.weather.gc.ca/{valid_time:%Y%m%d}/WXO-DD/"
        f"vertical_profile/observation/csv/{filename}"
    )

    now_utc = datetime.now(timezone.utc)

    if valid_time.date() == now_utc.date():
        today = (
            "https://dd.weather.gc.ca/today/"
            f"vertical_profile/observation/csv/{filename}"
        )
        return [today, dated]

    return [dated]


def _find_eccc_pressure_column(frame):
    preferred = [
        "PRESSURE", "PRES", "PRESS", "P", "LEVEL", "LEV",
        "HPA", "MB", "NIVEAU"
    ]

    upper_lookup = {
        str(col).strip().upper(): col
        for col in frame.columns
    }

    for name in preferred:
        if name in upper_lookup:
            return upper_lookup[name]

    target_levels = np.array([500.0, 700.0, 850.0])

    for col in frame.columns:
        vals = pd.to_numeric(frame[col], errors="coerce").dropna().values

        if vals.size == 0:
            continue

        matches = sum(
            np.any(np.isclose(vals, level, atol=0.1))
            for level in target_levels
        )

        if matches >= 2:
            return col

    return None


def _find_eccc_column(frame, candidates):
    lookup = {
        str(col).strip().upper(): col
        for col in frame.columns
    }

    for candidate in candidates:
        key = candidate.upper()
        if key in lookup:
            return lookup[key]

    return None


def _safe_float(value):
    """Convert ECCC numeric text to float, returning NaN for blanks/flags."""
    try:
        value = str(value).strip()
        if value == "" or value.lower() in {"nan", "-nan", "none"}:
            return np.nan
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _eccc_rh_from_t_td(temp_c, dewpoint_c):
    """
    Approximate relative humidity (%) from temperature/dewpoint.
    Used only to identify the correct thermodynamic fields in malformed
    ObsTephi rows; it is not plotted.
    """
    if not (np.isfinite(temp_c) and np.isfinite(dewpoint_c)):
        return np.nan

    try:
        es = 6.112 * np.exp((17.67 * temp_c) / (temp_c + 243.5))
        e = 6.112 * np.exp((17.67 * dewpoint_c) / (dewpoint_c + 243.5))
        return 100.0 * e / es
    except Exception:
        return np.nan


def _parse_eccc_thermo_row(fields, target_level):
    """
    Parse one malformed ECCC thermodynamic row.

    ObsTephi's first table is not reliably rectangular: optional height,
    thickness, stability, and wind fields can disappear without maintaining
    empty placeholders. Therefore temperature/dewpoint cannot safely be read
    by a fixed column index.

    We identify the thermodynamic quartet:
        TEMP, SPRED, THETAW, RELHUM
    by testing plausible consecutive numeric groups and selecting the group
    whose TEMP/SPRED combination reproduces the listed RH most closely.
    """
    values = [str(x).strip() for x in fields]

    # Remove pressure; examine the remainder.
    rest = values[1:]

    candidates = []

    # Search every 4-value consecutive group for:
    # T, dewpoint depression, theta-w-like value, RH
    for i in range(0, max(0, len(rest) - 3)):
        t = _safe_float(rest[i])
        spread = _safe_float(rest[i + 1])
        theta_w = _safe_float(rest[i + 2])
        rh = _safe_float(rest[i + 3])

        if not all(np.isfinite(x) for x in (t, spread, theta_w, rh)):
            continue

        # Broad meteorological plausibility filters.
        if not (-100.0 <= t <= 60.0):
            continue
        if not (0.0 <= spread <= 80.0):
            continue
        if not (-100.0 <= theta_w <= 80.0):
            continue
        if not (0.0 <= rh <= 100.0):
            continue

        td = t - spread
        rh_calc = _eccc_rh_from_t_td(t, td)

        if not np.isfinite(rh_calc):
            continue

        # Primary discriminator: T and dewpoint depression should reproduce RH.
        rh_error = abs(rh_calc - rh)

        # Small penalties discourage obviously non-thermodynamic groups.
        penalty = 0.0
        if td > t:
            penalty += 100.0
        if spread > 50:
            penalty += 20.0

        score = rh_error + penalty
        candidates.append((score, i, t, spread, theta_w, rh, td))

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[0])
    score, i, t, spread, theta_w, rh, td = candidates[0]

    # Require a reasonably self-consistent T/Td/RH combination.
    if score > 20.0:
        return None

    return {
        "pressure": float(target_level),
        "temperature": t,
        "dewpoint_depression": spread,
        "dewpoint": td,
        "relative_humidity": rh,
        "theta_w": theta_w,
    }


def _parse_eccc_wind_section(lines):
    """
    Parse the second ObsTephi table:
        Pres,Level,Level,Wind,Wind
        (mb),(m),(ft),(deg),(kt)

    Returns a dict keyed by pressure with height (m), direction (deg),
    and speed (kt).
    """
    wind_header_index = None

    for i, line in enumerate(lines):
        compact = line.replace(" ", "").upper()
        if compact.startswith("PRES,LEVEL,LEVEL,WIND,WIND"):
            wind_header_index = i
            break

    if wind_header_index is None:
        return {}

    # Units line follows the header, then data.
    data_start = wind_header_index + 2
    result = {}

    for line in lines[data_start:]:
        if not line.strip():
            continue

        fields = [part.strip() for part in line.split(",")]

        if len(fields) < 5:
            continue

        pressure = _safe_float(fields[0])
        height_m = _safe_float(fields[1])
        wind_dir = _safe_float(fields[3])
        wind_speed = _safe_float(fields[4])

        if not np.isfinite(pressure):
            continue

        # Only standard DIFAX levels are needed.
        for target_level in (500.0, 700.0, 850.0):
            if np.isclose(pressure, target_level, atol=0.1):

                # ECCC's ObsTephi wind table can encode lower-tropospheric
                # mandatory-level geopotential heights in truncated WMO form.
                # Reconstruct full geopotential metres before combining with
                # IEM data so station-model formatting and 12-h height changes
                # are consistent between U.S. and Canadian stations.
                #
                # Examples:
                #   850 hPa: 336  -> 1336 m
                #   700 hPa: 908  -> 2908 m
                #   500 hPa: 5491 -> 5491 m (already full height)
                if np.isfinite(height_m):
                    if np.isclose(target_level, 850.0) and height_m < 1000.0:
                        height_m += 1000.0
                    elif np.isclose(target_level, 700.0) and height_m < 2000.0:
                        height_m += 2000.0

                result[int(target_level)] = {
                    "height": height_m,
                    "direction": wind_dir,
                    "speed": wind_speed,
                }
                break

    return result


def _parse_eccc_obstephi_csv(csv_text, legacy_station, valid_time):
    """
    Parse an ECCC ObsTephi file by treating it as two distinct tables:

      1) malformed thermodynamic table -> T and dewpoint depression
      2) rectangular wind table       -> height, wind direction, wind speed

    The two products are merged by pressure at 850, 700, and 500 hPa.
    """
    if csv_text is None:
        return _empty_difax_dataframe()

    raw_lines = [
        line.strip("\ufeff\r\n")
        for line in csv_text.splitlines()
    ]

    nonempty = [line for line in raw_lines if line.strip()]

    if not nonempty:
        return _empty_difax_dataframe()

    joined_upper = " ".join(nonempty[:8]).upper()

    if "NO REPORT AVAILABLE" in joined_upper:
        print(
            f"ECCC {legacy_station} {valid_time:%Y-%m-%d %HZ}: "
            "no report available."
        )
        return _empty_difax_dataframe()

    # Locate where the second (wind) table begins.
    wind_header_index = None
    for i, line in enumerate(raw_lines):
        compact = line.replace(" ", "").upper()
        if compact.startswith("PRES,LEVEL,LEVEL,WIND,WIND"):
            wind_header_index = i
            break

    thermo_lines = (
        raw_lines[:wind_header_index]
        if wind_header_index is not None
        else raw_lines
    )

    # Extract the requested standard-level thermodynamic rows.
    thermo = {}

    for line in thermo_lines:
        if not line.strip():
            continue

        fields = [part.strip() for part in line.split(",")]

        if not fields:
            continue

        pressure = _safe_float(fields[0])

        if not np.isfinite(pressure):
            continue

        for target_level in (500, 700, 850):
            if np.isclose(pressure, float(target_level), atol=0.1):
                parsed = _parse_eccc_thermo_row(fields, target_level)
                if parsed is not None:
                    thermo[target_level] = parsed
                break

    winds = _parse_eccc_wind_section(raw_lines)

    latitude = np.nan
    longitude = np.nan

    # IEM metadata remains useful for coordinates even when IEM has no
    # Canadian sounding observation.
    try:
        metadata = get_iem_raob_metadata()
        station_meta = metadata.get(legacy_station)

        if station_meta is not None:
            latitude = station_meta["latitude"]
            longitude = station_meta["longitude"]
    except Exception as exc:
        print(
            f"Could not obtain metadata coordinates for Canadian station "
            f"{legacy_station}: {exc}"
        )

    rows = []

    for target_level in (500, 700, 850):
        thermo_row = thermo.get(target_level)
        wind_row = winds.get(target_level)

        if thermo_row is None:
            print(
                f"ECCC {legacy_station} {valid_time:%Y-%m-%d %HZ}: "
                f"thermodynamic data unavailable at {target_level} hPa."
            )
            continue

        if wind_row is None:
            print(
                f"ECCC {legacy_station} {valid_time:%Y-%m-%d %HZ}: "
                f"wind/height data unavailable at {target_level} hPa."
            )
            continue

        hght = wind_row["height"]
        wdir = wind_row["direction"]
        wspd = wind_row["speed"]

        u_wind = np.nan
        v_wind = np.nan

        if np.isfinite(wdir) and np.isfinite(wspd):
            u_component, v_component = mpcalc.wind_components(
                wspd * units.knots,
                wdir * units.degrees,
            )
            u_wind = u_component.to("m/s").magnitude
            v_wind = v_component.to("m/s").magnitude

        rows.append(
            {
                "pressure": float(target_level),
                "height": hght,
                "temperature": thermo_row["temperature"],
                "dewpoint": thermo_row["dewpoint"],
                "direction": wdir,
                "speed": wspd,
                "u_wind": u_wind,
                "v_wind": v_wind,
                "station": legacy_station,
                "time": valid_time,
                "latitude": latitude,
                "longitude": longitude,
            }
        )

    if not rows:
        return _empty_difax_dataframe()

    result = pd.DataFrame(rows)

    if not (np.isfinite(latitude) and np.isfinite(longitude)):
        print(
            f"ECCC {legacy_station} {valid_time:%Y-%m-%d %HZ}: "
            "profile parsed but station coordinates are unavailable; skipping."
        )
        return _empty_difax_dataframe()

    # Useful runtime QC for Canadian data.
    qc = ", ".join(
        f"{int(row.pressure)} hPa: "
        f"T={row.temperature:.1f}C "
        f"Td={row.dewpoint:.1f}C "
        f"Wind={row.direction:.0f}/{row.speed:.0f}kt"
        for row in result.itertuples()
    )
    print(
        f"ECCC {legacy_station} {valid_time:%Y-%m-%d %HZ} parsed: {qc}"
    )

    return result


def get_eccc_canadian_difax_data(valid_time, max_retries=2, wait_seconds=3):
    cache_key = valid_time.strftime("%Y%m%d%H")

    if cache_key in _ECCC_RAOB_CACHE:
        return _ECCC_RAOB_CACHE[cache_key].copy()

    frames = []
    available = []
    unavailable = []

    headers = {"User-Agent": "USA-Meteorology-Synoptic-Charts/1.0"}

    for legacy_station, eccc_station in ECCC_CANADIAN_STATIONS.items():
        retrieved = False

        for url in _eccc_obs_urls(valid_time, eccc_station):
            for attempt in range(1, max_retries + 1):
                try:
                    response = requests.get(
                        url,
                        timeout=30,
                        headers=headers,
                    )

                    if response.status_code == 404:
                        break

                    response.raise_for_status()

                    station_frame = _parse_eccc_obstephi_csv(
                        response.text,
                        legacy_station,
                        valid_time,
                    )

                    if not station_frame.empty:
                        frames.append(station_frame)
                        available.append(legacy_station)
                        retrieved = True

                    break

                except requests.RequestException as exc:
                    if attempt == max_retries:
                        print(
                            f"ECCC request failed for {legacy_station} "
                            f"({eccc_station}) at {valid_time:%Y-%m-%d %HZ}: {exc}"
                        )
                    else:
                        time.sleep(wait_seconds)

            if retrieved:
                break

        if not retrieved:
            unavailable.append(legacy_station)

    result = (
        pd.concat(frames, ignore_index=True)
        if frames
        else _empty_difax_dataframe()
    )

    print(
        f"ECCC Canadian RAOB retrieval for {valid_time:%Y-%m-%d %HZ}: "
        f"{len(available)} stations with usable 500/700/850-hPa data."
    )

    if available:
        print("  Available Canadian stations: " + " ".join(sorted(available)))

    if unavailable:
        print(
            "  Canadian stations unavailable/empty: "
            + " ".join(sorted(unavailable))
        )

    _ECCC_RAOB_CACHE[cache_key] = result.copy()
    return result


def get_hybrid_raob_difax_data(valid_time):
    iem = get_iem_raob_difax_data(valid_time)

    canadian_ids = set(ECCC_CANADIAN_STATIONS.keys())

    if not iem.empty:
        iem = iem[~iem["station"].isin(canadian_ids)].copy()

    eccc = get_eccc_canadian_difax_data(valid_time)

    frames = [frame for frame in (iem, eccc) if not frame.empty]

    if not frames:
        return _empty_difax_dataframe()

    combined = pd.concat(frames, ignore_index=True)

    combined = combined.drop_duplicates(
        subset=["station", "pressure"],
        keep="last",
    )

    return combined


def print_canadian_raob_diagnostics(dataframe, valid_time):
    """Print Canadian RAOB stations present in a retrieved IEM dataset."""
    if dataframe.empty or "station" not in dataframe.columns:
        print(f"Canadian RAOB stations retrieved for {valid_time:%Y-%m-%d %HZ}: none")
        return

    canadian_targets = set(IEM_STATION_ALIASES.values())
    present = sorted(
        set(dataframe["station"].dropna().astype(str).str.upper())
        & canadian_targets
    )

    if present:
        print(
            f"Canadian RAOB stations retrieved for {valid_time:%Y-%m-%d %HZ}: "
            + " ".join(present)
        )
    else:
        print(f"Canadian RAOB stations retrieved for {valid_time:%Y-%m-%d %HZ}: none")


def prepare_difax_level(data_current, data_previous, level):
    """
    Match current and previous observations by station before calculating
    12-hour height changes. This prevents array-length mismatch failures.
    """
    current = data_current[data_current["pressure"] == level].copy()
    previous = data_previous[data_previous["pressure"] == level].copy()

    current = current.drop_duplicates(subset="station")
    previous = previous.drop_duplicates(subset="station")

    previous_heights = previous[["station", "height"]].rename(
        columns={"height": "previous_height"}
    )

    current = current.merge(
        previous_heights,
        on="station",
        how="inner",
    )

    current["height_change"] = (
        current["height"] - current["previous_height"]
    )

    return current



def plot_maxmin_points(
    ax, lon, lat, data, extrema, nsize, symbol,
    color='k',
    plot_value=True,
    transform=ccrs.PlateCarree(),
    edge_buffer=0.02,
    min_dist=10e5,
    symbol_size=36,
    value_size=12,
    value_offset=-1.0,
    value_offset_points=None,
    stroke_color=None,
    stroke_width=2
):
    """
    Plot local maxima/minima while filtering invalid projected coordinates.

    This avoids the NaN-coordinate rendering failures that can occur with
    MetPy scattertext() when an extrema point projects outside/onto an invalid
    portion of the map.

    Parameters
    ----------
    extrema : {'max', 'min'}
        Type of extrema to locate.
    nsize : int
        Neighborhood size for the extrema filter.
    min_dist : float
        Minimum projected distance between retained extrema.
    value_offset_points : float or None
        Optional vertical label offset in display points. When supplied, this
        is preferred over value_offset and gives consistent visual spacing
        across distorted map projections.
    stroke_color : str or None
        Optional outline color for symbols/values.
    """
    from scipy.ndimage import maximum_filter, minimum_filter

    data = np.asarray(data)

    if extrema == 'max':
        data_ext = maximum_filter(data, nsize, mode='nearest')
    elif extrema == 'min':
        data_ext = minimum_filter(data, nsize, mode='nearest')
    else:
        raise ValueError("extrema must be 'max' or 'min'")

    # Ensure lon/lat are 2D.
    if np.ndim(lon) == 1 and np.ndim(lat) == 1:
        lon, lat = np.meshgrid(lon, lat)

    # Only consider finite data values.
    finite_data = np.isfinite(data)
    mxx, mxy = np.where((data_ext == data) & finite_data)

    if len(mxx) == 0:
        return

    points = np.stack([lon[mxx, mxy], lat[mxx, mxy]], axis=1)

    # Remove invalid source coordinates before projection.
    finite_source = np.isfinite(points).all(axis=1)
    points = points[finite_source]
    mxx = mxx[finite_source]
    mxy = mxy[finite_source]

    if len(points) == 0:
        return

    proj_points = ax.projection.transform_points(
        transform, points[:, 0], points[:, 1]
    )

    # Explicitly reject any non-finite projected coordinates.
    finite_proj = np.isfinite(proj_points[:, :2]).all(axis=1)
    points = points[finite_proj]
    proj_points = proj_points[finite_proj]
    mxx = mxx[finite_proj]
    mxy = mxy[finite_proj]

    if len(points) == 0:
        return

    # Get displayed extent in the axes projection.
    x0, x1, y0, y1 = ax.get_extent()
    dx = (x1 - x0) * edge_buffer
    dy = (y1 - y0) * edge_buffer
    x0_buff, x1_buff = x0 + dx, x1 - dx
    y0_buff, y1_buff = y0 + dy, y1 - dy

    inside = (
        (proj_points[:, 0] >= x0_buff)
        & (proj_points[:, 0] <= x1_buff)
        & (proj_points[:, 1] >= y0_buff)
        & (proj_points[:, 1] <= y1_buff)
    )

    points = points[inside]
    proj_keep = proj_points[inside, :2]
    data_keep = data[mxx, mxy][inside]

    if len(points) == 0:
        return

    # Thin clustered extrema.
    selected_idx = []
    for i, point in enumerate(proj_keep):
        if not selected_idx:
            selected_idx.append(i)
            continue

        distances = np.linalg.norm(
            proj_keep[selected_idx] - point,
            axis=1
        )

        if np.all(distances >= min_dist):
            selected_idx.append(i)

    points = points[selected_idx]
    data_keep = data_keep[selected_idx]

    # Plot the symbol and value with ordinary Matplotlib text objects.
    for (x, y), val in zip(points, data_keep):
        symbol_text = ax.text(
            x, y, symbol,
            size=symbol_size,
            color=color,
            fontweight='bold',
            ha='center',
            va='center',
            transform=transform,
            clip_on=True,
            zorder=10
        )

        if stroke_color is not None:
            symbol_text.set_path_effects([
                PathEffects.withStroke(
                    linewidth=stroke_width,
                    foreground=stroke_color
                )
            ])

        if plot_value:
            if value_offset_points is not None:
                # Use a fixed screen-space offset so the value remains the
                # same visual distance below the H/L everywhere on highly
                # distorted projections such as NorthPolarStereo.
                value_text = ax.annotate(
                    f"{val:.0f}",
                    xy=(x, y),
                    xycoords=transform._as_mpl_transform(ax),
                    xytext=(0, value_offset_points),
                    textcoords='offset points',
                    fontsize=value_size,
                    fontweight='bold',
                    color=color,
                    ha='center',
                    va='top',
                    annotation_clip=True,
                    zorder=10
                )
            else:
                # Preserve the original map-coordinate offset behavior for
                # all other charts unless a point offset is requested.
                value_text = ax.text(
                    x, y + value_offset,
                    f"{val:.0f}",
                    size=value_size,
                    fontweight='bold',
                    color=color,
                    ha='center',
                    va='top',
                    transform=transform,
                    clip_on=True,
                    zorder=10
                )

            if stroke_color is not None:
                value_text.set_path_effects([
                    PathEffects.withStroke(
                        linewidth=stroke_width,
                        foreground=stroke_color
                    )
                ])


output_dir = Path("/mnt/c/Users/jakewiley/Documents")

for png_file in output_dir.glob("*.png"):
    png_file.unlink()

print("Existing PNG files removed.")

os.chdir(r"/mnt/c/Users/jakewiley/Documents")

from datetime import datetime, timezone

now_utc = datetime.now(timezone.utc)

# 00Z analysis date
master_date_00Z = now_utc.strftime("%Y%m%d_0000")
year_00Z = now_utc.year
month_00Z = now_utc.month
month_name_00Z = now_utc.strftime("%B")
day_00Z = now_utc.day
hour_00Z = "00"

# 12Z analysis date
master_date_12Z = now_utc.strftime("%Y%m%d_1200")
year_12Z = now_utc.year
month_12Z = now_utc.month
month_name_12Z = now_utc.strftime("%B")
day_12Z = now_utc.day
hour_12Z = "12"

# Forecast charts use the 06Z GFS cycle so they are available earlier,
# while retaining the same valid times previously produced from the
# 12Z cycle at +12 h and +24 h. Therefore, the 06Z forecasts use
# +18 h and +30 h lead times, respectively.

date_12z_for_forecast = datetime(year_12Z, month_12Z, day_12Z, int(hour_12Z))
date_06z = date_12z_for_forecast - timedelta(hours=6)

master_date_06Z = date_06z.strftime("%Y%m%d_%H%M")
year_06Z = date_06z.year
month_06Z = date_06z.month
month_name_06Z = date_06z.strftime("%B")
day_06Z = date_06z.day
hour_06Z = date_06z.strftime("%H")

fcst_18hr = date_06z + timedelta(hours=18)
fcst_30hr = date_06z + timedelta(hours=30)

fcst_18hr_year = fcst_18hr.year
fcst_18hr_month = fcst_18hr.month
fcst_18hr_month_name = fcst_18hr.strftime("%B")
fcst_18hr_day = fcst_18hr.day
fcst_18hr_hour = fcst_18hr.strftime("%H")

fcst_30hr_year = fcst_30hr.year
fcst_30hr_month = fcst_30hr.month
fcst_30hr_month_name = fcst_30hr.strftime("%B")
fcst_30hr_day = fcst_30hr.day
fcst_30hr_hour = fcst_30hr.strftime("%H")


################################################################################################################
# ANALYSIS CHARTS (00Z)
################################################################################################################

# CHART 1: 500-mb Hemispheric Chart (Analysis)

# Create NCSS object to access the NetcdfSubset
url = 'https://thredds.ucar.edu/thredds/ncss/grid/grib/NCEP/GFS/Global_0p5deg_ana/GFS_Global_0p5deg_ana_' + str(master_date_00Z) + '.grib2'
ncss = NCSS(url)

# Create lat/lon box for location you want to get data for
query = ncss.query()
query.lonlat_box(north=90, south=0, east=360, west=0)
query.time(datetime(year_00Z, month_00Z, day_00Z, int(hour_00Z)))

# Request data for geopotential height
query.variables('Geopotential_height_isobaric', 'u-component_of_wind_isobaric',
                'v-component_of_wind_isobaric')
query.vertical_level(50000)
data_query = ncss.get_data(query)

# Pull out variables you want to use
height_var = data_query.variables['Geopotential_height_isobaric']
u_wind_var = data_query.variables['u-component_of_wind_isobaric']
v_wind_var = data_query.variables['v-component_of_wind_isobaric']

dlat = data_query.variables['Geopotential_height_isobaric'].dimensions[2]
dlon = data_query.variables['Geopotential_height_isobaric'].dimensions[3]
lat = data_query.variables[dlat][:]
lon = data_query.variables[dlon][:]

# Smooth the 500-hPa heights using a gaussian filter from scipy.ndimage
hgt_500, lon = cutil.add_cyclic_point(data_query.variables['Geopotential_height_isobaric'][:],
                                      coord=lon)
Z_500 = mpcalc.smooth_n_point(hgt_500[0, 0, :, :], 9, 50)

height_dam = Z_500 / 10

u500 = (units(data_query.variables['u-component_of_wind_isobaric'].units) *
        data_query.variables['u-component_of_wind_isobaric'][0, 0, :, :])
v500 = (units(data_query.variables['v-component_of_wind_isobaric'].units) *
        data_query.variables['v-component_of_wind_isobaric'][0, 0, :, :])
u500 = u500.units * cutil.add_cyclic_point(u500)
v500 = v500.units * cutil.add_cyclic_point(v500)
wspd500 = mpcalc.wind_speed(u500, v500).to('knots')

datacrs = ccrs.PlateCarree()
plotcrs = ccrs.NorthPolarStereo(central_longitude=-100.0)

# Make a grid of lat/lon values to use for plotting with Basemap.
lons, lats = np.meshgrid(lon, lat)

fig = plt.figure(1, figsize=(12., 13.))
gs = gridspec.GridSpec(2, 1, height_ratios=[1, .02],
                       bottom=.07, top=.99, hspace=0.01, wspace=0.01)

ax = plt.subplot(gs[0], projection=plotcrs)
ax.set_title('GFS 500-mb Geopotential Heights (dam) and Wind Barbs (kt)', loc='left')
ax.set_title('VALID: ' + str(day_00Z) + ' ' + str(month_name_00Z) + ' ' + str(year_00Z) + ' ' + str(hour_00Z) + '00 UTC', loc='right')

#   ax.set_extent([west long, east long, south lat, north lat])
ax.set_extent([-180, 180, 20, 90], ccrs.PlateCarree())
ax.add_feature(cfeature.BORDERS, edgecolor='brown', linewidth=1.5)
ax.add_feature(cfeature.COASTLINE, edgecolor='brown', linewidth=1.5)
ax.add_feature(cfeature.STATES, edgecolor='brown', linewidth=1)
#provinces = cartopy.feature.NaturalEarthFeature(category='cultural', 
#    name='admin_1_states_provinces_lines', scale='50m', facecolor='none', edgecolor='brown')
#ax.add_feature(provinces, linewidth=1, edgecolor="brown", zorder=10)

clev500 = np.arange(0, 800, 6)
cs = ax.contour(lons, lats, height_dam, clev500, colors='k',
                linewidths=1.5, linestyles='solid', transform=datacrs)
plt.clabel(cs, fontsize=10, inline=1, inline_spacing=10, fmt='%i',
           rightside_up=True, use_clabeltext=True)

clev564 = np.arange(564, 570, 6)
cs2 = ax.contour(lons, lats, height_dam, clev564, colors='k',
                linewidths=3, linestyles='solid', transform=datacrs)
plt.clabel(cs2, fontsize=10, inline=1, inline_spacing=10, fmt='%i',
           rightside_up=True, use_clabeltext=True)

wind = ax.barbs(lons, lats,
                 u500.to('kt').m, v500.to('kt').m,
                 color='darkblue', regrid_shape=20, length=6, zorder=2, transform=ccrs.PlateCarree())

# Plot 500-mb height high and low centers safely.
plot_maxmin_points(
    ax, lons, lats, height_dam,
    'max', 50,
    symbol='H',
    color='blue',
    transform=ccrs.PlateCarree(),
    symbol_size=20,
    value_size=12,
    value_offset=-3.0,
    value_offset_points=-18,
    edge_buffer=0.05
)

plot_maxmin_points(
    ax, lons, lats, height_dam,
    'min', 25,
    symbol='L',
    color='red',
    transform=ccrs.PlateCarree(),
    symbol_size=20,
    value_size=12,
    value_offset=-3.0,
    value_offset_points=-18,
    edge_buffer=0.05
)

plot_forecast_city_marker(ax)
plt.savefig('500mb_Hemispheric_Chart_GFS_00Z_Analysis.png', dpi=500, bbox_inches='tight')
plt.close()
print('Chart 1 complete')


################################################################################################################

# CHART 2: 250-mb Jet Streak Chart (Analysis)

# Create NCSS object to access the NetcdfSubset
url = 'https://thredds.ucar.edu/thredds/ncss/grid/grib/NCEP/GFS/Global_0p5deg_ana/GFS_Global_0p5deg_ana_' + str(master_date_00Z) + '.grib2'
ncss = NCSS(url)

# Create lat/lon box for location you want to get data for
query = ncss.query()
query.lonlat_box(north=90, south=0, east=-30, west=-150)
query.time(datetime(year_00Z, month_00Z, day_00Z, int(hour_00Z)))

# Request data for geopotential height
query.variables('Geopotential_height_isobaric', 'u-component_of_wind_isobaric',
                'v-component_of_wind_isobaric')
query.vertical_level(25000)
data_query = ncss.get_data(query)

# Pull out variables you want to use
height_var = data_query.variables['Geopotential_height_isobaric']
u_wind_var = data_query.variables['u-component_of_wind_isobaric']
v_wind_var = data_query.variables['v-component_of_wind_isobaric']

dlat = data_query.variables['Geopotential_height_isobaric'].dimensions[2]
dlon = data_query.variables['Geopotential_height_isobaric'].dimensions[3]
lat = data_query.variables[dlat][:]
lon = data_query.variables[dlon][:]
lons, lats = np.meshgrid(lon, lat)

# Get actual data values and remove any size 1 dimensions
height = height_var[0, 0, :, :].squeeze()
u_wind = u_wind_var[0, 0, :, :].squeeze() * units('m/s')
v_wind = v_wind_var[0, 0, :, :].squeeze() * units('m/s')
u_var = u_wind_var[0, 0, :, :].squeeze().data * units('m/s')
v_var = v_wind_var[0, 0, :, :].squeeze().data * units('m/s')
u_knots = u_var.to('knots')
v_knots = v_var.to('knots')

# Smooth height data
height = mpcalc.smooth_n_point(height, 9, 5)

# Calculate the 250-mb wind speed
u250 = (units(data_query.variables['u-component_of_wind_isobaric'].units) *
        data_query.variables['u-component_of_wind_isobaric'][0, 0, :, :])
v250 = (units(data_query.variables['v-component_of_wind_isobaric'].units) *
        data_query.variables['v-component_of_wind_isobaric'][0, 0, :, :])
wspd250 = mpcalc.wind_speed(u250, v250).to('knots')
wspd250 = mpcalc.smooth_n_point(wspd250, 9, 5)

# Create new figure
fig = plt.figure(figsize=(14, 12))

# Add the map and set the extent
ax = plt.axes(projection=ccrs.LambertConformal(central_latitude=35., central_longitude=-100., standard_parallels=(30, 60)))
ax.set_extent([-128.5, -73.2, 18.5, 58])
#ax.background_patch.set_fill(False)

# Add state boundaries to plot
ax.add_feature(cfeature.BORDERS, edgecolor='brown', linewidth=1)
ax.add_feature(cfeature.COASTLINE, edgecolor='brown', linewidth=1)
ax.add_feature(cfeature.STATES, edgecolor='brown', linewidth=1)

# Contour the heights
height_dam = height / 10
contours = np.arange(0, 1200, 12)
# Because we have a very local graphics area, the contours have joints
# to smooth those out we can use `ndimage.zoom`
#zoom_500 = ndimage.zoom(height, 5)
#zlon = ndimage.zoom(lon, 5)
#zlat = ndimage.zoom(lat, 5)
c = ax.contour(lons, lats, height_dam, levels=contours,
               colors='black', linewidths=2, transform=ccrs.PlateCarree())
labels = ax.clabel(c, fontsize=10, inline=1, inline_spacing=3, fmt='%i')
for label in labels:
    label.set_bbox(dict(facecolor='white', edgecolor='black', boxstyle='square, pad=0.2'))

# Set up parameters for quiver plot. The slices below are used to subset the data (here
# taking every 4th point in x and y). The quiver_kwargs are parameters to control the
# appearance of the quiver so that they stay consistent between the calls.
barbs_slices = (slice(None, None, 5), slice(None, None, 5))

# Plot the 250-mb wind speed as contour fill
# Create colorbar and set properties
colors = [
"#00eeee",
"#1e90ff",
"#912cee",
"#ff00ff",
"#ff0000",
"#ffff00"
]
colormap = matplotlib.colors.ListedColormap(colors)
levels = np.array([70, 90, 110, 130, 150, 200, 250])
norm = matplotlib.colors.BoundaryNorm(boundaries=levels, ncolors=6)
cf = ax.contourf(lons, lats, wspd250, levels, norm=norm, cmap=colormap, transform=ccrs.PlateCarree())
cbar = plt.colorbar(cf, ticks=levels, orientation='horizontal', pad=0.02, shrink=0.5)

# Plot the wind vectors
wind = ax.barbs(lons[barbs_slices], lats[barbs_slices],
                 u_knots[barbs_slices], v_knots[barbs_slices],
                 color='darkblue', length=6, zorder=2, transform=ccrs.PlateCarree())

# Add a title to the plot
plt.title('GFS 250-mb Geopotential Heights (dam) and Wind Speed (knots)\n'
          'VALID: ' + str(day_00Z) + ' ' + str(month_name_00Z) + ' ' + str(year_00Z) + ' ' + str(hour_00Z) + '00 UTC',
          color='black', size=14)

plot_forecast_city_marker(ax)
plt.savefig('250mb_Jet_Streak_Chart_GFS_00Z_Analysis.png', dpi=500, bbox_inches='tight')
plt.close()
print('Chart 2 complete')


################################################################################################################

# CHART 3: 200-mb Jet Streak Chart (Analysis)

# Create NCSS object to access the NetcdfSubset
url = 'https://thredds.ucar.edu/thredds/ncss/grid/grib/NCEP/GFS/Global_0p5deg_ana/GFS_Global_0p5deg_ana_' + str(master_date_00Z) + '.grib2'
ncss = NCSS(url)

# Create lat/lon box for location you want to get data for
query = ncss.query()
query.lonlat_box(north=90, south=0, east=-30, west=-150)
query.time(datetime(year_00Z, month_00Z, day_00Z, int(hour_00Z)))

# Request data for geopotential height
query.variables('Geopotential_height_isobaric', 'u-component_of_wind_isobaric',
                'v-component_of_wind_isobaric')
query.vertical_level(20000)
data_query = ncss.get_data(query)

# Pull out variables you want to use
height_var = data_query.variables['Geopotential_height_isobaric']
u_wind_var = data_query.variables['u-component_of_wind_isobaric']
v_wind_var = data_query.variables['v-component_of_wind_isobaric']

dlat = data_query.variables['Geopotential_height_isobaric'].dimensions[2]
dlon = data_query.variables['Geopotential_height_isobaric'].dimensions[3]
lat = data_query.variables[dlat][:]
lon = data_query.variables[dlon][:]
lons, lats = np.meshgrid(lon, lat)

# Get actual data values and remove any size 1 dimensions
height = height_var[0, 0, :, :].squeeze()
u_wind = u_wind_var[0, 0, :, :].squeeze() * units('m/s')
v_wind = v_wind_var[0, 0, :, :].squeeze() * units('m/s')
u_var = u_wind_var[0, 0, :, :].squeeze().data * units('m/s')
v_var = v_wind_var[0, 0, :, :].squeeze().data * units('m/s')
u_knots = u_var.to('knots')
v_knots = v_var.to('knots')

# Smooth height data
height = mpcalc.smooth_n_point(height, 9, 5)

# Calculate the 200-mb wind speed
u200 = (units(data_query.variables['u-component_of_wind_isobaric'].units) *
        data_query.variables['u-component_of_wind_isobaric'][0, 0, :, :])
v200 = (units(data_query.variables['v-component_of_wind_isobaric'].units) *
        data_query.variables['v-component_of_wind_isobaric'][0, 0, :, :])
wspd200 = mpcalc.wind_speed(u200, v200).to('knots')
wspd200 = mpcalc.smooth_n_point(wspd200, 9, 5)

# Create new figure
fig = plt.figure(figsize=(14, 12))

# Add the map and set the extent
ax = plt.axes(projection=ccrs.LambertConformal(central_latitude=35., central_longitude=-100., standard_parallels=(30, 60)))
ax.set_extent([-128.5, -73.2, 18.5, 58])
#ax.background_patch.set_fill(False)

# Add state boundaries to plot
ax.add_feature(cfeature.BORDERS, edgecolor='brown', linewidth=1)
ax.add_feature(cfeature.COASTLINE, edgecolor='brown', linewidth=1)
ax.add_feature(cfeature.STATES, edgecolor='brown', linewidth=1)

# Contour the heights
height_dam = height / 10
contours = np.arange(0, 1300, 12)
# Because we have a very local graphics area, the contours have joints
# to smooth those out we can use `ndimage.zoom`
#zoom_500 = ndimage.zoom(height, 5)
#zlon = ndimage.zoom(lon, 5)
#zlat = ndimage.zoom(lat, 5)
c = ax.contour(lons, lats, height_dam, levels=contours,
               colors='black', linewidths=2, transform=ccrs.PlateCarree())
labels = ax.clabel(c, fontsize=10, inline=1, inline_spacing=3, fmt='%i')
for label in labels:
    label.set_bbox(dict(facecolor='white', edgecolor='black', boxstyle='square, pad=0.2'))

# Set up parameters for quiver plot. The slices below are used to subset the data (here
# taking every 4th point in x and y). The quiver_kwargs are parameters to control the
# appearance of the quiver so that they stay consistent between the calls.
barbs_slices = (slice(None, None, 5), slice(None, None, 5))

# Plot the 250-mb wind speed as contour fill
# Create colorbar and set properties
colors = [
"#00eeee",
"#1e90ff",
"#912cee",
"#ff00ff",
"#ff0000",
"#ffff00"
]
colormap = matplotlib.colors.ListedColormap(colors)
levels = np.array([70, 90, 110, 130, 150, 200, 250])
norm = matplotlib.colors.BoundaryNorm(boundaries=levels, ncolors=6)
cf = ax.contourf(lons, lats, wspd200, levels, norm=norm, cmap=colormap, transform=ccrs.PlateCarree())
cbar = plt.colorbar(cf, ticks=levels, orientation='horizontal', pad=0.02, shrink=0.5)

# Plot the wind vectors
wind = ax.barbs(lons[barbs_slices], lats[barbs_slices],
                 u_knots[barbs_slices], v_knots[barbs_slices],
                 color='darkblue', length=6, zorder=2, transform=ccrs.PlateCarree())

# Add a title to the plot
plt.title('GFS 200-mb Geopotential Heights (dam) and Wind Speed (knots)\n'
          'VALID: ' + str(day_00Z) + ' ' + str(month_name_00Z) + ' ' + str(year_00Z) + ' ' + str(hour_00Z) + '00 UTC',
          color='black', size=14)

plot_forecast_city_marker(ax)
plt.savefig('200mb_Jet_Streak_Chart_GFS_00Z_Analysis.png', dpi=500, bbox_inches='tight')
plt.close()
print('Chart 3 complete')


################################################################################################################

# CHARTS 4–6: DIFAX Replicas at the 500-mb, 700-mb, and 850-mb Levels (Analysis)

# Plotting High/Low Symbols
# -------------------------
#
# A helper function to plot a text symbol (e.g., H, L) for relative
# maximum/minimum for a given field (e.g., geopotential height).
#


# Observation Data
# ----------------
#
# Set a date and time for upper-air observations (should only be 00 or 12
# UTC for the hour).
#
# Request upper-air observations from the Iowa Environmental Mesonet
# RAOB JSON service using bulk pressure-level requests.
#

# Set date for desired UPA data
# today = datetime.now(timezone.utc)
date = datetime(year_00Z, month_00Z, day_00Z, int(hour_00Z))

# Previous is 12 hours before that time
prev = date - timedelta(hours=12)

# Hybrid upper-air retrieval: IEM for U.S. stations and ECCC ObsTephi
# for Canadian stations. Only 500, 700, and 850 hPa are needed.
upper_air_stations = [
    'BMX','FFC','EYW','MFL','TBW','JAX','CHS','MHX','GSO','RNK','IAD','WAL',
    'OKX','CAR','ALB','BUF','PIT','ILN','DTX','APX','BNA','JAN','LIX','BRO',
    'FWD','MAF','OUN','DDC','TOP','LZK','SGF','ILX','DVN','GRB','MPX','INL',
    'ABR','EPZ','RIW','BOI','SLC','VEF','LKN','REV','ANT','YZT','YYE','ZXS',
    'WVK','WSE','YQD','YYQ','YPL','WMW','YAH','YVP','YZV','YYR','YQI','YJT',
    'GGW','VBG','SLE','DRT','MCU','MLP','GYM','YSM'
]

upper_air_stations_00Z = [
    'GYX','SHV','LCH','CRP','AMA','BIS','UNR','LBF','OAX','ABQ','GJT','FGZ',
    'OTX','UIL','MFR','OAK','TUS'
]

all_current_ua = get_hybrid_raob_difax_data(date)
all_previous_ua = get_hybrid_raob_difax_data(prev)


data_ua = all_current_ua[
    all_current_ua["station"].isin(upper_air_stations)
].copy()

prev_data_ua = all_previous_ua[
    all_previous_ua["station"].isin(upper_air_stations)
].copy()

# Supplemental stations are plotted at the current cycle but are not used
# for 12-hour height-change calculations.
data_ua_00Z = all_current_ua[
    all_current_ua["station"].isin(upper_air_stations_00Z)
].copy()

if data_ua.empty:
    raise RuntimeError(
        "No hybrid IEM/ECCC RAOB observations were retrieved for the 00Z DIFAX charts."
    )

print(
    f"Hybrid IEM/ECCC 00Z DIFAX retrieval complete: "
    f"{data_ua['station'].nunique()} primary stations, "
    f"{data_ua_00Z['station'].nunique()} supplemental stations."
)

# Subset Observational Data
# -------------------------
#
# From the request above will give all levels from all radiosonde sites
# available through the service. For plotting a pressure surface map there
# is only need to have the data from that level. Below the data is subset
# and a few parameters set based on the level chosen. Additionally, the
# station information is obtained and latitude and longitude data is added
# to the DataFrame.
#

level = 500

if (level == 925) | (level == 850) | (level == 700):
    cint = 30
    def hght_format(v): return format(v, '.0f')[1:]
elif level == 500:
    cint = 60
    def hght_format(v): return format(v, '.0f')[:3]
elif level == 300:
    cint = 120
    def hght_format(v): return format(v, '.0f')[:3]
elif level < 300:
    cint = 120
    def hght_format(v): return format(v, '.0f')[1:4]

# Create subset of all data for a given level
data_subset = data_ua.pressure == level
df = data_ua[data_subset]

data_subset_00Z = data_ua_00Z.pressure == level
df_00Z = data_ua_00Z[data_subset_00Z]

# Given our data, add station information and drop any stations that are missing lat/lon. Then set
# 'station' as the index column so that operations between datasets will align on that column
#data = add_station_lat_lon(data).dropna(how='any', subset=('longitude', 'latitude')).set_index('station')
#prev_data = add_station_lat_lon(prev_data).dropna(how='any', subset=('longitude', 'latitude')).set_index('station')

# Match current/previous observations explicitly by station and calculate
# the 12-hour height change only where both cycles are available.
df = prepare_difax_level(data_ua, prev_data_ua, level)


# Gridded Data
# ------------
#
# Obtain NAM gridded output for contour plotting. Specifically,
# geopotential height and temperature data for the given level and subset
# for over North America. Data are smoothed for aesthetic reasons.
#

# Create NCSS object to access the NetcdfSubset
url = 'https://thredds.ucar.edu/thredds/ncss/grid/grib/NCEP/GFS/Global_0p5deg_ana/GFS_Global_0p5deg_ana_' + str(master_date_00Z) + '.grib2'
ncss = NCSS(url)

# Create lat/lon box for location you want to get data for
query = ncss.query()
query.lonlat_box(north=90, south=0, east=-30, west=-150)
query.time(datetime(year_00Z, month_00Z, day_00Z, int(hour_00Z)))

# Request data for geopotential height
query.variables('Geopotential_height_isobaric', 'Temperature_isobaric')
query.vertical_level(50000)
data_query = ncss.get_data(query)

# Pull out variables you want to use
height_var = data_query.variables['Geopotential_height_isobaric']
temp_var = data_query.variables['Temperature_isobaric']

dlat = data_query.variables['Geopotential_height_isobaric'].dimensions[2]
dlon = data_query.variables['Geopotential_height_isobaric'].dimensions[3]
lat = data_query.variables[dlat][:]
lon = data_query.variables[dlon][:]
lons, lats = np.meshgrid(lon, lat)

height = height_var[0, 0, :, :].squeeze()
smooth_height = mpcalc.smooth_n_point(height, 9, 5)
temp = temp_var[0, 0, :, :].squeeze()
smooth_tmpk = mpcalc.smooth_n_point(temp, 9, 5)
smooth_tmpc = smooth_tmpk - 273.15


# Create DIFAX Replication
# ------------------------
#
# Plot the observational data and contours on a Lambert Conformal map and
# add features that resemble the historic DIFAX maps.
#

# Set up map coordinate reference system
mapcrs = ccrs.LambertConformal(
    central_latitude=45, central_longitude=-100, standard_parallels=(30, 60))

# Start figure and set graphics extent
fig = plt.figure(1, figsize=(17, 15))
ax = plt.subplot(111, projection=mapcrs)
ax.set_extent([-128.5, -72.5, 20.5, 58])

# Add map features for geographic reference
ax.add_feature(cfeature.COASTLINE.with_scale('50m'), edgecolor='grey', zorder=1)
ax.add_feature(cfeature.LAND.with_scale('50m'), facecolor='white', edgecolor='grey', zorder=1)
ax.add_feature(cfeature.STATES.with_scale('50m'), edgecolor='grey', zorder=1)

# Plot plus signs every degree lat/lon
plus_lat = []
plus_lon = []
other_lat = []
other_lon = []

for x in range(0,360,1):
    for y in range(0,90,1):
        if (round(x) % 5 == 0) | (round(y) % 5 == 0):
            plus_lon.append(x)
            plus_lat.append(y)
        else:
            other_lon.append(x)
            other_lat.append(y)
ax.scatter(other_lon, other_lat, s=5, marker='o',
           transform=ccrs.PlateCarree(), color='lightgrey', zorder=-1)
ax.scatter(plus_lon, plus_lat, s=30, marker='+',
           transform=ccrs.PlateCarree(), color='lightgrey', zorder=-1)

# Add gridlines for every 5 degree lat/lon
gl = ax.gridlines(linestyle='solid', ylocs=range(15, 71, 5), xlocs=range(-160, -49, 5))
gl.zorder = -1

# Start the station plot by specifying the axes to draw on, as well as the
# lon/lat of the stations (with transform). We also the fontsize to 10 pt.
stationplot = StationPlot(ax, df['longitude'].values, df['latitude'].values, clip_on=True,
                          transform=ccrs.PlateCarree(), fontsize=10)

# Plot the temperature and dew point depression to the upper and lower left, respectively, of
# the center point.
stationplot.plot_parameter('NW', df['temperature'], color='black')
stationplot.plot_parameter('SW', df['temperature'] - df['dewpoint'], color='black')

# A more complex example uses a custom formatter to control how the geopotential height
# values are plotted. This is set in an earlier if-statement to work appropriate for
# different levels.
stationplot.plot_parameter('NE', df['height'], formatter=hght_format, color='black')

# Add wind barbs
MS_TO_KT = 1.94384

stationplot.plot_barb(df['u_wind'] * MS_TO_KT, df['v_wind'] * MS_TO_KT, length=7, pivot='tip')

# Add height falls. This converts the change to decameters with /10, then
# formats with a +/- and 0 padding
def height_fall_formatter(v):
    return f'{int(v / 10):+03d}'

# Plot the parameter with an italic font
stationplot.plot_parameter('SE', df['height_change'], formatter=height_fall_formatter,
                           fontstyle='italic', color='black')

# Start the station plot by specifying the axes to draw on, as well as the
# lon/lat of the stations (with transform). We also the fontsize to 10 pt.
stationplot_00Z = StationPlot(ax, df_00Z['longitude'].values, df_00Z['latitude'].values, clip_on=True,
                          transform=ccrs.PlateCarree(), fontsize=10)

# Plot the temperature and dew point depression to the upper and lower left, respectively, of
# the center point.
stationplot_00Z.plot_parameter('NW', df_00Z['temperature'], color='black')
stationplot_00Z.plot_parameter('SW', df_00Z['temperature'] - df_00Z['dewpoint'], color='black')

# A more complex example uses a custom formatter to control how the geopotential height
# values are plotted. This is set in an earlier if-statement to work appropriate for
# different levels.
stationplot_00Z.plot_parameter('NE', df_00Z['height'], formatter=hght_format, color='black')

# Add wind barbs
stationplot_00Z.plot_barb(df_00Z['u_wind'] * MS_TO_KT, df_00Z['v_wind'] * MS_TO_KT, length=7, pivot='tip')

# Plot Solid Contours of Geopotential Height
cs = ax.contour(lons, lats, smooth_height,
                range(0, 20000, cint), colors='black', transform=ccrs.PlateCarree())
clabels = plt.clabel(cs, fmt='%d', colors='white', fontsize=8, inline_spacing=5, use_clabeltext=True)

# Contour labels with black boxes and white text
for t in clabels:
    t.set_bbox({'facecolor': 'black', 'pad': 4})
    t.set_fontweight('heavy')

# Plot Dashed Contours of Temperature
cs2 = ax.contour(lons, lats, smooth_tmpc, range(-60, 51, 3),
                 colors='#c9514b', transform=ccrs.PlateCarree())
clabels = plt.clabel(cs2, fmt='%d', colors='white', fontsize=8, inline_spacing=5, use_clabeltext=True)

# Set longer dashes than default
try:
    for c in cs2.collections:
        c.set_dashes([(0, (5.0, 3.0))])
except AttributeError:
    for c in cs2.get_children():
        c.set_dashes([(0, (5.0, 3.0))])

# Contour labels with black boxes and white text
for t in clabels:
    t.set_bbox({'facecolor': 'black', 'pad': 4})
    t.set_fontweight('heavy')

# Plot filled circles for Radiosonde Obs
ax.scatter(df['longitude'].values, df['latitude'].values, s=12,
           marker='o', color='black', transform=ccrs.PlateCarree())
ax.scatter(df_00Z['longitude'].values, df_00Z['latitude'].values, s=12,
           marker='o', color='black', transform=ccrs.PlateCarree())

# Use definition to plot H/L symbols
plot_maxmin_points(ax, lons, lats, smooth_height, 'max', 50,
                   symbol='H', color='black', transform=ccrs.PlateCarree())
plot_maxmin_points(ax, lons, lats, smooth_height, 'min', 25,
                   symbol='L', color='black', transform=ccrs.PlateCarree())

# Add titles
plt.title('GFS {}-mb Analysis of Geopotential Heights (m) and Temperature (°C)'.format(level),
          loc='left')
plt.title('VALID: ' + str(day_00Z) + ' ' + str(month_name_00Z) + ' ' + str(year_00Z) + ' ' + str(hour_00Z) + '00 UTC', loc='right')

plt.savefig('500mb_DIFAX_Chart_GFS_00Z_Analysis.png', dpi=500, bbox_inches='tight')
plt.close()
print('Chart 4 complete')


# Subset Observational Data
# -------------------------
#
# From the request above will give all levels from all radiosonde sites
# available through the service. For plotting a pressure surface map there
# is only need to have the data from that level. Below the data is subset
# and a few parameters set based on the level chosen. Additionally, the
# station information is obtained and latitude and longitude data is added
# to the DataFrame.
#

level = 700

if (level == 925) | (level == 850) | (level == 700):
    cint = 30
    def hght_format(v): return format(v, '.0f')[1:]
elif level == 500:
    cint = 60
    def hght_format(v): return format(v, '.0f')[:3]
elif level == 300:
    cint = 120
    def hght_format(v): return format(v, '.0f')[:3]
elif level < 300:
    cint = 120
    def hght_format(v): return format(v, '.0f')[1:4]

# Create subset of all data for a given level
data_subset = data_ua.pressure == level
df = data_ua[data_subset]

data_subset_00Z = data_ua_00Z.pressure == level
df_00Z = data_ua_00Z[data_subset_00Z]

# Given our data, add station information and drop any stations that are missing lat/lon. Then set
# 'station' as the index column so that operations between datasets will align on that column
#data = add_station_lat_lon(data).dropna(how='any', subset=('longitude', 'latitude')).set_index('station')
#prev_data = add_station_lat_lon(prev_data).dropna(how='any', subset=('longitude', 'latitude')).set_index('station')

# Match current/previous observations explicitly by station and calculate
# the 12-hour height change only where both cycles are available.
df = prepare_difax_level(data_ua, prev_data_ua, level)


# Gridded Data
# ------------
#
# Obtain NAM gridded output for contour plotting. Specifically,
# geopotential height and temperature data for the given level and subset
# for over North America. Data are smoothed for aesthetic reasons.
#

# Create NCSS object to access the NetcdfSubset
url = 'https://thredds.ucar.edu/thredds/ncss/grid/grib/NCEP/GFS/Global_0p5deg_ana/GFS_Global_0p5deg_ana_' + str(master_date_00Z) + '.grib2'
ncss = NCSS(url)

# Create lat/lon box for location you want to get data for
query = ncss.query()
query.lonlat_box(north=90, south=0, east=-30, west=-150)
query.time(datetime(year_00Z, month_00Z, day_00Z, int(hour_00Z)))

# Request data for geopotential height
query.variables('Geopotential_height_isobaric', 'Temperature_isobaric')
query.vertical_level(70000)
data_query = ncss.get_data(query)

# Pull out variables you want to use
height_var = data_query.variables['Geopotential_height_isobaric']
temp_var = data_query.variables['Temperature_isobaric']

dlat = data_query.variables['Geopotential_height_isobaric'].dimensions[2]
dlon = data_query.variables['Geopotential_height_isobaric'].dimensions[3]
lat = data_query.variables[dlat][:]
lon = data_query.variables[dlon][:]
lons, lats = np.meshgrid(lon, lat)

height = height_var[0, 0, :, :].squeeze()
smooth_height = mpcalc.smooth_n_point(height, 9, 5)
temp = temp_var[0, 0, :, :].squeeze()
smooth_tmpk = mpcalc.smooth_n_point(temp, 9, 5)
smooth_tmpc = smooth_tmpk - 273.15

# Create DIFAX Replication
# ------------------------
#
# Plot the observational data and contours on a Lambert Conformal map and
# add features that resemble the historic DIFAX maps.
#

# Set up map coordinate reference system
mapcrs = ccrs.LambertConformal(
    central_latitude=45, central_longitude=-100, standard_parallels=(30, 60))

# Start figure and set graphics extent
fig = plt.figure(1, figsize=(17, 15))
ax = plt.subplot(111, projection=mapcrs)
ax.set_extent([-128.5, -72.5, 20.5, 58])

# Add map features for geographic reference
ax.add_feature(cfeature.COASTLINE.with_scale('50m'), edgecolor='grey', zorder=1)
ax.add_feature(cfeature.LAND.with_scale('50m'), facecolor='white', edgecolor='grey', zorder=1)
ax.add_feature(cfeature.STATES.with_scale('50m'), edgecolor='grey', zorder=1)

# Plot plus signs every degree lat/lon
plus_lat = []
plus_lon = []
other_lat = []
other_lon = []

for x in range(0,360,1):
    for y in range(0,90,1):
        if (round(x) % 5 == 0) | (round(y) % 5 == 0):
            plus_lon.append(x)
            plus_lat.append(y)
        else:
            other_lon.append(x)
            other_lat.append(y)
ax.scatter(other_lon, other_lat, s=5, marker='o',
           transform=ccrs.PlateCarree(), color='lightgrey', zorder=-1)
ax.scatter(plus_lon, plus_lat, s=30, marker='+',
           transform=ccrs.PlateCarree(), color='lightgrey', zorder=-1)

# Add gridlines for every 5 degree lat/lon
gl = ax.gridlines(linestyle='solid', ylocs=range(15, 71, 5), xlocs=range(-160, -49, 5))
gl.zorder = -1

# Start the station plot by specifying the axes to draw on, as well as the
# lon/lat of the stations (with transform). We also the fontsize to 10 pt.
stationplot = StationPlot(ax, df['longitude'].values, df['latitude'].values, clip_on=True,
                          transform=ccrs.PlateCarree(), fontsize=10)

# Plot the temperature and dew point depression to the upper and lower left, respectively, of
# the center point.
stationplot.plot_parameter('NW', df['temperature'], color='black')
stationplot.plot_parameter('SW', df['temperature'] - df['dewpoint'], color='black')

# A more complex example uses a custom formatter to control how the geopotential height
# values are plotted. This is set in an earlier if-statement to work appropriate for
# different levels.
stationplot.plot_parameter('NE', df['height'], formatter=hght_format, color='black')

# Add wind barbs
stationplot.plot_barb(df['u_wind'] * MS_TO_KT, df['v_wind'] * MS_TO_KT, length=7, pivot='tip')

# Add height falls. This converts the change to decameters with /10, then
# formats with a +/- and 0 padding
def height_fall_formatter(v):
    return f'{int(v / 10):+03d}'

# Plot the parameter with an italic font
stationplot.plot_parameter('SE', df['height_change'], formatter=height_fall_formatter,
                           fontstyle='italic', color='black')

# Start the station plot by specifying the axes to draw on, as well as the
# lon/lat of the stations (with transform). We also the fontsize to 10 pt.
stationplot_00Z = StationPlot(ax, df_00Z['longitude'].values, df_00Z['latitude'].values, clip_on=True,
                          transform=ccrs.PlateCarree(), fontsize=10)

# Plot the temperature and dew point depression to the upper and lower left, respectively, of
# the center point.
stationplot_00Z.plot_parameter('NW', df_00Z['temperature'], color='black')
stationplot_00Z.plot_parameter('SW', df_00Z['temperature'] - df_00Z['dewpoint'], color='black')

# A more complex example uses a custom formatter to control how the geopotential height
# values are plotted. This is set in an earlier if-statement to work appropriate for
# different levels.
stationplot_00Z.plot_parameter('NE', df_00Z['height'], formatter=hght_format, color='black')

# Add wind barbs
stationplot_00Z.plot_barb(df_00Z['u_wind'] * MS_TO_KT, df_00Z['v_wind'] * MS_TO_KT, length=7, pivot='tip')

# Plot Solid Contours of Geopotential Height
cs = ax.contour(lons, lats, smooth_height,
                range(0, 20000, cint), colors='black', transform=ccrs.PlateCarree())
clabels = plt.clabel(cs, fmt='%d', colors='white', fontsize=8, inline_spacing=5, use_clabeltext=True)

# Contour labels with black boxes and white text
for t in clabels:
    t.set_bbox({'facecolor': 'black', 'pad': 4})
    t.set_fontweight('heavy')

# Plot Dashed Contours of Temperature
cs2 = ax.contour(lons, lats, smooth_tmpc, range(-60, 51, 3),
                 colors='#c9514b', transform=ccrs.PlateCarree())
clabels = plt.clabel(cs2, fmt='%d', colors='white', fontsize=8, inline_spacing=5, use_clabeltext=True)

# Set longer dashes than default
try:
    for c in cs2.collections:
        c.set_dashes([(0, (5.0, 3.0))])
except AttributeError:
    for c in cs2.get_children():
        c.set_dashes([(0, (5.0, 3.0))])

# Contour labels with black boxes and white text
for t in clabels:
    t.set_bbox({'facecolor': 'black', 'pad': 4})
    t.set_fontweight('heavy')

# Plot filled circles for Radiosonde Obs
ax.scatter(df['longitude'].values, df['latitude'].values, s=12,
           marker='o', color='black', transform=ccrs.PlateCarree())
ax.scatter(df_00Z['longitude'].values, df_00Z['latitude'].values, s=12,
           marker='o', color='black', transform=ccrs.PlateCarree())

# Use definition to plot H/L symbols
plot_maxmin_points(ax, lons, lats, smooth_height, 'max', 50,
                   symbol='H', color='black', transform=ccrs.PlateCarree())
plot_maxmin_points(ax, lons, lats, smooth_height, 'min', 25,
                   symbol='L', color='black', transform=ccrs.PlateCarree())

# Add titles
plt.title('GFS {}-mb Analysis of Geopotential Heights (m) and Temperature (°C)'.format(level),
          loc='left')
plt.title('VALID: ' + str(day_00Z) + ' ' + str(month_name_00Z) + ' ' + str(year_00Z) + ' ' + str(hour_00Z) + '00 UTC', loc='right')

plt.savefig('700mb_DIFAX_Chart_GFS_00Z_Analysis.png', dpi=500, bbox_inches='tight')
plt.close()
print('Chart 5 complete')


# Subset Observational Data
# -------------------------
#
# From the request above will give all levels from all radiosonde sites
# available through the service. For plotting a pressure surface map there
# is only need to have the data from that level. Below the data is subset
# and a few parameters set based on the level chosen. Additionally, the
# station information is obtained and latitude and longitude data is added
# to the DataFrame.
#

level = 850

if (level == 925) | (level == 850) | (level == 700):
    cint = 30
    def hght_format(v): return format(v, '.0f')[1:]
elif level == 500:
    cint = 60
    def hght_format(v): return format(v, '.0f')[:3]
elif level == 300:
    cint = 120
    def hght_format(v): return format(v, '.0f')[:3]
elif level < 300:
    cint = 120
    def hght_format(v): return format(v, '.0f')[1:4]

# Create subset of all data for a given level
data_subset = data_ua.pressure == level
df = data_ua[data_subset]

data_subset_00Z = data_ua_00Z.pressure == level
df_00Z = data_ua_00Z[data_subset_00Z]

# Given our data, add station information and drop any stations that are missing lat/lon. Then set
# 'station' as the index column so that operations between datasets will align on that column
#data = add_station_lat_lon(data).dropna(how='any', subset=('longitude', 'latitude')).set_index('station')
#prev_data = add_station_lat_lon(prev_data).dropna(how='any', subset=('longitude', 'latitude')).set_index('station')

# Match current/previous observations explicitly by station and calculate
# the 12-hour height change only where both cycles are available.
df = prepare_difax_level(data_ua, prev_data_ua, level)


# Gridded Data
# ------------
#
# Obtain NAM gridded output for contour plotting. Specifically,
# geopotential height and temperature data for the given level and subset
# for over North America. Data are smoothed for aesthetic reasons.
#

# Create NCSS object to access the NetcdfSubset
url = 'https://thredds.ucar.edu/thredds/ncss/grid/grib/NCEP/GFS/Global_0p5deg_ana/GFS_Global_0p5deg_ana_' + str(master_date_00Z) + '.grib2'
ncss = NCSS(url)

# Create lat/lon box for location you want to get data for
query = ncss.query()
query.lonlat_box(north=90, south=0, east=-30, west=-150)
query.time(datetime(year_00Z, month_00Z, day_00Z, int(hour_00Z)))

# Request data for geopotential height
query.variables('Geopotential_height_isobaric', 'Temperature_isobaric')
query.vertical_level(85000)
data_query = ncss.get_data(query)

# Pull out variables you want to use
height_var = data_query.variables['Geopotential_height_isobaric']
temp_var = data_query.variables['Temperature_isobaric']

dlat = data_query.variables['Geopotential_height_isobaric'].dimensions[2]
dlon = data_query.variables['Geopotential_height_isobaric'].dimensions[3]
lat = data_query.variables[dlat][:]
lon = data_query.variables[dlon][:]
lons, lats = np.meshgrid(lon, lat)

height = height_var[0, 0, :, :].squeeze()
smooth_height = mpcalc.smooth_n_point(height, 9, 5)
temp = temp_var[0, 0, :, :].squeeze()
smooth_tmpk = mpcalc.smooth_n_point(temp, 9, 5)
smooth_tmpc = smooth_tmpk - 273.15

# Create DIFAX Replication
# ------------------------
#
# Plot the observational data and contours on a Lambert Conformal map and
# add features that resemble the historic DIFAX maps.
#

# Set up map coordinate reference system
mapcrs = ccrs.LambertConformal(
    central_latitude=45, central_longitude=-100, standard_parallels=(30, 60))

# Start figure and set graphics extent
fig = plt.figure(1, figsize=(17, 15))
ax = plt.subplot(111, projection=mapcrs)
ax.set_extent([-128.5, -72.5, 20.5, 58])

# Add map features for geographic reference
ax.add_feature(cfeature.COASTLINE.with_scale('50m'), edgecolor='grey', zorder=1)
ax.add_feature(cfeature.LAND.with_scale('50m'), facecolor='white', edgecolor='grey', zorder=1)
ax.add_feature(cfeature.STATES.with_scale('50m'), edgecolor='grey', zorder=1)

# Plot plus signs every degree lat/lon
plus_lat = []
plus_lon = []
other_lat = []
other_lon = []

for x in range(0,360,1):
    for y in range(0,90,1):
        if (round(x) % 5 == 0) | (round(y) % 5 == 0):
            plus_lon.append(x)
            plus_lat.append(y)
        else:
            other_lon.append(x)
            other_lat.append(y)
ax.scatter(other_lon, other_lat, s=5, marker='o',
           transform=ccrs.PlateCarree(), color='lightgrey', zorder=-1)
ax.scatter(plus_lon, plus_lat, s=30, marker='+',
           transform=ccrs.PlateCarree(), color='lightgrey', zorder=-1)

# Add gridlines for every 5 degree lat/lon
gl = ax.gridlines(linestyle='solid', ylocs=range(15, 71, 5), xlocs=range(-160, -49, 5))
gl.zorder = -1

# Start the station plot by specifying the axes to draw on, as well as the
# lon/lat of the stations (with transform). We also the fontsize to 10 pt.
stationplot = StationPlot(ax, df['longitude'].values, df['latitude'].values, clip_on=True,
                          transform=ccrs.PlateCarree(), fontsize=10)

# Plot the temperature and dew point depression to the upper and lower left, respectively, of
# the center point.
stationplot.plot_parameter('NW', df['temperature'], color='black')
stationplot.plot_parameter('SW', df['temperature'] - df['dewpoint'], color='black')

# A more complex example uses a custom formatter to control how the geopotential height
# values are plotted. This is set in an earlier if-statement to work appropriate for
# different levels.
stationplot.plot_parameter('NE', df['height'], formatter=hght_format, color='black')

# Add wind barbs
stationplot.plot_barb(df['u_wind'] * MS_TO_KT, df['v_wind'] * MS_TO_KT, length=7, pivot='tip')

# Add height falls. This converts the change to decameters with /10, then
# formats with a +/- and 0 padding
def height_fall_formatter(v):
    return f'{int(v / 10):+03d}'

# Plot the parameter with an italic font
stationplot.plot_parameter('SE', df['height_change'], formatter=height_fall_formatter,
                           fontstyle='italic', color='black')

# Start the station plot by specifying the axes to draw on, as well as the
# lon/lat of the stations (with transform). We also the fontsize to 10 pt.
stationplot_00Z = StationPlot(ax, df_00Z['longitude'].values, df_00Z['latitude'].values, clip_on=True,
                          transform=ccrs.PlateCarree(), fontsize=10)

# Plot the temperature and dew point depression to the upper and lower left, respectively, of
# the center point.
stationplot_00Z.plot_parameter('NW', df_00Z['temperature'], color='black')
stationplot_00Z.plot_parameter('SW', df_00Z['temperature'] - df_00Z['dewpoint'], color='black')

# A more complex example uses a custom formatter to control how the geopotential height
# values are plotted. This is set in an earlier if-statement to work appropriate for
# different levels.
stationplot_00Z.plot_parameter('NE', df_00Z['height'], formatter=hght_format, color='black')

# Add wind barbs
stationplot_00Z.plot_barb(df_00Z['u_wind'] * MS_TO_KT, df_00Z['v_wind'] * MS_TO_KT, length=7, pivot='tip')

# Plot Solid Contours of Geopotential Height
cs = ax.contour(lons, lats, smooth_height,
                range(0, 20000, cint), colors='black', transform=ccrs.PlateCarree())
clabels = plt.clabel(cs, fmt='%d', colors='white', fontsize=8, inline_spacing=5, use_clabeltext=True)

# Contour labels with black boxes and white text
for t in clabels:
    t.set_bbox({'facecolor': 'black', 'pad': 4})
    t.set_fontweight('heavy')

# Plot Dashed Contours of Temperature
cs2 = ax.contour(lons, lats, smooth_tmpc, range(-60, 51, 3),
                 colors='#c9514b', transform=ccrs.PlateCarree())
clabels = plt.clabel(cs2, fmt='%d', colors='white', fontsize=8, inline_spacing=5, use_clabeltext=True)

# Set longer dashes than default
try:
    for c in cs2.collections:
        c.set_dashes([(0, (5.0, 3.0))])
except AttributeError:
    for c in cs2.get_children():
        c.set_dashes([(0, (5.0, 3.0))])

# Contour labels with black boxes and white text
for t in clabels:
    t.set_bbox({'facecolor': 'black', 'pad': 4})
    t.set_fontweight('heavy')

# Plot filled circles for Radiosonde Obs
ax.scatter(df['longitude'].values, df['latitude'].values, s=12,
           marker='o', color='black', transform=ccrs.PlateCarree())
ax.scatter(df_00Z['longitude'].values, df_00Z['latitude'].values, s=12,
           marker='o', color='black', transform=ccrs.PlateCarree())

# Use definition to plot H/L symbols
plot_maxmin_points(ax, lons, lats, smooth_height, 'max', 50,
                   symbol='H', color='black', transform=ccrs.PlateCarree())
plot_maxmin_points(ax, lons, lats, smooth_height, 'min', 25,
                   symbol='L', color='black', transform=ccrs.PlateCarree())

# Add titles
plt.title('GFS {}-mb Analysis of Geopotential Heights (m) and Temperature (°C)'.format(level),
          loc='left')
plt.title('VALID: ' + str(day_00Z) + ' ' + str(month_name_00Z) + ' ' + str(year_00Z) + ' ' + str(hour_00Z) + '00 UTC', loc='right')

plt.savefig('850mb_DIFAX_Chart_GFS_00Z_Analysis.png', dpi=500, bbox_inches='tight')
plt.close()
print('Chart 6 complete')


################################################################################################################

# CHART 7: 500-mb Vorticity Chart (Analysis)

# Create NCSS object to access the NetcdfSubset
url = 'https://thredds.ucar.edu/thredds/ncss/grid/grib/NCEP/GFS/Global_0p5deg_ana/GFS_Global_0p5deg_ana_' + str(master_date_00Z) + '.grib2'
ncss = NCSS(url)

# Create lat/lon box for location you want to get data for
query = ncss.query()
query.lonlat_box(north=90, south=0, east=-30, west=-150)
query.time(datetime(year_00Z, month_00Z, day_00Z, int(hour_00Z)))

# Request data for geopotential height
query.variables('Geopotential_height_isobaric', 'u-component_of_wind_isobaric',
                'v-component_of_wind_isobaric')
query.vertical_level(50000)
data_query = ncss.get_data(query)

# Pull out variables you want to use
height_var = data_query.variables['Geopotential_height_isobaric']
u_wind_var = data_query.variables['u-component_of_wind_isobaric']
v_wind_var = data_query.variables['v-component_of_wind_isobaric']

dlat = data_query.variables['Geopotential_height_isobaric'].dimensions[2]
dlon = data_query.variables['Geopotential_height_isobaric'].dimensions[3]
lat = data_query.variables[dlat][:]
lon = data_query.variables[dlon][:]
lons, lats = np.meshgrid(lon, lat)

height = height_var[0, 0, :, :].squeeze()
height = mpcalc.smooth_n_point(height, 9, 5)
u_wind = u_wind_var[0, 0, :, :].squeeze() * units('m/s')
v_wind = v_wind_var[0, 0, :, :].squeeze() * units('m/s')

# Compute north-relative wind components for plotting purposes
#uwnd_er, vwnd_er = earth_relative_wind_components(u_wind, v_wind)

# Smooth wind components as desired
uwnd_500 = mpcalc.smooth_n_point(u_wind, 9, 5) * units('m/s')
vwnd_500 = mpcalc.smooth_n_point(v_wind, 9, 5) * units('m/s')

# Calculate grid spacing that is sign aware to use in absolute vorticity calculation
dx, dy = mpcalc.lat_lon_grid_deltas(lons, lats)

# Calculate absolute vorticity from MetPy function
lats = lats.filled(np.nan)  # Replace mask with NaNs
avor_500 = mpcalc.absolute_vorticity(uwnd_500, vwnd_500, dx, dy, lats * units.degrees)

# Set up the projection that will be used for plotting
mapcrs = ccrs.LambertConformal(central_longitude=-100, central_latitude=35,
                               standard_parallels=(30, 60))

# Set up the projection of the data; if lat/lon then PlateCarree is what you want
datacrs = ccrs.PlateCarree()

# Start the figure and create plot axes with proper projection
fig = plt.figure(1, figsize=(14, 12))
ax = plt.subplot(111, projection=mapcrs)
ax.set_extent([-128.5, -72.5, 20.5, 58], ccrs.PlateCarree())

# Add geopolitical boundaries for map reference
ax.add_feature(cfeature.COASTLINE.with_scale('50m'))
ax.add_feature(cfeature.STATES.with_scale('50m'))

# Absolute Vorticity contour levels
clevs_500_avor = np.arange(-4, 34, 2)

#Colors
colors = [
"#fe01fe",
"#be00fd",
"#7e00fe",
"#0000fe",
"#0158fe",
"#028bfe",
"#02befe",
"#00fefd",
"#03e4ca",
"#01cb7d",
"#00b300",
"#7ecb05",
"#cbe506",
"#fdfd00",
"#fdcb00",
"#fe9800",
"#fd6503",
"#fd0100"
]
colormap = matplotlib.colors.ListedColormap(colors)
colormap.set_under("#cc00cc")
colormap.set_over("#cc0100")
levels = np.array([-4, -2, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32])
norm = matplotlib.colors.BoundaryNorm(boundaries=levels, ncolors=18)

# Plot absolute vorticity values (multiplying by 10^5 to scale appropriately)
cf = ax.contourf(lons, lats, avor_500*1e5, clevs_500_avor, norm=norm, cmap=colormap, extend='both',
                 transform=datacrs)
cb = plt.colorbar(cf, orientation='horizontal', pad=0.02, aspect=50)
cb.set_label('Abs. Vorticity ($x 10^{-5} s^{-1}$)')

# Plot 500-hPa Geopotential Heights in decameters
hght_500_dam = height / 10
clevs_500_hght = np.arange(0, 8000, 6)
cs = ax.contour(lons, lats, hght_500_dam, clevs_500_hght, colors='white', transform=datacrs)
plt.clabel(cs, fmt='%d')

hght_500_data = hght_500_dam.magnitude

# Plot high and low centers safely, preserving the white text with black outline.
plot_maxmin_points(
    ax, lons, lats, hght_500_data,
    'max', 50,
    symbol='H',
    color='white',
    transform=ccrs.PlateCarree(),
    symbol_size=20,
    value_size=12,
    stroke_color='black',
    stroke_width=2
)

plot_maxmin_points(
    ax, lons, lats, hght_500_data,
    'min', 25,
    symbol='L',
    color='white',
    transform=ccrs.PlateCarree(),
    symbol_size=20,
    value_size=12,
    stroke_color='black',
    stroke_width=2
)

# Plot two titles, one on right and left side
plt.title('GFS 500-mb Geopotential Heights (dam)'
          ' and Abs. Vorticity ($x 10^{-5} s^{-1}$)', loc='left')
plt.title('VALID: ' + str(day_00Z) + ' ' + str(month_name_00Z) + ' ' + str(year_00Z) + ' ' + str(hour_00Z) + '00 UTC', loc='right')

plot_forecast_city_marker(ax)
plt.savefig('500mb_Vorticity_Chart_GFS_00Z_Analysis.png', dpi=500, bbox_inches='tight')
plt.close()
print('Chart 7 complete')


################################################################################################################

# CHART 8: 850-mb Temperature Chart (Analysis)

# Create NCSS object to access the NetcdfSubset
url = 'https://thredds.ucar.edu/thredds/ncss/grid/grib/NCEP/GFS/Global_0p5deg_ana/GFS_Global_0p5deg_ana_' + str(master_date_00Z) + '.grib2'
ncss = NCSS(url)

# Create lat/lon box for location you want to get data for
query = ncss.query()
query.lonlat_box(north=90, south=0, east=-30, west=-150)
query.time(datetime(year_00Z, month_00Z, day_00Z, int(hour_00Z)))

# Request data for geopotential height
query.variables('Geopotential_height_isobaric', 'Temperature_isobaric')
query.vertical_level(85000)
data_query = ncss.get_data(query)

# Pull out variables you want to use
height_var = data_query.variables['Geopotential_height_isobaric']
temp_var = data_query.variables['Temperature_isobaric']

dlat = data_query.variables['Geopotential_height_isobaric'].dimensions[2]
dlon = data_query.variables['Geopotential_height_isobaric'].dimensions[3]
lat = data_query.variables[dlat][:]
lon = data_query.variables[dlon][:]
lons, lats = np.meshgrid(lon, lat)

height = height_var[0, 0, :, :].squeeze()
smooth_height = mpcalc.smooth_n_point(height, 9, 5)
temp = temp_var[0, 0, :, :].squeeze()
smooth_tmpk = mpcalc.smooth_n_point(temp, 9, 5)
smooth_tmpc = smooth_tmpk - 273.15

# Set up the projection that will be used for plotting
mapcrs = ccrs.LambertConformal(central_longitude=-100, central_latitude=35,
                               standard_parallels=(30, 60))

# Set up the projection of the data; if lat/lon then PlateCarree is what you want
datacrs = ccrs.PlateCarree()

# Start the figure and create plot axes with proper projection
fig = plt.figure(1, figsize=(14, 12))
ax = plt.subplot(111, projection=mapcrs)
ax.set_extent([-128.5, -72.5, 20.5, 58], ccrs.PlateCarree())

# Add geopolitical boundaries for map reference
ax.add_feature(cfeature.COASTLINE.with_scale('50m'))
ax.add_feature(cfeature.STATES.with_scale('50m'))

# Temperature contour levels
clevs_850_tmp = np.arange(-14, 32, 2)

#Colors
colors = [
"#fe01fe",
"#be00fd",
"#7e00fe",
"#0000fe",
"#0059fe",
"#008cfe",
"#03bffe",
"#00fefe",
"#00e5ca",
"#00cc7f",
"#02b202",
"#7fcb00",
"#cce505",
"#fefe00",
"#fecb02",
"#fe9800",
"#fe6600",
"#fe0000",
"#cc0100",
"#980001",
"#660000",
"#660066"
]
colormap = matplotlib.colors.ListedColormap(colors)
colormap.set_under("#cc00cc")
colormap.set_over("#980098")
levels = np.array([-14, -12, -10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30])
norm = matplotlib.colors.BoundaryNorm(boundaries=levels, ncolors=22)

# Plot 850-mb temperature values
cf = ax.contourf(lons, lats, smooth_tmpc, clevs_850_tmp, norm=norm, cmap=colormap, extend='both',
                 transform=datacrs)
cb = plt.colorbar(cf, orientation='horizontal', pad=0.02, aspect=50)
cb.set_ticks(levels[::1])
cb.set_label('Temperature (°C)')

# Plot 850-hPa Geopotential Heights in decameters
hght_850_dam = smooth_height / 10
clevs_850_hght = np.arange(0, 8000, 3)
cs = ax.contour(lons, lats, hght_850_dam, clevs_850_hght, colors='white', transform=datacrs)
plt.clabel(cs, fmt='%d')

hght_850_data = hght_850_dam.magnitude

# Plot high and low centers safely, preserving the white text with black outline.
plot_maxmin_points(
    ax, lons, lats, hght_850_data,
    'max', 50,
    symbol='H',
    color='white',
    transform=ccrs.PlateCarree(),
    symbol_size=20,
    value_size=12,
    stroke_color='black',
    stroke_width=2
)

plot_maxmin_points(
    ax, lons, lats, hght_850_data,
    'min', 25,
    symbol='L',
    color='white',
    transform=ccrs.PlateCarree(),
    symbol_size=20,
    value_size=12,
    stroke_color='black',
    stroke_width=2
)

# Plot two titles, one on right and left side
plt.title('GFS 850-mb Geopotential Heights (dam)'
          ' and Temperature (°C)', loc='left')
plt.title('VALID: ' + str(day_00Z) + ' ' + str(month_name_00Z) + ' ' + str(year_00Z) + ' ' + str(hour_00Z) + '00 UTC', loc='right')

plot_forecast_city_marker(ax)
plt.savefig('850mb_Temperature_Chart_GFS_00Z_Analysis.png', dpi=500, bbox_inches='tight')
plt.close()
print('Chart 8 complete')


################################################################################################################

# CHART 9: MSLP/500-mb Height Stack Chart (Analysis)

# Gridded Data
# ------------
#
# Obtain NAM gridded output for contour plotting. Specifically,
# geopotential height and temperature data for the given level and subset
# for over North America. Data are smoothed for aesthetic reasons.
#

# Create NCSS object to access the NetcdfSubset
url = 'https://thredds.ucar.edu/thredds/ncss/grid/grib/NCEP/GFS/Global_0p5deg_ana/GFS_Global_0p5deg_ana_' + str(master_date_00Z) + '.grib2'
ncss = NCSS(url)

# Create lat/lon box for location you want to get data for
query = ncss.query()
query.lonlat_box(north=90, south=0, east=-30, west=-150)
query.time(datetime(year_00Z, month_00Z, day_00Z, int(hour_00Z)))

# Request data for geopotential height
query.variables('Geopotential_height_isobaric')
query.vertical_level(50000)
data_query = ncss.get_data(query)

# Create lat/lon box for location you want to get data for
query_mslp = ncss.query()
query_mslp.lonlat_box(north=90, south=0, east=-30, west=-150)
query_mslp.time(datetime(year_00Z, month_00Z, day_00Z, int(hour_00Z)))

# Request data for MSLP
query_mslp.variables('Pressure_reduced_to_MSL_msl')
data_query_mslp = ncss.get_data(query_mslp)

# Pull out variables you want to use
height_var = data_query.variables['Geopotential_height_isobaric']
mslp_var = data_query_mslp.variables['Pressure_reduced_to_MSL_msl']

dlat = data_query.variables['Geopotential_height_isobaric'].dimensions[2]
dlon = data_query.variables['Geopotential_height_isobaric'].dimensions[3]
lat = data_query.variables[dlat][:]
lon = data_query.variables[dlon][:]
lons, lats = np.meshgrid(lon, lat)

height = height_var[0, 0, :, :].squeeze()
smooth_height = mpcalc.smooth_n_point(height, 9, 5)
mslp = mslp_var[0, :, :].squeeze()
smooth_mslp_Pa = mpcalc.smooth_n_point(mslp, 9, 5)
smooth_mslp_hPa = smooth_mslp_Pa / 100

# Create the Chart
# ------------
#
# Set projection of map display
mapproj = ccrs.LambertConformal(central_latitude=45., central_longitude=-100.)

# Set projection of data
dataproj = ccrs.PlateCarree()

fig = plt.figure(1, figsize=(14, 12))
ax = plt.subplot(111, projection=mapproj)
ax.set_extent([-128.5, -72.5, 20.5, 58], ccrs.PlateCarree())

# Add map features for geographic reference
ax.add_feature(cfeature.COASTLINE.with_scale('50m'), edgecolor='0.25')
ax.add_feature(cfeature.LAND.with_scale('50m'), edgecolor='0.25')
ax.add_feature(cfeature.STATES.with_scale('50m'), edgecolor='0.25')

#Colors
colors = [
"#be00fd",
"#7e00fe",
"#0000fe",
"#0158fe",
#"#028bfe",
"#02befe",
"#00fefd",
#"#03e4ca",
"#01cb7d",
"#00b300",
#"#7ecb05",
"#cbe506",
"#fdfd00",
"#fdcb00",
"#fe9800",
"#fd6503",
"#fd0100",
"#980001"
]
colormap = matplotlib.colors.ListedColormap(colors)
colormap.set_under("#fe01fe")
colormap.set_over("#660066")
levels = np.array([504, 510, 516, 522, 528, 534, 540, 546, 552, 558, 564, 570, 576, 582, 588, 594])
norm = matplotlib.colors.BoundaryNorm(boundaries=levels, ncolors=15)

# Plot 500-mb Heights
hght_500_dam = smooth_height / 10
clevs_500_hght = np.arange(504, 600, 6)
cf = ax.contourf(lons, lats, hght_500_dam, clevs_500_hght, norm=norm, cmap=colormap, extend='both', transform=dataproj)

cb = plt.colorbar(cf, orientation='horizontal', pad=0.02, aspect=50)
cb.set_ticks(levels[::1])

# Plot MSLP
clevmslp = np.arange(800., 1120., 4)
cs = ax.contour(lons, lats, smooth_mslp_hPa, clevmslp, colors='k',
                 linestyles='solid', transform=dataproj)
kw_clabels = {'fontsize': 11, 'inline': True, 'inline_spacing': 5, 'fmt': '%i',
              'rightside_up': True, 'use_clabeltext': True}
plt.clabel(cs, **kw_clabels, zorder=1)

# Use definition to plot H/L symbols
# Use smoothed MSLP directly (in hPa) to find local maxima/minima
# Remove units before processing
mslp_data = smooth_mslp_hPa.magnitude

# Plot high and low centers safely, preserving the white text with black outline.
plot_maxmin_points(
    ax, lons, lats, mslp_data,
    'max', 50,
    symbol='H',
    color='white',
    transform=ccrs.PlateCarree(),
    symbol_size=20,
    value_size=12,
    stroke_color='black',
    stroke_width=2
)

plot_maxmin_points(
    ax, lons, lats, mslp_data,
    'min', 25,
    symbol='L',
    color='white',
    transform=ccrs.PlateCarree(),
    symbol_size=20,
    value_size=12,
    stroke_color='black',
    stroke_width=2
)

# Put on some titles
plt.title('GFS MSLP (mb) and 500-mb Geopotential Heights (dam)', loc='left')
plt.title('VALID: ' + str(day_00Z) + ' ' + str(month_name_00Z) + ' ' + str(year_00Z) + ' ' + str(hour_00Z) + '00 UTC', loc='right')

plot_forecast_city_marker(ax)
plt.savefig('Stack_Chart_GFS_00Z_Analysis.png', dpi=500, bbox_inches='tight')
plt.close()
print('Chart 9 complete')


################################################################################################################

# CHART 10: MSLP/1000–500 mb Thickness Chart (Analysis)

# Gridded Data
# ------------
#
# Obtain NAM gridded output for contour plotting. Specifically,
# geopotential height and temperature data for the given level and subset
# for over North America. Data are smoothed for aesthetic reasons.
#

# Create NCSS object to access the NetcdfSubset
url = 'https://thredds.ucar.edu/thredds/ncss/grid/grib/NCEP/GFS/Global_0p5deg_ana/GFS_Global_0p5deg_ana_' + str(master_date_00Z) + '.grib2'
ncss = NCSS(url)

# Create lat/lon box for location you want to get data for
query_500hPa = ncss.query()
query_500hPa.lonlat_box(north=90, south=0, east=-30, west=-150)
query_500hPa.time(datetime(year_00Z, month_00Z, day_00Z, int(hour_00Z)))

# Request data for geopotential height
query_500hPa.variables('Geopotential_height_isobaric')
query_500hPa.vertical_level(50000)
data_query_500hPa = ncss.get_data(query_500hPa)

# Create lat/lon box for location you want to get data for
query_1000hPa = ncss.query()
query_1000hPa.lonlat_box(north=90, south=0, east=-30, west=-150)
query_1000hPa.time(datetime(year_00Z, month_00Z, day_00Z, int(hour_00Z)))

# Request data for geopotential height
query_1000hPa.variables('Geopotential_height_isobaric')
query_1000hPa.vertical_level(100000)
data_query_1000hPa = ncss.get_data(query_1000hPa)

# Create lat/lon box for location you want to get data for
query_mslp = ncss.query()
query_mslp.lonlat_box(north=90, south=0, east=-30, west=-150)
query_mslp.time(datetime(year_00Z, month_00Z, day_00Z, int(hour_00Z)))

# Request data for MSLP
query_mslp.variables('Pressure_reduced_to_MSL_msl')
data_query_mslp = ncss.get_data(query_mslp)

# Pull out variables you want to use
height_500hPa_var = data_query_500hPa.variables['Geopotential_height_isobaric']
height_1000hPa_var = data_query_1000hPa.variables['Geopotential_height_isobaric']
mslp_var = data_query_mslp.variables['Pressure_reduced_to_MSL_msl']

dlat = data_query_500hPa.variables['Geopotential_height_isobaric'].dimensions[2]
dlon = data_query_500hPa.variables['Geopotential_height_isobaric'].dimensions[3]
lat = data_query_500hPa.variables[dlat][:]
lon = data_query_500hPa.variables[dlon][:]
lons, lats = np.meshgrid(lon, lat)

height500 = height_500hPa_var[0, 0, :, :].squeeze()
smooth_height500 = mpcalc.smooth_n_point(height500, 9, 5)
height1000 = height_1000hPa_var[0, 0, :, :].squeeze()
smooth_height1000 = mpcalc.smooth_n_point(height1000, 9, 5)
mslp = mslp_var[0, :, :].squeeze()
smooth_mslp_Pa = mpcalc.smooth_n_point(mslp, 9, 5)
smooth_mslp_hPa = smooth_mslp_Pa / 100

thickness_1000_500 = smooth_height500 - smooth_height1000

# Create the Chart
# ------------
#
# Set projection of map display
mapproj = ccrs.LambertConformal(central_latitude=45., central_longitude=-100.)

# Set projection of data
dataproj = ccrs.PlateCarree()

fig = plt.figure(1, figsize=(14, 12))
ax = plt.subplot(111, projection=mapproj)
ax.set_extent([-128.5, -72.5, 20.5, 58], ccrs.PlateCarree())

# Add map features for geographic reference
ax.add_feature(cfeature.COASTLINE.with_scale('50m'), edgecolor='grey', zorder=1)
ax.add_feature(cfeature.LAND.with_scale('50m'), facecolor='white', edgecolor='grey', zorder=1)
ax.add_feature(cfeature.STATES.with_scale('50m'), edgecolor='grey', zorder=1)

# Plot 1000-500 mb thickness with multiple colors
clevs = (np.arange(0, 5400, 60),
         np.array([5400]),
         np.arange(5460, 7000, 60))
colors = ('tab:blue', 'b', 'tab:red')
kw_clabels = {'fontsize': 11, 'inline': True, 'inline_spacing': 5, 'fmt': '%i',
              'rightside_up': True, 'use_clabeltext': True}
for clevthick, color in zip(clevs, colors):
    cs = ax.contour(lons, lats, thickness_1000_500, levels=clevthick, colors=color,
                    linewidths=1.25, linestyles='dashed', transform=dataproj)
    plt.clabel(cs, **kw_clabels)

# Plot MSLP
clevmslp = np.arange(800., 1120., 4)
cs2 = ax.contour(lons, lats, smooth_mslp_hPa, clevmslp, colors='k', linewidths=1.5,
                 linestyles='solid', transform=dataproj)
plt.clabel(cs2, **kw_clabels)

# Use definition to plot H/L symbols
# Use smoothed MSLP directly (in hPa) to find local maxima/minima
# Remove units before processing

# Plot MSLP high and low centers
mslp_data = smooth_mslp_hPa.magnitude

plot_maxmin_points(
    ax, lons, lats, mslp_data,
    'max', 50,
    symbol='H',
    color='blue',
    transform=ccrs.PlateCarree()
)

plot_maxmin_points(
    ax, lons, lats, mslp_data,
    'min', 25,
    symbol='L',
    color='red',
    transform=ccrs.PlateCarree()
)

# Put on some titles
plt.title('GFS MSLP (mb) and 1000–500 mb Thickness (m)', loc='left')
plt.title('VALID: ' + str(day_00Z) + ' ' + str(month_name_00Z) + ' ' + str(year_00Z) + ' ' + str(hour_00Z) + '00 UTC', loc='right')

plot_forecast_city_marker(ax)
plt.savefig('MSLP_1000-500mb_Thickness_Chart_GFS_00Z_Analysis.png', dpi=500, bbox_inches='tight')
plt.close()
print('Chart 10 complete')


################################################################################################################
# ANALYSIS CHARTS (12Z)
################################################################################################################

# CHART 11: 250-mb Jet Streak Chart (Analysis)

# Create NCSS object to access the NetcdfSubset
url = 'https://thredds.ucar.edu/thredds/ncss/grid/grib/NCEP/GFS/Global_0p5deg_ana/GFS_Global_0p5deg_ana_' + str(master_date_12Z) + '.grib2'
ncss = NCSS(url)

# Create lat/lon box for location you want to get data for
query = ncss.query()
query.lonlat_box(north=90, south=0, east=-30, west=-150)
query.time(datetime(year_12Z, month_12Z, day_12Z, int(hour_12Z)))

# Request data for geopotential height
query.variables('Geopotential_height_isobaric', 'u-component_of_wind_isobaric',
                'v-component_of_wind_isobaric')
query.vertical_level(25000)
data_query = ncss.get_data(query)

# Pull out variables you want to use
height_var = data_query.variables['Geopotential_height_isobaric']
u_wind_var = data_query.variables['u-component_of_wind_isobaric']
v_wind_var = data_query.variables['v-component_of_wind_isobaric']

dlat = data_query.variables['Geopotential_height_isobaric'].dimensions[2]
dlon = data_query.variables['Geopotential_height_isobaric'].dimensions[3]
lat = data_query.variables[dlat][:]
lon = data_query.variables[dlon][:]
lons, lats = np.meshgrid(lon, lat)

# Get actual data values and remove any size 1 dimensions
height = height_var[0, 0, :, :].squeeze()
u_wind = u_wind_var[0, 0, :, :].squeeze() * units('m/s')
v_wind = v_wind_var[0, 0, :, :].squeeze() * units('m/s')
u_var = u_wind_var[0, 0, :, :].squeeze().data * units('m/s')
v_var = v_wind_var[0, 0, :, :].squeeze().data * units('m/s')
u_knots = u_var.to('knots')
v_knots = v_var.to('knots')

# Smooth height data
height = mpcalc.smooth_n_point(height, 9, 5)

# Calculate the 250-mb wind speed
u250 = (units(data_query.variables['u-component_of_wind_isobaric'].units) *
        data_query.variables['u-component_of_wind_isobaric'][0, 0, :, :])
v250 = (units(data_query.variables['v-component_of_wind_isobaric'].units) *
        data_query.variables['v-component_of_wind_isobaric'][0, 0, :, :])
wspd250 = mpcalc.wind_speed(u250, v250).to('knots')
wspd250 = mpcalc.smooth_n_point(wspd250, 9, 5)

# Create new figure
fig = plt.figure(figsize=(14, 12))

# Add the map and set the extent
ax = plt.axes(projection=ccrs.LambertConformal(central_latitude=35., central_longitude=-100., standard_parallels=(30, 60)))
ax.set_extent([-128.5, -73.2, 18.5, 58])
#ax.background_patch.set_fill(False)

# Add state boundaries to plot
ax.add_feature(cfeature.BORDERS, edgecolor='brown', linewidth=1)
ax.add_feature(cfeature.COASTLINE, edgecolor='brown', linewidth=1)
ax.add_feature(cfeature.STATES, edgecolor='brown', linewidth=1)

# Contour the heights
height_dam = height / 10
contours = np.arange(0, 1200, 12)
# Because we have a very local graphics area, the contours have joints
# to smooth those out we can use `ndimage.zoom`
#zoom_500 = ndimage.zoom(height, 5)
#zlon = ndimage.zoom(lon, 5)
#zlat = ndimage.zoom(lat, 5)
c = ax.contour(lons, lats, height_dam, levels=contours,
               colors='black', linewidths=2, transform=ccrs.PlateCarree())
labels = ax.clabel(c, fontsize=10, inline=1, inline_spacing=3, fmt='%i')
for label in labels:
    label.set_bbox(dict(facecolor='white', edgecolor='black', boxstyle='square, pad=0.2'))

# Set up parameters for quiver plot. The slices below are used to subset the data (here
# taking every 4th point in x and y). The quiver_kwargs are parameters to control the
# appearance of the quiver so that they stay consistent between the calls.
barbs_slices = (slice(None, None, 5), slice(None, None, 5))

# Plot the 250-mb wind speed as contour fill
# Create colorbar and set properties
colors = [
"#00eeee",
"#1e90ff",
"#912cee",
"#ff00ff",
"#ff0000",
"#ffff00"
]
colormap = matplotlib.colors.ListedColormap(colors)
levels = np.array([70, 90, 110, 130, 150, 200, 250])
norm = matplotlib.colors.BoundaryNorm(boundaries=levels, ncolors=6)
cf = ax.contourf(lons, lats, wspd250, levels, norm=norm, cmap=colormap, transform=ccrs.PlateCarree())
cbar = plt.colorbar(cf, ticks=levels, orientation='horizontal', pad=0.02, shrink=0.5)

# Plot the wind vectors
wind = ax.barbs(lons[barbs_slices], lats[barbs_slices],
                 u_knots[barbs_slices], v_knots[barbs_slices],
                 color='darkblue', length=6, zorder=2, transform=ccrs.PlateCarree())

# Add a title to the plot
plt.title('GFS 250-mb Geopotential Heights (dam) and Wind Speed (knots)\n'
          'VALID: ' + str(day_12Z) + ' ' + str(month_name_12Z) + ' ' + str(year_12Z) + ' ' + str(hour_12Z) + '00 UTC',
          color='black', size=14)

plot_forecast_city_marker(ax)
plt.savefig('250mb_Jet_Streak_Chart_GFS_12Z_Analysis.png', dpi=500, bbox_inches='tight')
plt.close()
print('Chart 11 complete')


################################################################################################################

# CHART 12: 200-mb Jet Streak Chart (Analysis)

# Create NCSS object to access the NetcdfSubset
url = 'https://thredds.ucar.edu/thredds/ncss/grid/grib/NCEP/GFS/Global_0p5deg_ana/GFS_Global_0p5deg_ana_' + str(master_date_12Z) + '.grib2'
ncss = NCSS(url)

# Create lat/lon box for location you want to get data for
query = ncss.query()
query.lonlat_box(north=90, south=0, east=-30, west=-150)
query.time(datetime(year_12Z, month_12Z, day_12Z, int(hour_12Z)))

# Request data for geopotential height
query.variables('Geopotential_height_isobaric', 'u-component_of_wind_isobaric',
                'v-component_of_wind_isobaric')
query.vertical_level(20000)
data_query = ncss.get_data(query)

# Pull out variables you want to use
height_var = data_query.variables['Geopotential_height_isobaric']
u_wind_var = data_query.variables['u-component_of_wind_isobaric']
v_wind_var = data_query.variables['v-component_of_wind_isobaric']

dlat = data_query.variables['Geopotential_height_isobaric'].dimensions[2]
dlon = data_query.variables['Geopotential_height_isobaric'].dimensions[3]
lat = data_query.variables[dlat][:]
lon = data_query.variables[dlon][:]
lons, lats = np.meshgrid(lon, lat)

# Get actual data values and remove any size 1 dimensions
height = height_var[0, 0, :, :].squeeze()
u_wind = u_wind_var[0, 0, :, :].squeeze() * units('m/s')
v_wind = v_wind_var[0, 0, :, :].squeeze() * units('m/s')
u_var = u_wind_var[0, 0, :, :].squeeze().data * units('m/s')
v_var = v_wind_var[0, 0, :, :].squeeze().data * units('m/s')
u_knots = u_var.to('knots')
v_knots = v_var.to('knots')

# Smooth height data
height = mpcalc.smooth_n_point(height, 9, 5)

# Calculate the 200-mb wind speed
u200 = (units(data_query.variables['u-component_of_wind_isobaric'].units) *
        data_query.variables['u-component_of_wind_isobaric'][0, 0, :, :])
v200 = (units(data_query.variables['v-component_of_wind_isobaric'].units) *
        data_query.variables['v-component_of_wind_isobaric'][0, 0, :, :])
wspd200 = mpcalc.wind_speed(u200, v200).to('knots')
wspd200 = mpcalc.smooth_n_point(wspd200, 9, 5)

# Create new figure
fig = plt.figure(figsize=(14, 12))

# Add the map and set the extent
ax = plt.axes(projection=ccrs.LambertConformal(central_latitude=35., central_longitude=-100., standard_parallels=(30, 60)))
ax.set_extent([-128.5, -73.2, 18.5, 58])
#ax.background_patch.set_fill(False)

# Add state boundaries to plot
ax.add_feature(cfeature.BORDERS, edgecolor='brown', linewidth=1)
ax.add_feature(cfeature.COASTLINE, edgecolor='brown', linewidth=1)
ax.add_feature(cfeature.STATES, edgecolor='brown', linewidth=1)

# Contour the heights
height_dam = height / 10
contours = np.arange(0, 1300, 12)
# Because we have a very local graphics area, the contours have joints
# to smooth those out we can use `ndimage.zoom`
#zoom_500 = ndimage.zoom(height, 5)
#zlon = ndimage.zoom(lon, 5)
#zlat = ndimage.zoom(lat, 5)
c = ax.contour(lons, lats, height_dam, levels=contours,
               colors='black', linewidths=2, transform=ccrs.PlateCarree())
labels = ax.clabel(c, fontsize=10, inline=1, inline_spacing=3, fmt='%i')
for label in labels:
    label.set_bbox(dict(facecolor='white', edgecolor='black', boxstyle='square, pad=0.2'))

# Set up parameters for quiver plot. The slices below are used to subset the data (here
# taking every 4th point in x and y). The quiver_kwargs are parameters to control the
# appearance of the quiver so that they stay consistent between the calls.
barbs_slices = (slice(None, None, 5), slice(None, None, 5))

# Plot the 250-mb wind speed as contour fill
# Create colorbar and set properties
colors = [
"#00eeee",
"#1e90ff",
"#912cee",
"#ff00ff",
"#ff0000",
"#ffff00"
]
colormap = matplotlib.colors.ListedColormap(colors)
levels = np.array([70, 90, 110, 130, 150, 200, 250])
norm = matplotlib.colors.BoundaryNorm(boundaries=levels, ncolors=6)
cf = ax.contourf(lons, lats, wspd200, levels, norm=norm, cmap=colormap, transform=ccrs.PlateCarree())
cbar = plt.colorbar(cf, ticks=levels, orientation='horizontal', pad=0.02, shrink=0.5)

# Plot the wind vectors
wind = ax.barbs(lons[barbs_slices], lats[barbs_slices],
                 u_knots[barbs_slices], v_knots[barbs_slices],
                 color='darkblue', length=6, zorder=2, transform=ccrs.PlateCarree())

# Add a title to the plot
plt.title('GFS 200-mb Geopotential Heights (dam) and Wind Speed (knots)\n'
          'VALID: ' + str(day_12Z) + ' ' + str(month_name_12Z) + ' ' + str(year_12Z) + ' ' + str(hour_12Z) + '00 UTC',
          color='black', size=14)

plot_forecast_city_marker(ax)
plt.savefig('200mb_Jet_Streak_Chart_GFS_12Z_Analysis.png', dpi=500, bbox_inches='tight')
plt.close()
print('Chart 12 complete')


################################################################################################################

# CHARTS 13–15: DIFAX Replicas at the 500-mb, 700-mb, and 850-mb Levels (Analysis)

# Plotting High/Low Symbols
# -------------------------
#
# A helper function to plot a text symbol (e.g., H, L) for relative
# maximum/minimum for a given field (e.g., geopotential height).
#


# Observation Data
# ----------------
#
# Set a date and time for upper-air observations (should only be 00 or 12
# UTC for the hour).
#
# Request upper-air observations from the Iowa Environmental Mesonet
# RAOB JSON service using bulk pressure-level requests.
#

# Set date for desired UPA data
# today = datetime.now(timezone.utc)
date = datetime(year_12Z, month_12Z, day_12Z, int(hour_12Z))

# Previous is 12 hours before that time
prev = date - timedelta(hours=12)

# Hybrid upper-air retrieval: IEM for U.S. stations and ECCC for Canada.
upper_air_stations = [
    'BMX','FFC','EYW','MFL','TBW','JAX','CHS','MHX','GSO','RNK','IAD','WAL',
    'OKX','GYX','CAR','ALB','BUF','PIT','ILN','DTX','APX','BNA','JAN','LIX',
    'SHV','LCH','BRO','CRP','FWD','MAF','AMA','OUN','DDC','TOP','LZK','SGF',
    'ILX','DVN','GRB','MPX','INL','BIS','ABR','UNR','LBF','OAX','EPZ','ABQ',
    'GJT','RIW','TFX','BOI','SLC','FGZ','VEF','LKN','REV','OTX','UIL','MFR',
    'OAK','ANT','YZT','YYE','ZXS','WVK','WSE','YQD','YYQ','YPL','WMW','YAH',
    'YVP','YZV','YYR','YQI','AWE','YJT','GGW','VBG','SLE','DRT','MCU','MLP',
    'TUS','GYM','YSM'
]

all_current_ua = get_hybrid_raob_difax_data(date)
all_previous_ua = get_hybrid_raob_difax_data(prev)


data_ua = all_current_ua[
    all_current_ua["station"].isin(upper_air_stations)
].copy()

prev_data_ua = all_previous_ua[
    all_previous_ua["station"].isin(upper_air_stations)
].copy()

if data_ua.empty:
    raise RuntimeError(
        "No hybrid IEM/ECCC RAOB observations were retrieved for the 12Z DIFAX charts."
    )

print(
    f"Hybrid IEM/ECCC 12Z DIFAX retrieval complete: "
    f"{data_ua['station'].nunique()} current stations and "
    f"{prev_data_ua['station'].nunique()} previous-cycle stations."
)

# Subset Observational Data
# -------------------------
#
# From the request above will give all levels from all radiosonde sites
# available through the service. For plotting a pressure surface map there
# is only need to have the data from that level. Below the data is subset
# and a few parameters set based on the level chosen. Additionally, the
# station information is obtained and latitude and longitude data is added
# to the DataFrame.
#

level = 500

if (level == 925) | (level == 850) | (level == 700):
    cint = 30
    def hght_format(v): return format(v, '.0f')[1:]
elif level == 500:
    cint = 60
    def hght_format(v): return format(v, '.0f')[:3]
elif level == 300:
    cint = 120
    def hght_format(v): return format(v, '.0f')[:3]
elif level < 300:
    cint = 120
    def hght_format(v): return format(v, '.0f')[1:4]

# Create subset of all data for a given level
data_subset = data_ua.pressure == level
df = data_ua[data_subset]

# Given our data, add station information and drop any stations that are missing lat/lon. Then set
# 'station' as the index column so that operations between datasets will align on that column
#data = add_station_lat_lon(data).dropna(how='any', subset=('longitude', 'latitude')).set_index('station')
#prev_data = add_station_lat_lon(prev_data).dropna(how='any', subset=('longitude', 'latitude')).set_index('station')

# Match current/previous observations explicitly by station and calculate
# the 12-hour height change only where both cycles are available.
df = prepare_difax_level(data_ua, prev_data_ua, level)


# Gridded Data
# ------------
#
# Obtain NAM gridded output for contour plotting. Specifically,
# geopotential height and temperature data for the given level and subset
# for over North America. Data are smoothed for aesthetic reasons.
#

# Create NCSS object to access the NetcdfSubset
url = 'https://thredds.ucar.edu/thredds/ncss/grid/grib/NCEP/GFS/Global_0p5deg_ana/GFS_Global_0p5deg_ana_' + str(master_date_12Z) + '.grib2'
ncss = NCSS(url)

# Create lat/lon box for location you want to get data for
query = ncss.query()
query.lonlat_box(north=90, south=0, east=-30, west=-150)
query.time(datetime(year_12Z, month_12Z, day_12Z, int(hour_12Z)))

# Request data for geopotential height
query.variables('Geopotential_height_isobaric', 'Temperature_isobaric')
query.vertical_level(50000)
data_query = ncss.get_data(query)

# Pull out variables you want to use
height_var = data_query.variables['Geopotential_height_isobaric']
temp_var = data_query.variables['Temperature_isobaric']

dlat = data_query.variables['Geopotential_height_isobaric'].dimensions[2]
dlon = data_query.variables['Geopotential_height_isobaric'].dimensions[3]
lat = data_query.variables[dlat][:]
lon = data_query.variables[dlon][:]
lons, lats = np.meshgrid(lon, lat)

height = height_var[0, 0, :, :].squeeze()
smooth_height = mpcalc.smooth_n_point(height, 9, 5)
temp = temp_var[0, 0, :, :].squeeze()
smooth_tmpk = mpcalc.smooth_n_point(temp, 9, 5)
smooth_tmpc = smooth_tmpk - 273.15

# Create DIFAX Replication
# ------------------------
#
# Plot the observational data and contours on a Lambert Conformal map and
# add features that resemble the historic DIFAX maps.
#

# Set up map coordinate reference system
mapcrs = ccrs.LambertConformal(
    central_latitude=45, central_longitude=-100, standard_parallels=(30, 60))

# Start figure and set graphics extent
fig = plt.figure(1, figsize=(17, 15))
ax = plt.subplot(111, projection=mapcrs)
ax.set_extent([-128.5, -72.5, 20.5, 58])

# Add map features for geographic reference
ax.add_feature(cfeature.COASTLINE.with_scale('50m'), edgecolor='grey', zorder=1)
ax.add_feature(cfeature.LAND.with_scale('50m'), facecolor='white', edgecolor='grey', zorder=1)
ax.add_feature(cfeature.STATES.with_scale('50m'), edgecolor='grey', zorder=1)

# Plot plus signs every degree lat/lon
plus_lat = []
plus_lon = []
other_lat = []
other_lon = []

for x in range(0,360,1):
    for y in range(0,90,1):
        if (round(x) % 5 == 0) | (round(y) % 5 == 0):
            plus_lon.append(x)
            plus_lat.append(y)
        else:
            other_lon.append(x)
            other_lat.append(y)
ax.scatter(other_lon, other_lat, s=5, marker='o',
           transform=ccrs.PlateCarree(), color='lightgrey', zorder=-1)
ax.scatter(plus_lon, plus_lat, s=30, marker='+',
           transform=ccrs.PlateCarree(), color='lightgrey', zorder=-1)

# Add gridlines for every 5 degree lat/lon
gl = ax.gridlines(linestyle='solid', ylocs=range(15, 71, 5), xlocs=range(-160, -49, 5))
gl.zorder = -1

# Start the station plot by specifying the axes to draw on, as well as the
# lon/lat of the stations (with transform). We also the fontsize to 10 pt.
stationplot = StationPlot(ax, df['longitude'].values, df['latitude'].values, clip_on=True,
                          transform=ccrs.PlateCarree(), fontsize=10)

# Plot the temperature and dew point depression to the upper and lower left, respectively, of
# the center point.
stationplot.plot_parameter('NW', df['temperature'], color='black')
stationplot.plot_parameter('SW', df['temperature'] - df['dewpoint'], color='black')

# A more complex example uses a custom formatter to control how the geopotential height
# values are plotted. This is set in an earlier if-statement to work appropriate for
# different levels.
stationplot.plot_parameter('NE', df['height'], formatter=hght_format, color='black')

# Add wind barbs
stationplot.plot_barb(df['u_wind'] * MS_TO_KT, df['v_wind'] * MS_TO_KT, length=7, pivot='tip')

# Add height falls. This converts the change to decameters with /10, then
# formats with a +/- and 0 padding
def height_fall_formatter(v):
    return f'{int(v / 10):+03d}'

# Plot the parameter with an italic font
stationplot.plot_parameter('SE', df['height_change'], formatter=height_fall_formatter,
                           fontstyle='italic', color='black')

# Plot Solid Contours of Geopotential Height
cs = ax.contour(lons, lats, smooth_height,
                range(0, 20000, cint), colors='black', transform=ccrs.PlateCarree())
clabels = plt.clabel(cs, fmt='%d', colors='white', fontsize=8, inline_spacing=5, use_clabeltext=True)

# Contour labels with black boxes and white text
for t in clabels:
    t.set_bbox({'facecolor': 'black', 'pad': 4})
    t.set_fontweight('heavy')

# Plot Dashed Contours of Temperature
cs2 = ax.contour(lons, lats, smooth_tmpc, range(-60, 51, 3),
                 colors='#c9514b', transform=ccrs.PlateCarree())
clabels = plt.clabel(cs2, fmt='%d', colors='white', fontsize=8, inline_spacing=5, use_clabeltext=True)

# Set longer dashes than default
try:
    for c in cs2.collections:
        c.set_dashes([(0, (5.0, 3.0))])
except AttributeError:
    for c in cs2.get_children():
        c.set_dashes([(0, (5.0, 3.0))])

# Contour labels with black boxes and white text
for t in clabels:
    t.set_bbox({'facecolor': 'black', 'pad': 4})
    t.set_fontweight('heavy')

# Plot filled circles for Radiosonde Obs
ax.scatter(df['longitude'].values, df['latitude'].values, s=12,
           marker='o', color='black', transform=ccrs.PlateCarree())

# Use definition to plot H/L symbols
plot_maxmin_points(ax, lons, lats, smooth_height, 'max', 50,
                   symbol='H', color='black', transform=ccrs.PlateCarree())
plot_maxmin_points(ax, lons, lats, smooth_height, 'min', 25,
                   symbol='L', color='black', transform=ccrs.PlateCarree())

# Add titles
plt.title('GFS {}-mb Analysis of Geopotential Heights (m) and Temperature (°C)'.format(level),
          loc='left')
plt.title('VALID: ' + str(day_12Z) + ' ' + str(month_name_12Z) + ' ' + str(year_12Z) + ' ' + str(hour_12Z) + '00 UTC', loc='right')

plt.savefig('500mb_DIFAX_Chart_GFS_12Z_Analysis.png', dpi=500, bbox_inches='tight')
plt.close()
print('Chart 13 complete')


# Subset Observational Data
# -------------------------
#
# From the request above will give all levels from all radiosonde sites
# available through the service. For plotting a pressure surface map there
# is only need to have the data from that level. Below the data is subset
# and a few parameters set based on the level chosen. Additionally, the
# station information is obtained and latitude and longitude data is added
# to the DataFrame.
#

level = 700

if (level == 925) | (level == 850) | (level == 700):
    cint = 30
    def hght_format(v): return format(v, '.0f')[1:]
elif level == 500:
    cint = 60
    def hght_format(v): return format(v, '.0f')[:3]
elif level == 300:
    cint = 120
    def hght_format(v): return format(v, '.0f')[:3]
elif level < 300:
    cint = 120
    def hght_format(v): return format(v, '.0f')[1:4]

# Create subset of all data for a given level
data_subset = data_ua.pressure == level
df = data_ua[data_subset]

# Given our data, add station information and drop any stations that are missing lat/lon. Then set
# 'station' as the index column so that operations between datasets will align on that column
#data = add_station_lat_lon(data).dropna(how='any', subset=('longitude', 'latitude')).set_index('station')
#prev_data = add_station_lat_lon(prev_data).dropna(how='any', subset=('longitude', 'latitude')).set_index('station')

# Match current/previous observations explicitly by station and calculate
# the 12-hour height change only where both cycles are available.
df = prepare_difax_level(data_ua, prev_data_ua, level)


# Gridded Data
# ------------
#
# Obtain NAM gridded output for contour plotting. Specifically,
# geopotential height and temperature data for the given level and subset
# for over North America. Data are smoothed for aesthetic reasons.
#

# Create NCSS object to access the NetcdfSubset
url = 'https://thredds.ucar.edu/thredds/ncss/grid/grib/NCEP/GFS/Global_0p5deg_ana/GFS_Global_0p5deg_ana_' + str(master_date_12Z) + '.grib2'
ncss = NCSS(url)

# Create lat/lon box for location you want to get data for
query = ncss.query()
query.lonlat_box(north=90, south=0, east=-30, west=-150)
query.time(datetime(year_12Z, month_12Z, day_12Z, int(hour_12Z)))

# Request data for geopotential height
query.variables('Geopotential_height_isobaric', 'Temperature_isobaric')
query.vertical_level(70000)
data_query = ncss.get_data(query)

# Pull out variables you want to use
height_var = data_query.variables['Geopotential_height_isobaric']
temp_var = data_query.variables['Temperature_isobaric']

dlat = data_query.variables['Geopotential_height_isobaric'].dimensions[2]
dlon = data_query.variables['Geopotential_height_isobaric'].dimensions[3]
lat = data_query.variables[dlat][:]
lon = data_query.variables[dlon][:]
lons, lats = np.meshgrid(lon, lat)

height = height_var[0, 0, :, :].squeeze()
smooth_height = mpcalc.smooth_n_point(height, 9, 5)
temp = temp_var[0, 0, :, :].squeeze()
smooth_tmpk = mpcalc.smooth_n_point(temp, 9, 5)
smooth_tmpc = smooth_tmpk - 273.15

# Create DIFAX Replication
# ------------------------
#
# Plot the observational data and contours on a Lambert Conformal map and
# add features that resemble the historic DIFAX maps.
#

# Set up map coordinate reference system
mapcrs = ccrs.LambertConformal(
    central_latitude=45, central_longitude=-100, standard_parallels=(30, 60))

# Start figure and set graphics extent
fig = plt.figure(1, figsize=(17, 15))
ax = plt.subplot(111, projection=mapcrs)
ax.set_extent([-128.5, -72.5, 20.5, 58])

# Add map features for geographic reference
ax.add_feature(cfeature.COASTLINE.with_scale('50m'), edgecolor='grey', zorder=1)
ax.add_feature(cfeature.LAND.with_scale('50m'), facecolor='white', edgecolor='grey', zorder=1)
ax.add_feature(cfeature.STATES.with_scale('50m'), edgecolor='grey', zorder=1)

# Plot plus signs every degree lat/lon
plus_lat = []
plus_lon = []
other_lat = []
other_lon = []

for x in range(0,360,1):
    for y in range(0,90,1):
        if (round(x) % 5 == 0) | (round(y) % 5 == 0):
            plus_lon.append(x)
            plus_lat.append(y)
        else:
            other_lon.append(x)
            other_lat.append(y)
ax.scatter(other_lon, other_lat, s=5, marker='o',
           transform=ccrs.PlateCarree(), color='lightgrey', zorder=-1)
ax.scatter(plus_lon, plus_lat, s=30, marker='+',
           transform=ccrs.PlateCarree(), color='lightgrey', zorder=-1)

# Add gridlines for every 5 degree lat/lon
gl = ax.gridlines(linestyle='solid', ylocs=range(15, 71, 5), xlocs=range(-160, -49, 5))
gl.zorder = -1

# Start the station plot by specifying the axes to draw on, as well as the
# lon/lat of the stations (with transform). We also the fontsize to 10 pt.
stationplot = StationPlot(ax, df['longitude'].values, df['latitude'].values, clip_on=True,
                          transform=ccrs.PlateCarree(), fontsize=10)

# Plot the temperature and dew point depression to the upper and lower left, respectively, of
# the center point.
stationplot.plot_parameter('NW', df['temperature'], color='black')
stationplot.plot_parameter('SW', df['temperature'] - df['dewpoint'], color='black')

# A more complex example uses a custom formatter to control how the geopotential height
# values are plotted. This is set in an earlier if-statement to work appropriate for
# different levels.
stationplot.plot_parameter('NE', df['height'], formatter=hght_format, color='black')

# Add wind barbs
stationplot.plot_barb(df['u_wind'] * MS_TO_KT, df['v_wind'] * MS_TO_KT, length=7, pivot='tip')

# Add height falls. This converts the change to decameters with /10, then
# formats with a +/- and 0 padding
def height_fall_formatter(v):
    return f'{int(v / 10):+03d}'

# Plot the parameter with an italic font
stationplot.plot_parameter('SE', df['height_change'], formatter=height_fall_formatter,
                           fontstyle='italic', color='black')

# Plot Solid Contours of Geopotential Height
cs = ax.contour(lons, lats, smooth_height,
                range(0, 20000, cint), colors='black', transform=ccrs.PlateCarree())
clabels = plt.clabel(cs, fmt='%d', colors='white', fontsize=8, inline_spacing=5, use_clabeltext=True)

# Contour labels with black boxes and white text
for t in clabels:
    t.set_bbox({'facecolor': 'black', 'pad': 4})
    t.set_fontweight('heavy')

# Plot Dashed Contours of Temperature
cs2 = ax.contour(lons, lats, smooth_tmpc, range(-60, 51, 3),
                 colors='#c9514b', transform=ccrs.PlateCarree())
clabels = plt.clabel(cs2, fmt='%d', colors='white', fontsize=8, inline_spacing=5, use_clabeltext=True)

# Set longer dashes than default
try:
    for c in cs2.collections:
        c.set_dashes([(0, (5.0, 3.0))])
except AttributeError:
    for c in cs2.get_children():
        c.set_dashes([(0, (5.0, 3.0))])

# Contour labels with black boxes and white text
for t in clabels:
    t.set_bbox({'facecolor': 'black', 'pad': 4})
    t.set_fontweight('heavy')

# Plot filled circles for Radiosonde Obs
ax.scatter(df['longitude'].values, df['latitude'].values, s=12,
           marker='o', color='black', transform=ccrs.PlateCarree())

# Use definition to plot H/L symbols
plot_maxmin_points(ax, lons, lats, smooth_height, 'max', 50,
                   symbol='H', color='black', transform=ccrs.PlateCarree())
plot_maxmin_points(ax, lons, lats, smooth_height, 'min', 25,
                   symbol='L', color='black', transform=ccrs.PlateCarree())

# Add titles
plt.title('GFS {}-mb Analysis of Geopotential Heights (m) and Temperature (°C)'.format(level),
          loc='left')
plt.title('VALID: ' + str(day_12Z) + ' ' + str(month_name_12Z) + ' ' + str(year_12Z) + ' ' + str(hour_12Z) + '00 UTC', loc='right')

plt.savefig('700mb_DIFAX_Chart_GFS_12Z_Analysis.png', dpi=500, bbox_inches='tight')
plt.close()
print('Chart 14 complete')


# Subset Observational Data
# -------------------------
#
# From the request above will give all levels from all radiosonde sites
# available through the service. For plotting a pressure surface map there
# is only need to have the data from that level. Below the data is subset
# and a few parameters set based on the level chosen. Additionally, the
# station information is obtained and latitude and longitude data is added
# to the DataFrame.
#

level = 850

if (level == 925) | (level == 850) | (level == 700):
    cint = 30
    def hght_format(v): return format(v, '.0f')[1:]
elif level == 500:
    cint = 60
    def hght_format(v): return format(v, '.0f')[:3]
elif level == 300:
    cint = 120
    def hght_format(v): return format(v, '.0f')[:3]
elif level < 300:
    cint = 120
    def hght_format(v): return format(v, '.0f')[1:4]

# Create subset of all data for a given level
data_subset = data_ua.pressure == level
df = data_ua[data_subset]

# Given our data, add station information and drop any stations that are missing lat/lon. Then set
# 'station' as the index column so that operations between datasets will align on that column
#data = add_station_lat_lon(data).dropna(how='any', subset=('longitude', 'latitude')).set_index('station')
#prev_data = add_station_lat_lon(prev_data).dropna(how='any', subset=('longitude', 'latitude')).set_index('station')

# Match current/previous observations explicitly by station and calculate
# the 12-hour height change only where both cycles are available.
df = prepare_difax_level(data_ua, prev_data_ua, level)


# Gridded Data
# ------------
#
# Obtain NAM gridded output for contour plotting. Specifically,
# geopotential height and temperature data for the given level and subset
# for over North America. Data are smoothed for aesthetic reasons.
#

# Create NCSS object to access the NetcdfSubset
url = 'https://thredds.ucar.edu/thredds/ncss/grid/grib/NCEP/GFS/Global_0p5deg_ana/GFS_Global_0p5deg_ana_' + str(master_date_12Z) + '.grib2'
ncss = NCSS(url)

# Create lat/lon box for location you want to get data for
query = ncss.query()
query.lonlat_box(north=90, south=0, east=-30, west=-150)
query.time(datetime(year_12Z, month_12Z, day_12Z, int(hour_12Z)))

# Request data for geopotential height
query.variables('Geopotential_height_isobaric', 'Temperature_isobaric')
query.vertical_level(85000)
data_query = ncss.get_data(query)

# Pull out variables you want to use
height_var = data_query.variables['Geopotential_height_isobaric']
temp_var = data_query.variables['Temperature_isobaric']

dlat = data_query.variables['Geopotential_height_isobaric'].dimensions[2]
dlon = data_query.variables['Geopotential_height_isobaric'].dimensions[3]
lat = data_query.variables[dlat][:]
lon = data_query.variables[dlon][:]
lons, lats = np.meshgrid(lon, lat)

height = height_var[0, 0, :, :].squeeze()
smooth_height = mpcalc.smooth_n_point(height, 9, 5)
temp = temp_var[0, 0, :, :].squeeze()
smooth_tmpk = mpcalc.smooth_n_point(temp, 9, 5)
smooth_tmpc = smooth_tmpk - 273.15

# Create DIFAX Replication
# ------------------------
#
# Plot the observational data and contours on a Lambert Conformal map and
# add features that resemble the historic DIFAX maps.
#

# Set up map coordinate reference system
mapcrs = ccrs.LambertConformal(
    central_latitude=45, central_longitude=-100, standard_parallels=(30, 60))

# Start figure and set graphics extent
fig = plt.figure(1, figsize=(17, 15))
ax = plt.subplot(111, projection=mapcrs)
ax.set_extent([-128.5, -72.5, 20.5, 58])

# Add map features for geographic reference
ax.add_feature(cfeature.COASTLINE.with_scale('50m'), edgecolor='grey', zorder=1)
ax.add_feature(cfeature.LAND.with_scale('50m'), facecolor='white', edgecolor='grey', zorder=1)
ax.add_feature(cfeature.STATES.with_scale('50m'), edgecolor='grey', zorder=1)

# Plot plus signs every degree lat/lon
plus_lat = []
plus_lon = []
other_lat = []
other_lon = []

for x in range(0,360,1):
    for y in range(0,90,1):
        if (round(x) % 5 == 0) | (round(y) % 5 == 0):
            plus_lon.append(x)
            plus_lat.append(y)
        else:
            other_lon.append(x)
            other_lat.append(y)
ax.scatter(other_lon, other_lat, s=5, marker='o',
           transform=ccrs.PlateCarree(), color='lightgrey', zorder=-1)
ax.scatter(plus_lon, plus_lat, s=30, marker='+',
           transform=ccrs.PlateCarree(), color='lightgrey', zorder=-1)

# Add gridlines for every 5 degree lat/lon
gl = ax.gridlines(linestyle='solid', ylocs=range(15, 71, 5), xlocs=range(-160, -49, 5))
gl.zorder = -1

# Start the station plot by specifying the axes to draw on, as well as the
# lon/lat of the stations (with transform). We also the fontsize to 10 pt.
stationplot = StationPlot(ax, df['longitude'].values, df['latitude'].values, clip_on=True,
                          transform=ccrs.PlateCarree(), fontsize=10)

# Plot the temperature and dew point depression to the upper and lower left, respectively, of
# the center point.
stationplot.plot_parameter('NW', df['temperature'], color='black')
stationplot.plot_parameter('SW', df['temperature'] - df['dewpoint'], color='black')

# A more complex example uses a custom formatter to control how the geopotential height
# values are plotted. This is set in an earlier if-statement to work appropriate for
# different levels.
stationplot.plot_parameter('NE', df['height'], formatter=hght_format, color='black')

# Add wind barbs
stationplot.plot_barb(df['u_wind'] * MS_TO_KT, df['v_wind'] * MS_TO_KT, length=7, pivot='tip')

# Add height falls. This converts the change to decameters with /10, then
# formats with a +/- and 0 padding
def height_fall_formatter(v):
    return f'{int(v / 10):+03d}'

# Plot the parameter with an italic font
stationplot.plot_parameter('SE', df['height_change'], formatter=height_fall_formatter,
                           fontstyle='italic', color='black')

# Plot Solid Contours of Geopotential Height
cs = ax.contour(lons, lats, smooth_height,
                range(0, 20000, cint), colors='black', transform=ccrs.PlateCarree())
clabels = plt.clabel(cs, fmt='%d', colors='white', fontsize=8, inline_spacing=5, use_clabeltext=True)

# Contour labels with black boxes and white text
for t in clabels:
    t.set_bbox({'facecolor': 'black', 'pad': 4})
    t.set_fontweight('heavy')

# Plot Dashed Contours of Temperature
cs2 = ax.contour(lons, lats, smooth_tmpc, range(-60, 51, 3),
                 colors='#c9514b', transform=ccrs.PlateCarree())
clabels = plt.clabel(cs2, fmt='%d', colors='white', fontsize=8, inline_spacing=5, use_clabeltext=True)

# Set longer dashes than default
try:
    for c in cs2.collections:
        c.set_dashes([(0, (5.0, 3.0))])
except AttributeError:
    for c in cs2.get_children():
        c.set_dashes([(0, (5.0, 3.0))])

# Contour labels with black boxes and white text
for t in clabels:
    t.set_bbox({'facecolor': 'black', 'pad': 4})
    t.set_fontweight('heavy')

# Plot filled circles for Radiosonde Obs
ax.scatter(df['longitude'].values, df['latitude'].values, s=12,
           marker='o', color='black', transform=ccrs.PlateCarree())

# Use definition to plot H/L symbols
plot_maxmin_points(ax, lons, lats, smooth_height, 'max', 50,
                   symbol='H', color='black', transform=ccrs.PlateCarree())
plot_maxmin_points(ax, lons, lats, smooth_height, 'min', 25,
                   symbol='L', color='black', transform=ccrs.PlateCarree())

# Add titles
plt.title('GFS {}-mb Analysis of Geopotential Heights (m) and Temperature (°C)'.format(level),
          loc='left')
plt.title('VALID: ' + str(day_12Z) + ' ' + str(month_name_12Z) + ' ' + str(year_12Z) + ' ' + str(hour_12Z) + '00 UTC', loc='right')

plt.savefig('850mb_DIFAX_Chart_GFS_12Z_Analysis.png', dpi=500, bbox_inches='tight')
plt.close()
print('Chart 15 complete')


################################################################################################################

# CHART 16: 500-mb Vorticity Chart (Analysis)

# Create NCSS object to access the NetcdfSubset
url = 'https://thredds.ucar.edu/thredds/ncss/grid/grib/NCEP/GFS/Global_0p5deg_ana/GFS_Global_0p5deg_ana_' + str(master_date_12Z) + '.grib2'
ncss = NCSS(url)

# Create lat/lon box for location you want to get data for
query = ncss.query()
query.lonlat_box(north=90, south=0, east=-30, west=-150)
query.time(datetime(year_12Z, month_12Z, day_12Z, int(hour_12Z)))

# Request data for geopotential height
query.variables('Geopotential_height_isobaric', 'u-component_of_wind_isobaric',
                'v-component_of_wind_isobaric')
query.vertical_level(50000)
data_query = ncss.get_data(query)

# Pull out variables you want to use
height_var = data_query.variables['Geopotential_height_isobaric']
u_wind_var = data_query.variables['u-component_of_wind_isobaric']
v_wind_var = data_query.variables['v-component_of_wind_isobaric']

dlat = data_query.variables['Geopotential_height_isobaric'].dimensions[2]
dlon = data_query.variables['Geopotential_height_isobaric'].dimensions[3]
lat = data_query.variables[dlat][:]
lon = data_query.variables[dlon][:]
lons, lats = np.meshgrid(lon, lat)

height = height_var[0, 0, :, :].squeeze()
height = mpcalc.smooth_n_point(height, 9, 5)
u_wind = u_wind_var[0, 0, :, :].squeeze() * units('m/s')
v_wind = v_wind_var[0, 0, :, :].squeeze() * units('m/s')

# Compute north-relative wind components for plotting purposes
#uwnd_er, vwnd_er = earth_relative_wind_components(u_wind, v_wind)

# Smooth wind components as desired
uwnd_500 = mpcalc.smooth_n_point(u_wind, 9, 5) * units('m/s')
vwnd_500 = mpcalc.smooth_n_point(v_wind, 9, 5) * units('m/s')

# Calculate grid spacing that is sign aware to use in absolute vorticity calculation
dx, dy = mpcalc.lat_lon_grid_deltas(lons, lats)

# Calculate absolute vorticity from MetPy function
lats = lats.filled(np.nan)  # Replace mask with NaNs
avor_500 = mpcalc.absolute_vorticity(uwnd_500, vwnd_500, dx, dy, lats * units.degrees)

# Set up the projection that will be used for plotting
mapcrs = ccrs.LambertConformal(central_longitude=-100, central_latitude=35,
                               standard_parallels=(30, 60))

# Set up the projection of the data; if lat/lon then PlateCarree is what you want
datacrs = ccrs.PlateCarree()

# Start the figure and create plot axes with proper projection
fig = plt.figure(1, figsize=(14, 12))
ax = plt.subplot(111, projection=mapcrs)
ax.set_extent([-128.5, -72.5, 20.5, 58], ccrs.PlateCarree())

# Add geopolitical boundaries for map reference
ax.add_feature(cfeature.COASTLINE.with_scale('50m'))
ax.add_feature(cfeature.STATES.with_scale('50m'))

# Absolute Vorticity contour levels
clevs_500_avor = np.arange(-4, 34, 2)

#Colors
colors = [
"#fe01fe",
"#be00fd",
"#7e00fe",
"#0000fe",
"#0158fe",
"#028bfe",
"#02befe",
"#00fefd",
"#03e4ca",
"#01cb7d",
"#00b300",
"#7ecb05",
"#cbe506",
"#fdfd00",
"#fdcb00",
"#fe9800",
"#fd6503",
"#fd0100"
]
colormap = matplotlib.colors.ListedColormap(colors)
colormap.set_under("#cc00cc")
colormap.set_over("#cc0100")
levels = np.array([-4, -2, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32])
norm = matplotlib.colors.BoundaryNorm(boundaries=levels, ncolors=18)

# Plot absolute vorticity values (multiplying by 10^5 to scale appropriately)
cf = ax.contourf(lons, lats, avor_500*1e5, clevs_500_avor, norm=norm, cmap=colormap, extend='both',
                 transform=datacrs)
cb = plt.colorbar(cf, orientation='horizontal', pad=0.02, aspect=50)
cb.set_label('Abs. Vorticity ($x 10^{-5} s^{-1}$)')

# Plot 500-hPa Geopotential Heights in decameters
hght_500_dam = height / 10
clevs_500_hght = np.arange(0, 8000, 6)
cs = ax.contour(lons, lats, hght_500_dam, clevs_500_hght, colors='white', transform=datacrs)
plt.clabel(cs, fmt='%d')

hght_500_data = hght_500_dam.magnitude

# Plot high and low centers safely, preserving the white text with black outline.
plot_maxmin_points(
    ax, lons, lats, hght_500_data,
    'max', 50,
    symbol='H',
    color='white',
    transform=ccrs.PlateCarree(),
    symbol_size=20,
    value_size=12,
    stroke_color='black',
    stroke_width=2
)

plot_maxmin_points(
    ax, lons, lats, hght_500_data,
    'min', 25,
    symbol='L',
    color='white',
    transform=ccrs.PlateCarree(),
    symbol_size=20,
    value_size=12,
    stroke_color='black',
    stroke_width=2
)

# Plot two titles, one on right and left side
plt.title('GFS 500-mb Geopotential Heights (dam)'
          ' and Abs. Vorticity ($x 10^{-5} s^{-1}$)', loc='left')
plt.title('VALID: ' + str(day_12Z) + ' ' + str(month_name_12Z) + ' ' + str(year_12Z) + ' ' + str(hour_12Z) + '00 UTC', loc='right')

plot_forecast_city_marker(ax)
plt.savefig('500mb_Vorticity_Chart_GFS_12Z_Analysis.png', dpi=500, bbox_inches='tight')
plt.close()
print('Chart 16 complete')


################################################################################################################

# CHART 17: 850-mb Temperature Chart (Analysis)

# Create NCSS object to access the NetcdfSubset
url = 'https://thredds.ucar.edu/thredds/ncss/grid/grib/NCEP/GFS/Global_0p5deg_ana/GFS_Global_0p5deg_ana_' + str(master_date_12Z) + '.grib2'
ncss = NCSS(url)

# Create lat/lon box for location you want to get data for
query = ncss.query()
query.lonlat_box(north=90, south=0, east=-30, west=-150)
query.time(datetime(year_12Z, month_12Z, day_12Z, int(hour_12Z)))

# Request data for geopotential height
query.variables('Geopotential_height_isobaric', 'Temperature_isobaric')
query.vertical_level(85000)
data_query = ncss.get_data(query)

# Pull out variables you want to use
height_var = data_query.variables['Geopotential_height_isobaric']
temp_var = data_query.variables['Temperature_isobaric']

dlat = data_query.variables['Geopotential_height_isobaric'].dimensions[2]
dlon = data_query.variables['Geopotential_height_isobaric'].dimensions[3]
lat = data_query.variables[dlat][:]
lon = data_query.variables[dlon][:]
lons, lats = np.meshgrid(lon, lat)

height = height_var[0, 0, :, :].squeeze()
smooth_height = mpcalc.smooth_n_point(height, 9, 5)
temp = temp_var[0, 0, :, :].squeeze()
smooth_tmpk = mpcalc.smooth_n_point(temp, 9, 5)
smooth_tmpc = smooth_tmpk - 273.15

# Set up the projection that will be used for plotting
mapcrs = ccrs.LambertConformal(central_longitude=-100, central_latitude=35,
                               standard_parallels=(30, 60))

# Set up the projection of the data; if lat/lon then PlateCarree is what you want
datacrs = ccrs.PlateCarree()

# Start the figure and create plot axes with proper projection
fig = plt.figure(1, figsize=(14, 12))
ax = plt.subplot(111, projection=mapcrs)
ax.set_extent([-128.5, -72.5, 20.5, 58], ccrs.PlateCarree())

# Add geopolitical boundaries for map reference
ax.add_feature(cfeature.COASTLINE.with_scale('50m'))
ax.add_feature(cfeature.STATES.with_scale('50m'))

# Temperature contour levels
clevs_850_tmp = np.arange(-14, 32, 2)

#Colors
colors = [
"#fe01fe",
"#be00fd",
"#7e00fe",
"#0000fe",
"#0059fe",
"#008cfe",
"#03bffe",
"#00fefe",
"#00e5ca",
"#00cc7f",
"#02b202",
"#7fcb00",
"#cce505",
"#fefe00",
"#fecb02",
"#fe9800",
"#fe6600",
"#fe0000",
"#cc0100",
"#980001",
"#660000",
"#660066"
]
colormap = matplotlib.colors.ListedColormap(colors)
colormap.set_under("#cc00cc")
colormap.set_over("#980098")
levels = np.array([-14, -12, -10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30])
norm = matplotlib.colors.BoundaryNorm(boundaries=levels, ncolors=22)

# Plot 850-mb temperature values
cf = ax.contourf(lons, lats, smooth_tmpc, clevs_850_tmp, norm=norm, cmap=colormap, extend='both',
                 transform=datacrs)
cb = plt.colorbar(cf, orientation='horizontal', pad=0.02, aspect=50)
cb.set_ticks(levels[::1])
cb.set_label('Temperature (°C)')

# Plot 850-hPa Geopotential Heights in decameters
hght_850_dam = smooth_height / 10
clevs_850_hght = np.arange(0, 8000, 3)
cs = ax.contour(lons, lats, hght_850_dam, clevs_850_hght, colors='white', linewidths=1.5, transform=datacrs)
plt.clabel(cs, fmt='%d')

hght_850_data = hght_850_dam.magnitude

# Plot high and low centers safely, preserving the white text with black outline.
plot_maxmin_points(
    ax, lons, lats, hght_850_data,
    'max', 50,
    symbol='H',
    color='white',
    transform=ccrs.PlateCarree(),
    symbol_size=20,
    value_size=12,
    stroke_color='black',
    stroke_width=2
)

plot_maxmin_points(
    ax, lons, lats, hght_850_data,
    'min', 25,
    symbol='L',
    color='white',
    transform=ccrs.PlateCarree(),
    symbol_size=20,
    value_size=12,
    stroke_color='black',
    stroke_width=2
)

# Plot two titles, one on right and left side
plt.title('GFS 850-mb Geopotential Heights (dam)'
          ' and Temperature (°C)', loc='left')
plt.title('VALID: ' + str(day_12Z) + ' ' + str(month_name_12Z) + ' ' + str(year_12Z) + ' ' + str(hour_12Z) + '00 UTC', loc='right')

plot_forecast_city_marker(ax)
plt.savefig('850mb_Temperature_Chart_GFS_12Z_Analysis.png', dpi=500, bbox_inches='tight')
plt.close()
print('Chart 17 complete')


################################################################################################################

# CHART 18: MSLP/500-mb Height Stack Chart (Analysis)

# Gridded Data
# ------------
#
# Obtain NAM gridded output for contour plotting. Specifically,
# geopotential height and temperature data for the given level and subset
# for over North America. Data are smoothed for aesthetic reasons.
#

# Create NCSS object to access the NetcdfSubset
url = 'https://thredds.ucar.edu/thredds/ncss/grid/grib/NCEP/GFS/Global_0p5deg_ana/GFS_Global_0p5deg_ana_' + str(master_date_12Z) + '.grib2'
ncss = NCSS(url)

# Create lat/lon box for location you want to get data for
query = ncss.query()
query.lonlat_box(north=90, south=0, east=-30, west=-150)
query.time(datetime(year_12Z, month_12Z, day_12Z, int(hour_12Z)))

# Request data for geopotential height
query.variables('Geopotential_height_isobaric')
query.vertical_level(50000)
data_query = ncss.get_data(query)

# Create lat/lon box for location you want to get data for
query_mslp = ncss.query()
query_mslp.lonlat_box(north=90, south=0, east=-30, west=-150)
query_mslp.time(datetime(year_12Z, month_12Z, day_12Z, int(hour_12Z)))

# Request data for MSLP
query_mslp.variables('Pressure_reduced_to_MSL_msl')
data_query_mslp = ncss.get_data(query_mslp)

# Pull out variables you want to use
height_var = data_query.variables['Geopotential_height_isobaric']
mslp_var = data_query_mslp.variables['Pressure_reduced_to_MSL_msl']

dlat = data_query.variables['Geopotential_height_isobaric'].dimensions[2]
dlon = data_query.variables['Geopotential_height_isobaric'].dimensions[3]
lat = data_query.variables[dlat][:]
lon = data_query.variables[dlon][:]
lons, lats = np.meshgrid(lon, lat)

height = height_var[0, 0, :, :].squeeze()
smooth_height = mpcalc.smooth_n_point(height, 9, 5)
mslp = mslp_var[0, :, :].squeeze()
smooth_mslp_Pa = mpcalc.smooth_n_point(mslp, 9, 5)
smooth_mslp_hPa = smooth_mslp_Pa / 100

# Create the Chart
# ------------
#
# Set projection of map display
mapproj = ccrs.LambertConformal(central_latitude=45., central_longitude=-100.)

# Set projection of data
dataproj = ccrs.PlateCarree()

fig = plt.figure(1, figsize=(14, 12))
ax = plt.subplot(111, projection=mapproj)
ax.set_extent([-128.5, -72.5, 20.5, 58], ccrs.PlateCarree())

# Add map features for geographic reference
ax.add_feature(cfeature.COASTLINE.with_scale('50m'), edgecolor='0.25')
ax.add_feature(cfeature.LAND.with_scale('50m'), edgecolor='0.25')
ax.add_feature(cfeature.STATES.with_scale('50m'), edgecolor='0.25')

#Colors
colors = [
"#be00fd",
"#7e00fe",
"#0000fe",
"#0158fe",
#"#028bfe",
"#02befe",
"#00fefd",
#"#03e4ca",
"#01cb7d",
"#00b300",
#"#7ecb05",
"#cbe506",
"#fdfd00",
"#fdcb00",
"#fe9800",
"#fd6503",
"#fd0100",
"#980001"
]
colormap = matplotlib.colors.ListedColormap(colors)
colormap.set_under("#fe01fe")
colormap.set_over("#660066")
levels = np.array([504, 510, 516, 522, 528, 534, 540, 546, 552, 558, 564, 570, 576, 582, 588, 594])
norm = matplotlib.colors.BoundaryNorm(boundaries=levels, ncolors=15)

# Plot 500-mb Heights
hght_500_dam = smooth_height / 10
clevs_500_hght = np.arange(504, 600, 6)
cf = ax.contourf(lons, lats, hght_500_dam, clevs_500_hght, norm=norm, cmap=colormap, extend='both', transform=dataproj)

cb = plt.colorbar(cf, orientation='horizontal', pad=0.02, aspect=50)
cb.set_ticks(levels[::1])

# Plot MSLP
clevmslp = np.arange(800., 1120., 4)
cs = ax.contour(lons, lats, smooth_mslp_hPa, clevmslp, colors='k', linewidths=1.5,
                 linestyles='solid', transform=dataproj)
kw_clabels = {'fontsize': 11, 'inline': True, 'inline_spacing': 5, 'fmt': '%i',
              'rightside_up': True, 'use_clabeltext': True}
plt.clabel(cs, **kw_clabels, zorder=1)

# Use definition to plot H/L symbols
# Use smoothed MSLP directly (in hPa) to find local maxima/minima
# Remove units before processing
mslp_data = smooth_mslp_hPa.magnitude

# Plot high and low centers safely, preserving the white text with black outline.
plot_maxmin_points(
    ax, lons, lats, mslp_data,
    'max', 50,
    symbol='H',
    color='white',
    transform=ccrs.PlateCarree(),
    symbol_size=20,
    value_size=12,
    stroke_color='black',
    stroke_width=2
)

plot_maxmin_points(
    ax, lons, lats, mslp_data,
    'min', 25,
    symbol='L',
    color='white',
    transform=ccrs.PlateCarree(),
    symbol_size=20,
    value_size=12,
    stroke_color='black',
    stroke_width=2
)

# Put on some titles
plt.title('GFS MSLP (mb) and 500-mb Geopotential Heights (dam)', loc='left')
plt.title('VALID: ' + str(day_12Z) + ' ' + str(month_name_12Z) + ' ' + str(year_12Z) + ' ' + str(hour_12Z) + '00 UTC', loc='right')

plot_forecast_city_marker(ax)
plt.savefig('Stack_Chart_GFS_12Z_Analysis.png', dpi=500, bbox_inches='tight')
plt.close()
print('Chart 18 complete')


################################################################################################################

# CHART 19: MSLP/1000–500 mb Thickness Chart (Analysis)

# Gridded Data
# ------------
#
# Obtain NAM gridded output for contour plotting. Specifically,
# geopotential height and temperature data for the given level and subset
# for over North America. Data are smoothed for aesthetic reasons.
#

# Create NCSS object to access the NetcdfSubset
url = 'https://thredds.ucar.edu/thredds/ncss/grid/grib/NCEP/GFS/Global_0p5deg_ana/GFS_Global_0p5deg_ana_' + str(master_date_12Z) + '.grib2'
ncss = NCSS(url)

# Create lat/lon box for location you want to get data for
query_500hPa = ncss.query()
query_500hPa.lonlat_box(north=90, south=0, east=-30, west=-150)
query_500hPa.time(datetime(year_12Z, month_12Z, day_12Z, int(hour_12Z)))

# Request data for geopotential height
query_500hPa.variables('Geopotential_height_isobaric')
query_500hPa.vertical_level(50000)
data_query_500hPa = ncss.get_data(query_500hPa)

# Create lat/lon box for location you want to get data for
query_1000hPa = ncss.query()
query_1000hPa.lonlat_box(north=90, south=0, east=-30, west=-150)
query_1000hPa.time(datetime(year_12Z, month_12Z, day_12Z, int(hour_12Z)))

# Request data for geopotential height
query_1000hPa.variables('Geopotential_height_isobaric')
query_1000hPa.vertical_level(100000)
data_query_1000hPa = ncss.get_data(query_1000hPa)

# Create lat/lon box for location you want to get data for
query_mslp = ncss.query()
query_mslp.lonlat_box(north=90, south=0, east=-30, west=-150)
query_mslp.time(datetime(year_12Z, month_12Z, day_12Z, int(hour_12Z)))

# Request data for MSLP
query_mslp.variables('Pressure_reduced_to_MSL_msl')
data_query_mslp = ncss.get_data(query_mslp)

# Pull out variables you want to use
height_500hPa_var = data_query_500hPa.variables['Geopotential_height_isobaric']
height_1000hPa_var = data_query_1000hPa.variables['Geopotential_height_isobaric']
mslp_var = data_query_mslp.variables['Pressure_reduced_to_MSL_msl']

dlat = data_query_500hPa.variables['Geopotential_height_isobaric'].dimensions[2]
dlon = data_query_500hPa.variables['Geopotential_height_isobaric'].dimensions[3]
lat = data_query_500hPa.variables[dlat][:]
lon = data_query_500hPa.variables[dlon][:]
lons, lats = np.meshgrid(lon, lat)

height500 = height_500hPa_var[0, 0, :, :].squeeze()
smooth_height500 = mpcalc.smooth_n_point(height500, 9, 5)
height1000 = height_1000hPa_var[0, 0, :, :].squeeze()
smooth_height1000 = mpcalc.smooth_n_point(height1000, 9, 5)
mslp = mslp_var[0, :, :].squeeze()
smooth_mslp_Pa = mpcalc.smooth_n_point(mslp, 9, 5)
smooth_mslp_hPa = smooth_mslp_Pa / 100

thickness_1000_500 = smooth_height500 - smooth_height1000

# Create the Chart
# ------------
#
# Set projection of map display
mapproj = ccrs.LambertConformal(central_latitude=45., central_longitude=-100.)

# Set projection of data
dataproj = ccrs.PlateCarree()

fig = plt.figure(1, figsize=(14, 12))
ax = plt.subplot(111, projection=mapproj)
ax.set_extent([-128.5, -72.5, 20.5, 58], ccrs.PlateCarree())

# Add map features for geographic reference
ax.add_feature(cfeature.COASTLINE.with_scale('50m'), edgecolor='grey', zorder=1)
ax.add_feature(cfeature.LAND.with_scale('50m'), facecolor='white', edgecolor='grey', zorder=1)
ax.add_feature(cfeature.STATES.with_scale('50m'), edgecolor='grey', zorder=1)

# Plot 1000-500 mb thickness with multiple colors
clevs = (np.arange(0, 5400, 60),
         np.array([5400]),
         np.arange(5460, 7000, 60))
colors = ('tab:blue', 'b', 'tab:red')
kw_clabels = {'fontsize': 11, 'inline': True, 'inline_spacing': 5, 'fmt': '%i',
              'rightside_up': True, 'use_clabeltext': True}
for clevthick, color in zip(clevs, colors):
    cs = ax.contour(lons, lats, thickness_1000_500, levels=clevthick, colors=color,
                    linewidths=1.5, linestyles='dashed', transform=dataproj)
    plt.clabel(cs, **kw_clabels)

# Plot MSLP
clevmslp = np.arange(800., 1120., 4)
cs2 = ax.contour(lons, lats, smooth_mslp_hPa, clevmslp, colors='k', linewidths=1.5,
                 linestyles='solid', transform=dataproj)
plt.clabel(cs2, **kw_clabels)

# Plot MSLP high and low centers using the helper function.
# This filters extrema to the displayed map extent and avoids invalid projected
# coordinates that can cause scattertext() to fail during savefig().
mslp_data = smooth_mslp_hPa.magnitude

plot_maxmin_points(
    ax, lons, lats, mslp_data,
    'max', 50,
    symbol='H',
    color='blue',
    transform=ccrs.PlateCarree()
)

plot_maxmin_points(
    ax, lons, lats, mslp_data,
    'min', 25,
    symbol='L',
    color='red',
    transform=ccrs.PlateCarree()
)

# Put on some titles
plt.title('GFS MSLP (mb) and 1000–500 mb Thickness (m)', loc='left')
plt.title('VALID: ' + str(day_12Z) + ' ' + str(month_name_12Z) + ' ' + str(year_12Z) + ' ' + str(hour_12Z) + '00 UTC', loc='right')

plot_forecast_city_marker(ax)
plt.savefig('MSLP_1000-500mb_Thickness_Chart_GFS_12Z_Analysis.png', dpi=500, bbox_inches='tight')
plt.close()
print('Chart 19 complete')


################################################################################################################
# FORECAST CHARTS
################################################################################################################

# CHART 20: 500-mb Vorticity Chart (18-hr Forecast)

# Create NCSS object to access the NetcdfSubset
url = 'https://thredds.ucar.edu/thredds/ncss/grid/grib/NCEP/GFS/Global_0p5deg/GFS_Global_0p5deg_' + str(master_date_06Z) + '.grib2'
ncss = NCSS(url)

# Create lat/lon box for location you want to get data for
query = ncss.query()
query.lonlat_box(north=90, south=0, east=-30, west=-150)
query.time(datetime(fcst_18hr_year, fcst_18hr_month, fcst_18hr_day, int(fcst_18hr_hour)))

# Request data for geopotential height
query.variables('Geopotential_height_isobaric', 'u-component_of_wind_isobaric',
                'v-component_of_wind_isobaric')
query.vertical_level(50000)
data_query = ncss.get_data(query)

# Pull out variables you want to use
height_var = data_query.variables['Geopotential_height_isobaric']
u_wind_var = data_query.variables['u-component_of_wind_isobaric']
v_wind_var = data_query.variables['v-component_of_wind_isobaric']

dlat = data_query.variables['Geopotential_height_isobaric'].dimensions[2]
dlon = data_query.variables['Geopotential_height_isobaric'].dimensions[3]
lat = data_query.variables[dlat][:]
lon = data_query.variables[dlon][:]
lons, lats = np.meshgrid(lon, lat)

height = height_var[0, 0, :, :].squeeze()
height = mpcalc.smooth_n_point(height, 9, 5)
u_wind = u_wind_var[0, 0, :, :].squeeze() * units('m/s')
v_wind = v_wind_var[0, 0, :, :].squeeze() * units('m/s')

# Compute north-relative wind components for plotting purposes
#uwnd_er, vwnd_er = earth_relative_wind_components(u_wind, v_wind)

# Smooth wind components as desired
uwnd_500 = mpcalc.smooth_n_point(u_wind, 9, 5) * units('m/s')
vwnd_500 = mpcalc.smooth_n_point(v_wind, 9, 5) * units('m/s')

# Calculate grid spacing that is sign aware to use in absolute vorticity calculation
dx, dy = mpcalc.lat_lon_grid_deltas(lons, lats)

# Calculate absolute vorticity from MetPy function
lats = lats.filled(np.nan)  # Replace mask with NaNs
avor_500 = mpcalc.absolute_vorticity(uwnd_500, vwnd_500, dx, dy, lats * units.degrees)

# Set up the projection that will be used for plotting
mapcrs = ccrs.LambertConformal(central_longitude=-100, central_latitude=35,
                               standard_parallels=(30, 60))

# Set up the projection of the data; if lat/lon then PlateCarree is what you want
datacrs = ccrs.PlateCarree()

# Start the figure and create plot axes with proper projection
fig = plt.figure(1, figsize=(14, 12))
ax = plt.subplot(111, projection=mapcrs)
ax.set_extent([-128.5, -72.5, 20.5, 58], ccrs.PlateCarree())

# Add geopolitical boundaries for map reference
ax.add_feature(cfeature.COASTLINE.with_scale('50m'))
ax.add_feature(cfeature.STATES.with_scale('50m'))

# Absolute Vorticity contour levels
clevs_500_avor = np.arange(-4, 34, 2)

#Colors
colors = [
"#fe01fe",
"#be00fd",
"#7e00fe",
"#0000fe",
"#0158fe",
"#028bfe",
"#02befe",
"#00fefd",
"#03e4ca",
"#01cb7d",
"#00b300",
"#7ecb05",
"#cbe506",
"#fdfd00",
"#fdcb00",
"#fe9800",
"#fd6503",
"#fd0100"
]
colormap = matplotlib.colors.ListedColormap(colors)
colormap.set_under("#cc00cc")
colormap.set_over("#cc0100")
levels = np.array([-4, -2, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32])
norm = matplotlib.colors.BoundaryNorm(boundaries=levels, ncolors=18)

# Plot absolute vorticity values (multiplying by 10^5 to scale appropriately)
cf = ax.contourf(lons, lats, avor_500*1e5, clevs_500_avor, norm=norm, cmap=colormap, extend='both',
                 transform=datacrs)
cb = plt.colorbar(cf, orientation='horizontal', pad=0.02, aspect=50)
cb.set_label('Abs. Vorticity ($x 10^{-5} s^{-1}$)')

# Plot 500-hPa Geopotential Heights in decameters
hght_500_dam = height / 10
clevs_500_hght = np.arange(0, 8000, 6)
cs = ax.contour(lons, lats, hght_500_dam, clevs_500_hght, colors='white', transform=datacrs)
plt.clabel(cs, fmt='%d')

hght_500_data = hght_500_dam.magnitude

# Plot high and low centers safely, preserving the white text with black outline.
plot_maxmin_points(
    ax, lons, lats, hght_500_data,
    'max', 50,
    symbol='H',
    color='white',
    transform=ccrs.PlateCarree(),
    symbol_size=20,
    value_size=12,
    stroke_color='black',
    stroke_width=2
)

plot_maxmin_points(
    ax, lons, lats, hght_500_data,
    'min', 25,
    symbol='L',
    color='white',
    transform=ccrs.PlateCarree(),
    symbol_size=20,
    value_size=12,
    stroke_color='black',
    stroke_width=2
)

# Plot two titles, one on right and left side
plt.title('GFS 500-mb Geopotential Heights (dam)'
          ' and Abs. Vorticity ($x 10^{-5} s^{-1}$)', loc='left')
plt.title('VALID: ' + str(fcst_18hr_day) + ' ' + str(fcst_18hr_month_name) + ' ' + str(fcst_18hr_year) + ' ' + str(fcst_18hr_hour) + '00 UTC', loc='right')

plot_forecast_city_marker(ax)
plt.savefig('500mb_Vorticity_Chart_GFS_18hr_Forecast.png', dpi=500, bbox_inches='tight')
plt.close()
print('Chart 20 complete')


################################################################################################################

# CHART 21: 500-mb Vorticity Chart (30-hr Forecast)

# Create NCSS object to access the NetcdfSubset
url = 'https://thredds.ucar.edu/thredds/ncss/grid/grib/NCEP/GFS/Global_0p5deg/GFS_Global_0p5deg_' + str(master_date_06Z) + '.grib2'
ncss = NCSS(url)

# Create lat/lon box for location you want to get data for
query = ncss.query()
query.lonlat_box(north=90, south=0, east=-30, west=-150)
query.time(datetime(fcst_30hr_year, fcst_30hr_month, fcst_30hr_day, int(fcst_30hr_hour)))

# Request data for geopotential height
query.variables('Geopotential_height_isobaric', 'u-component_of_wind_isobaric',
                'v-component_of_wind_isobaric')
query.vertical_level(50000)
data_query = ncss.get_data(query)

# Pull out variables you want to use
height_var = data_query.variables['Geopotential_height_isobaric']
u_wind_var = data_query.variables['u-component_of_wind_isobaric']
v_wind_var = data_query.variables['v-component_of_wind_isobaric']

dlat = data_query.variables['Geopotential_height_isobaric'].dimensions[2]
dlon = data_query.variables['Geopotential_height_isobaric'].dimensions[3]
lat = data_query.variables[dlat][:]
lon = data_query.variables[dlon][:]
lons, lats = np.meshgrid(lon, lat)

height = height_var[0, 0, :, :].squeeze()
height = mpcalc.smooth_n_point(height, 9, 5)
u_wind = u_wind_var[0, 0, :, :].squeeze() * units('m/s')
v_wind = v_wind_var[0, 0, :, :].squeeze() * units('m/s')

# Compute north-relative wind components for plotting purposes
#uwnd_er, vwnd_er = earth_relative_wind_components(u_wind, v_wind)

# Smooth wind components as desired
uwnd_500 = mpcalc.smooth_n_point(u_wind, 9, 5) * units('m/s')
vwnd_500 = mpcalc.smooth_n_point(v_wind, 9, 5) * units('m/s')

# Calculate grid spacing that is sign aware to use in absolute vorticity calculation
dx, dy = mpcalc.lat_lon_grid_deltas(lons, lats)

# Calculate absolute vorticity from MetPy function
lats = lats.filled(np.nan)  # Replace mask with NaNs
avor_500 = mpcalc.absolute_vorticity(uwnd_500, vwnd_500, dx, dy, lats * units.degrees)

# Set up the projection that will be used for plotting
mapcrs = ccrs.LambertConformal(central_longitude=-100, central_latitude=35,
                               standard_parallels=(30, 60))

# Set up the projection of the data; if lat/lon then PlateCarree is what you want
datacrs = ccrs.PlateCarree()

# Start the figure and create plot axes with proper projection
fig = plt.figure(1, figsize=(14, 12))
ax = plt.subplot(111, projection=mapcrs)
ax.set_extent([-128.5, -72.5, 20.5, 58], ccrs.PlateCarree())

# Add geopolitical boundaries for map reference
ax.add_feature(cfeature.COASTLINE.with_scale('50m'))
ax.add_feature(cfeature.STATES.with_scale('50m'))

# Absolute Vorticity contour levels
clevs_500_avor = np.arange(-4, 34, 2)

#Colors
colors = [
"#fe01fe",
"#be00fd",
"#7e00fe",
"#0000fe",
"#0158fe",
"#028bfe",
"#02befe",
"#00fefd",
"#03e4ca",
"#01cb7d",
"#00b300",
"#7ecb05",
"#cbe506",
"#fdfd00",
"#fdcb00",
"#fe9800",
"#fd6503",
"#fd0100"
]
colormap = matplotlib.colors.ListedColormap(colors)
colormap.set_under("#cc00cc")
colormap.set_over("#cc0100")
levels = np.array([-4, -2, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32])
norm = matplotlib.colors.BoundaryNorm(boundaries=levels, ncolors=18)

# Plot absolute vorticity values (multiplying by 10^5 to scale appropriately)
cf = ax.contourf(lons, lats, avor_500*1e5, clevs_500_avor, norm=norm, cmap=colormap, extend='both',
                 transform=datacrs)
cb = plt.colorbar(cf, orientation='horizontal', pad=0.02, aspect=50)
cb.set_label('Abs. Vorticity ($x 10^{-5} s^{-1}$)')

# Plot 500-hPa Geopotential Heights in decameters
hght_500_dam = height / 10
clevs_500_hght = np.arange(0, 8000, 6)
cs = ax.contour(lons, lats, hght_500_dam, clevs_500_hght, colors='white', transform=datacrs)
plt.clabel(cs, fmt='%d')

hght_500_data = hght_500_dam.magnitude

# Plot high and low centers safely, preserving the white text with black outline.
plot_maxmin_points(
    ax, lons, lats, hght_500_data,
    'max', 50,
    symbol='H',
    color='white',
    transform=ccrs.PlateCarree(),
    symbol_size=20,
    value_size=12,
    stroke_color='black',
    stroke_width=2
)

plot_maxmin_points(
    ax, lons, lats, hght_500_data,
    'min', 25,
    symbol='L',
    color='white',
    transform=ccrs.PlateCarree(),
    symbol_size=20,
    value_size=12,
    stroke_color='black',
    stroke_width=2
)

# Plot two titles, one on right and left side
plt.title('GFS 500-mb Geopotential Heights (dam)'
          ' and Abs. Vorticity ($x 10^{-5} s^{-1}$)', loc='left')
plt.title('VALID: ' + str(fcst_30hr_day) + ' ' + str(fcst_30hr_month_name) + ' ' + str(fcst_30hr_year) + ' ' + str(fcst_30hr_hour) + '00 UTC', loc='right')

plot_forecast_city_marker(ax)
plt.savefig('500mb_Vorticity_Chart_GFS_30hr_Forecast.png', dpi=500, bbox_inches='tight')
plt.close()
print('Chart 21 complete')


################################################################################################################

# CHART 22: 850-mb Temperature Chart (18-hr Forecast)

# Create NCSS object to access the NetcdfSubset
url = 'https://thredds.ucar.edu/thredds/ncss/grid/grib/NCEP/GFS/Global_0p5deg/GFS_Global_0p5deg_' + str(master_date_06Z) + '.grib2'
ncss = NCSS(url)

# Create lat/lon box for location you want to get data for
query = ncss.query()
query.lonlat_box(north=90, south=0, east=-30, west=-150)
query.time(datetime(fcst_18hr_year, fcst_18hr_month, fcst_18hr_day, int(fcst_18hr_hour)))

# Request data for geopotential height
query.variables('Geopotential_height_isobaric', 'Temperature_isobaric')
query.vertical_level(85000)
data_query = ncss.get_data(query)

# Pull out variables you want to use
height_var = data_query.variables['Geopotential_height_isobaric']
temp_var = data_query.variables['Temperature_isobaric']

dlat = data_query.variables['Geopotential_height_isobaric'].dimensions[2]
dlon = data_query.variables['Geopotential_height_isobaric'].dimensions[3]
lat = data_query.variables[dlat][:]
lon = data_query.variables[dlon][:]
lons, lats = np.meshgrid(lon, lat)

height = height_var[0, 0, :, :].squeeze()
smooth_height = mpcalc.smooth_n_point(height, 9, 5)
temp = temp_var[0, 0, :, :].squeeze()
smooth_tmpk = mpcalc.smooth_n_point(temp, 9, 5)
smooth_tmpc = smooth_tmpk - 273.15

# Set up the projection that will be used for plotting
mapcrs = ccrs.LambertConformal(central_longitude=-100, central_latitude=35,
                               standard_parallels=(30, 60))

# Set up the projection of the data; if lat/lon then PlateCarree is what you want
datacrs = ccrs.PlateCarree()

# Start the figure and create plot axes with proper projection
fig = plt.figure(1, figsize=(14, 12))
ax = plt.subplot(111, projection=mapcrs)
ax.set_extent([-128.5, -72.5, 20.5, 58], ccrs.PlateCarree())

# Add geopolitical boundaries for map reference
ax.add_feature(cfeature.COASTLINE.with_scale('50m'))
ax.add_feature(cfeature.STATES.with_scale('50m'))

# Temperature contour levels
clevs_850_tmp = np.arange(-14, 32, 2)

#Colors
colors = [
"#fe01fe",
"#be00fd",
"#7e00fe",
"#0000fe",
"#0059fe",
"#008cfe",
"#03bffe",
"#00fefe",
"#00e5ca",
"#00cc7f",
"#02b202",
"#7fcb00",
"#cce505",
"#fefe00",
"#fecb02",
"#fe9800",
"#fe6600",
"#fe0000",
"#cc0100",
"#980001",
"#660000",
"#660066"
]
colormap = matplotlib.colors.ListedColormap(colors)
colormap.set_under("#cc00cc")
colormap.set_over("#980098")
levels = np.array([-14, -12, -10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30])
norm = matplotlib.colors.BoundaryNorm(boundaries=levels, ncolors=22)

# Plot 850-mb temperature values
cf = ax.contourf(lons, lats, smooth_tmpc, clevs_850_tmp, norm=norm, cmap=colormap, extend='both',
                 transform=datacrs)
cb = plt.colorbar(cf, orientation='horizontal', pad=0.02, aspect=50)
cb.set_ticks(levels[::1])
cb.set_label('Temperature (°C)')

# Plot 850-hPa Geopotential Heights in decameters
hght_850_dam = smooth_height / 10
clevs_850_hght = np.arange(0, 8000, 3)
cs = ax.contour(lons, lats, hght_850_dam, clevs_850_hght, colors='white', transform=datacrs)
plt.clabel(cs, fmt='%d')

hght_850_data = hght_850_dam.magnitude

# Plot high and low centers safely, preserving the white text with black outline.
plot_maxmin_points(
    ax, lons, lats, hght_850_data,
    'max', 50,
    symbol='H',
    color='white',
    transform=ccrs.PlateCarree(),
    symbol_size=20,
    value_size=12,
    stroke_color='black',
    stroke_width=2
)

plot_maxmin_points(
    ax, lons, lats, hght_850_data,
    'min', 25,
    symbol='L',
    color='white',
    transform=ccrs.PlateCarree(),
    symbol_size=20,
    value_size=12,
    stroke_color='black',
    stroke_width=2
)

# Plot two titles, one on right and left side
plt.title('GFS 850-mb Geopotential Heights (dam)'
          ' and Temperature (°C)', loc='left')
plt.title('VALID: ' + str(fcst_18hr_day) + ' ' + str(fcst_18hr_month_name) + ' ' + str(fcst_18hr_year) + ' ' + str(fcst_18hr_hour) + '00 UTC', loc='right')

plot_forecast_city_marker(ax)
plt.savefig('850mb_Temperature_Chart_GFS_18hr_Forecast.png', dpi=500, bbox_inches='tight')
plt.close()
print('Chart 22 complete')


################################################################################################################

# CHART 23: 850-mb Temperature Chart (30-hr Forecast)

# Create NCSS object to access the NetcdfSubset
url = 'https://thredds.ucar.edu/thredds/ncss/grid/grib/NCEP/GFS/Global_0p5deg/GFS_Global_0p5deg_' + str(master_date_06Z) + '.grib2'
ncss = NCSS(url)

# Create lat/lon box for location you want to get data for
query = ncss.query()
query.lonlat_box(north=90, south=0, east=-30, west=-150)
query.time(datetime(fcst_30hr_year, fcst_30hr_month, fcst_30hr_day, int(fcst_30hr_hour)))

# Request data for geopotential height
query.variables('Geopotential_height_isobaric', 'Temperature_isobaric')
query.vertical_level(85000)
data_query = ncss.get_data(query)

# Pull out variables you want to use
height_var = data_query.variables['Geopotential_height_isobaric']
temp_var = data_query.variables['Temperature_isobaric']

dlat = data_query.variables['Geopotential_height_isobaric'].dimensions[2]
dlon = data_query.variables['Geopotential_height_isobaric'].dimensions[3]
lat = data_query.variables[dlat][:]
lon = data_query.variables[dlon][:]
lons, lats = np.meshgrid(lon, lat)

height = height_var[0, 0, :, :].squeeze()
smooth_height = mpcalc.smooth_n_point(height, 9, 5)
temp = temp_var[0, 0, :, :].squeeze()
smooth_tmpk = mpcalc.smooth_n_point(temp, 9, 5)
smooth_tmpc = smooth_tmpk - 273.15

# Set up the projection that will be used for plotting
mapcrs = ccrs.LambertConformal(central_longitude=-100, central_latitude=35,
                               standard_parallels=(30, 60))

# Set up the projection of the data; if lat/lon then PlateCarree is what you want
datacrs = ccrs.PlateCarree()

# Start the figure and create plot axes with proper projection
fig = plt.figure(1, figsize=(14, 12))
ax = plt.subplot(111, projection=mapcrs)
ax.set_extent([-128.5, -72.5, 20.5, 58], ccrs.PlateCarree())

# Add geopolitical boundaries for map reference
ax.add_feature(cfeature.COASTLINE.with_scale('50m'))
ax.add_feature(cfeature.STATES.with_scale('50m'))

# Temperature contour levels
clevs_850_tmp = np.arange(-14, 32, 2)

#Colors
colors = [
"#fe01fe",
"#be00fd",
"#7e00fe",
"#0000fe",
"#0059fe",
"#008cfe",
"#03bffe",
"#00fefe",
"#00e5ca",
"#00cc7f",
"#02b202",
"#7fcb00",
"#cce505",
"#fefe00",
"#fecb02",
"#fe9800",
"#fe6600",
"#fe0000",
"#cc0100",
"#980001",
"#660000",
"#660066"
]
colormap = matplotlib.colors.ListedColormap(colors)
colormap.set_under("#cc00cc")
colormap.set_over("#980098")
levels = np.array([-14, -12, -10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30])
norm = matplotlib.colors.BoundaryNorm(boundaries=levels, ncolors=22)

# Plot 850-mb temperature values
cf = ax.contourf(lons, lats, smooth_tmpc, clevs_850_tmp, norm=norm, cmap=colormap, extend='both',
                 transform=datacrs)
cb = plt.colorbar(cf, orientation='horizontal', pad=0.02, aspect=50)
cb.set_ticks(levels[::1])
cb.set_label('Temperature (°C)')

# Plot 850-hPa Geopotential Heights in decameters
hght_850_dam = smooth_height / 10
clevs_850_hght = np.arange(0, 8000, 3)
cs = ax.contour(lons, lats, hght_850_dam, clevs_850_hght, colors='white', transform=datacrs)
plt.clabel(cs, fmt='%d')

hght_850_data = hght_850_dam.magnitude

# Plot high and low centers safely, preserving the white text with black outline.
plot_maxmin_points(
    ax, lons, lats, hght_850_data,
    'max', 50,
    symbol='H',
    color='white',
    transform=ccrs.PlateCarree(),
    symbol_size=20,
    value_size=12,
    stroke_color='black',
    stroke_width=2
)

plot_maxmin_points(
    ax, lons, lats, hght_850_data,
    'min', 25,
    symbol='L',
    color='white',
    transform=ccrs.PlateCarree(),
    symbol_size=20,
    value_size=12,
    stroke_color='black',
    stroke_width=2
)

# Plot two titles, one on right and left side
plt.title('GFS 850-mb Geopotential Heights (dam)'
          ' and Temperature (°C)', loc='left')
plt.title('VALID: ' + str(fcst_30hr_day) + ' ' + str(fcst_30hr_month_name) + ' ' + str(fcst_30hr_year) + ' ' + str(fcst_30hr_hour) + '00 UTC', loc='right')

plot_forecast_city_marker(ax)
plt.savefig('850mb_Temperature_Chart_GFS_30hr_Forecast.png', dpi=500, bbox_inches='tight')
plt.close()
print('Chart 23 complete')


################################################################################################################

# CHART 24: MSLP/500-mb Height Stack Chart (18-hr Forecast)

# Gridded Data
# ------------
#
# Obtain NAM gridded output for contour plotting. Specifically,
# geopotential height and temperature data for the given level and subset
# for over North America. Data are smoothed for aesthetic reasons.
#

# Create NCSS object to access the NetcdfSubset
url = 'https://thredds.ucar.edu/thredds/ncss/grid/grib/NCEP/GFS/Global_0p5deg/GFS_Global_0p5deg_' + str(master_date_06Z) + '.grib2'
ncss = NCSS(url)

# Create lat/lon box for location you want to get data for
query = ncss.query()
query.lonlat_box(north=90, south=0, east=-30, west=-150)
query.time(datetime(fcst_18hr_year, fcst_18hr_month, fcst_18hr_day, int(fcst_18hr_hour)))

# Request data for geopotential height
query.variables('Geopotential_height_isobaric')
query.vertical_level(50000)
data_query = ncss.get_data(query)

# Create lat/lon box for location you want to get data for
query_mslp = ncss.query()
query_mslp.lonlat_box(north=90, south=0, east=-30, west=-150)
query_mslp.time(datetime(fcst_18hr_year, fcst_18hr_month, fcst_18hr_day, int(fcst_18hr_hour)))

# Request data for MSLP
query_mslp.variables('Pressure_reduced_to_MSL_msl')
data_query_mslp = ncss.get_data(query_mslp)

# Pull out variables you want to use
height_var = data_query.variables['Geopotential_height_isobaric']
mslp_var = data_query_mslp.variables['Pressure_reduced_to_MSL_msl']

dlat = data_query.variables['Geopotential_height_isobaric'].dimensions[2]
dlon = data_query.variables['Geopotential_height_isobaric'].dimensions[3]
lat = data_query.variables[dlat][:]
lon = data_query.variables[dlon][:]
lons, lats = np.meshgrid(lon, lat)

height = height_var[0, 0, :, :].squeeze()
smooth_height = mpcalc.smooth_n_point(height, 9, 5)
mslp = mslp_var[0, :, :].squeeze()
smooth_mslp_Pa = mpcalc.smooth_n_point(mslp, 9, 5)
smooth_mslp_hPa = smooth_mslp_Pa / 100

# Create the Chart
# ------------
#
# Set projection of map display
mapproj = ccrs.LambertConformal(central_latitude=45., central_longitude=-100.)

# Set projection of data
dataproj = ccrs.PlateCarree()

fig = plt.figure(1, figsize=(14, 12))
ax = plt.subplot(111, projection=mapproj)
ax.set_extent([-128.5, -72.5, 20.5, 58], ccrs.PlateCarree())

# Add map features for geographic reference
ax.add_feature(cfeature.COASTLINE.with_scale('50m'), edgecolor='0.25')
ax.add_feature(cfeature.LAND.with_scale('50m'), edgecolor='0.25')
ax.add_feature(cfeature.STATES.with_scale('50m'), edgecolor='0.25')

#Colors
colors = [
"#be00fd",
"#7e00fe",
"#0000fe",
"#0158fe",
#"#028bfe",
"#02befe",
"#00fefd",
#"#03e4ca",
"#01cb7d",
"#00b300",
#"#7ecb05",
"#cbe506",
"#fdfd00",
"#fdcb00",
"#fe9800",
"#fd6503",
"#fd0100",
"#980001"
]
colormap = matplotlib.colors.ListedColormap(colors)
colormap.set_under("#fe01fe")
colormap.set_over("#660066")
levels = np.array([504, 510, 516, 522, 528, 534, 540, 546, 552, 558, 564, 570, 576, 582, 588, 594])
norm = matplotlib.colors.BoundaryNorm(boundaries=levels, ncolors=15)

# Plot 500-mb Heights
hght_500_dam = smooth_height / 10
clevs_500_hght = np.arange(504, 600, 6)
cf = ax.contourf(lons, lats, hght_500_dam, clevs_500_hght, norm=norm, cmap=colormap, extend='both', transform=dataproj)

cb = plt.colorbar(cf, orientation='horizontal', pad=0.02, aspect=50)
cb.set_ticks(levels[::1])

# Plot MSLP
clevmslp = np.arange(800., 1120., 4)
cs = ax.contour(lons, lats, smooth_mslp_hPa, clevmslp, colors='k', linewidths=1.5,
                 linestyles='solid', transform=dataproj)
kw_clabels = {'fontsize': 11, 'inline': True, 'inline_spacing': 5, 'fmt': '%i',
              'rightside_up': True, 'use_clabeltext': True}
plt.clabel(cs, **kw_clabels, zorder=1)

# Use definition to plot H/L symbols
# Use smoothed MSLP directly (in hPa) to find local maxima/minima
# Remove units before processing
mslp_data = smooth_mslp_hPa.magnitude

# Plot high and low centers safely, preserving the white text with black outline.
plot_maxmin_points(
    ax, lons, lats, mslp_data,
    'max', 50,
    symbol='H',
    color='white',
    transform=ccrs.PlateCarree(),
    symbol_size=20,
    value_size=12,
    stroke_color='black',
    stroke_width=2
)

plot_maxmin_points(
    ax, lons, lats, mslp_data,
    'min', 25,
    symbol='L',
    color='white',
    transform=ccrs.PlateCarree(),
    symbol_size=20,
    value_size=12,
    stroke_color='black',
    stroke_width=2
)

# Put on some titles
plt.title('GFS MSLP (mb) and 500-mb Geopotential Heights (dam)', loc='left')
plt.title('VALID: ' + str(fcst_18hr_day) + ' ' + str(fcst_18hr_month_name) + ' ' + str(fcst_18hr_year) + ' ' + str(fcst_18hr_hour) + '00 UTC', loc='right')

plot_forecast_city_marker(ax)
plt.savefig('Stack_Chart_GFS_18hr_Forecast.png', dpi=500, bbox_inches='tight')
plt.close()
print('Chart 24 complete')


################################################################################################################

# CHART 25: MSLP/500-mb Height Stack Chart (30-hr Forecast)

# Gridded Data
# ------------
#
# Obtain NAM gridded output for contour plotting. Specifically,
# geopotential height and temperature data for the given level and subset
# for over North America. Data are smoothed for aesthetic reasons.
#

# Create NCSS object to access the NetcdfSubset
url = 'https://thredds.ucar.edu/thredds/ncss/grid/grib/NCEP/GFS/Global_0p5deg/GFS_Global_0p5deg_' + str(master_date_06Z) + '.grib2'
ncss = NCSS(url)

# Create lat/lon box for location you want to get data for
query = ncss.query()
query.lonlat_box(north=90, south=0, east=-30, west=-150)
query.time(datetime(fcst_30hr_year, fcst_30hr_month, fcst_30hr_day, int(fcst_30hr_hour)))

# Request data for geopotential height
query.variables('Geopotential_height_isobaric')
query.vertical_level(50000)
data_query = ncss.get_data(query)

# Create lat/lon box for location you want to get data for
query_mslp = ncss.query()
query_mslp.lonlat_box(north=90, south=0, east=-30, west=-150)
query_mslp.time(datetime(fcst_30hr_year, fcst_30hr_month, fcst_30hr_day, int(fcst_30hr_hour)))

# Request data for MSLP
query_mslp.variables('Pressure_reduced_to_MSL_msl')
data_query_mslp = ncss.get_data(query_mslp)

# Pull out variables you want to use
height_var = data_query.variables['Geopotential_height_isobaric']
mslp_var = data_query_mslp.variables['Pressure_reduced_to_MSL_msl']

dlat = data_query.variables['Geopotential_height_isobaric'].dimensions[2]
dlon = data_query.variables['Geopotential_height_isobaric'].dimensions[3]
lat = data_query.variables[dlat][:]
lon = data_query.variables[dlon][:]
lons, lats = np.meshgrid(lon, lat)

height = height_var[0, 0, :, :].squeeze()
smooth_height = mpcalc.smooth_n_point(height, 9, 5)
mslp = mslp_var[0, :, :].squeeze()
smooth_mslp_Pa = mpcalc.smooth_n_point(mslp, 9, 5)
smooth_mslp_hPa = smooth_mslp_Pa / 100

# Create the Chart
# ------------
#
# Set projection of map display
mapproj = ccrs.LambertConformal(central_latitude=45., central_longitude=-100.)

# Set projection of data
dataproj = ccrs.PlateCarree()

fig = plt.figure(1, figsize=(14, 12))
ax = plt.subplot(111, projection=mapproj)
ax.set_extent([-128.5, -72.5, 20.5, 58], ccrs.PlateCarree())

# Add map features for geographic reference
ax.add_feature(cfeature.COASTLINE.with_scale('50m'), edgecolor='0.25')
ax.add_feature(cfeature.LAND.with_scale('50m'), edgecolor='0.25')
ax.add_feature(cfeature.STATES.with_scale('50m'), edgecolor='0.25')

#Colors
colors = [
"#be00fd",
"#7e00fe",
"#0000fe",
"#0158fe",
#"#028bfe",
"#02befe",
"#00fefd",
#"#03e4ca",
"#01cb7d",
"#00b300",
#"#7ecb05",
"#cbe506",
"#fdfd00",
"#fdcb00",
"#fe9800",
"#fd6503",
"#fd0100",
"#980001"
]
colormap = matplotlib.colors.ListedColormap(colors)
colormap.set_under("#fe01fe")
colormap.set_over("#660066")
levels = np.array([504, 510, 516, 522, 528, 534, 540, 546, 552, 558, 564, 570, 576, 582, 588, 594])
norm = matplotlib.colors.BoundaryNorm(boundaries=levels, ncolors=15)

# Plot 500-mb Heights
hght_500_dam = smooth_height / 10
clevs_500_hght = np.arange(504, 600, 6)
cf = ax.contourf(lons, lats, hght_500_dam, clevs_500_hght, norm=norm, cmap=colormap, extend='both', transform=dataproj)

cb = plt.colorbar(cf, orientation='horizontal', pad=0.02, aspect=50)
cb.set_ticks(levels[::1])

# Plot MSLP
clevmslp = np.arange(800., 1120., 4)
cs = ax.contour(lons, lats, smooth_mslp_hPa, clevmslp, colors='k', linewidths=1.5,
                 linestyles='solid', transform=dataproj)
kw_clabels = {'fontsize': 11, 'inline': True, 'inline_spacing': 5, 'fmt': '%i',
              'rightside_up': True, 'use_clabeltext': True}
plt.clabel(cs, **kw_clabels, zorder=1)

# Use definition to plot H/L symbols
# Use smoothed MSLP directly (in hPa) to find local maxima/minima
# Remove units before processing
mslp_data = smooth_mslp_hPa.magnitude

# Plot high and low centers safely, preserving the white text with black outline.
plot_maxmin_points(
    ax, lons, lats, mslp_data,
    'max', 50,
    symbol='H',
    color='white',
    transform=ccrs.PlateCarree(),
    symbol_size=20,
    value_size=12,
    stroke_color='black',
    stroke_width=2
)

plot_maxmin_points(
    ax, lons, lats, mslp_data,
    'min', 25,
    symbol='L',
    color='white',
    transform=ccrs.PlateCarree(),
    symbol_size=20,
    value_size=12,
    stroke_color='black',
    stroke_width=2
)

# Put on some titles
plt.title('GFS MSLP (mb) and 500-mb Geopotential Heights (dam)', loc='left')
plt.title('VALID: ' + str(fcst_30hr_day) + ' ' + str(fcst_30hr_month_name) + ' ' + str(fcst_30hr_year) + ' ' + str(fcst_30hr_hour) + '00 UTC', loc='right')

plot_forecast_city_marker(ax)
plt.savefig('Stack_Chart_GFS_30hr_Forecast.png', dpi=500, bbox_inches='tight')
plt.close()
print('Chart 25 complete')


################################################################################################################

# CHART 26: MSLP/1000–500 mb Thickness Chart (18-hr Forecast)

# Gridded Data
# ------------
#
# Obtain NAM gridded output for contour plotting. Specifically,
# geopotential height and temperature data for the given level and subset
# for over North America. Data are smoothed for aesthetic reasons.
#

# Create NCSS object to access the NetcdfSubset
url = 'https://thredds.ucar.edu/thredds/ncss/grid/grib/NCEP/GFS/Global_0p5deg/GFS_Global_0p5deg_' + str(master_date_06Z) + '.grib2'
ncss = NCSS(url)

# Create lat/lon box for location you want to get data for
query_500hPa = ncss.query()
query_500hPa.lonlat_box(north=90, south=0, east=-30, west=-150)
query_500hPa.time(datetime(fcst_18hr_year, fcst_18hr_month, fcst_18hr_day, int(fcst_18hr_hour)))

# Request data for geopotential height
query_500hPa.variables('Geopotential_height_isobaric')
query_500hPa.vertical_level(50000)
data_query_500hPa = ncss.get_data(query_500hPa)

# Create lat/lon box for location you want to get data for
query_1000hPa = ncss.query()
query_1000hPa.lonlat_box(north=90, south=0, east=-30, west=-150)
query_1000hPa.time(datetime(fcst_18hr_year, fcst_18hr_month, fcst_18hr_day, int(fcst_18hr_hour)))

# Request data for geopotential height
query_1000hPa.variables('Geopotential_height_isobaric')
query_1000hPa.vertical_level(100000)
data_query_1000hPa = ncss.get_data(query_1000hPa)

# Create lat/lon box for location you want to get data for
query_mslp = ncss.query()
query_mslp.lonlat_box(north=90, south=0, east=-30, west=-150)
query_mslp.time(datetime(fcst_18hr_year, fcst_18hr_month, fcst_18hr_day, int(fcst_18hr_hour)))

# Request data for MSLP
query_mslp.variables('Pressure_reduced_to_MSL_msl')
data_query_mslp = ncss.get_data(query_mslp)

# Create lat/lon box for location you want to get data for
query_precip = ncss.query()
query_precip.lonlat_box(north=90, south=0, east=-30, west=-150)
query_precip.time(datetime(fcst_18hr_year, fcst_18hr_month, fcst_18hr_day, int(fcst_18hr_hour)))

# Request data for MSLP
query_precip.variables('Total_precipitation_surface_Mixed_intervals_Accumulation')
data_query_precip = ncss.get_data(query_precip)

# Pull out variables you want to use
height_500hPa_var = data_query_500hPa.variables['Geopotential_height_isobaric']
height_1000hPa_var = data_query_1000hPa.variables['Geopotential_height_isobaric']
mslp_var = data_query_mslp.variables['Pressure_reduced_to_MSL_msl']
precip_var = data_query_precip.variables['Total_precipitation_surface_Mixed_intervals_Accumulation']

dlat = data_query_500hPa.variables['Geopotential_height_isobaric'].dimensions[2]
dlon = data_query_500hPa.variables['Geopotential_height_isobaric'].dimensions[3]
lat = data_query_500hPa.variables[dlat][:]
lon = data_query_500hPa.variables[dlon][:]
lons, lats = np.meshgrid(lon, lat)

height500 = height_500hPa_var[0, 0, :, :].squeeze()
smooth_height500 = mpcalc.smooth_n_point(height500, 9, 5)
height1000 = height_1000hPa_var[0, 0, :, :].squeeze()
smooth_height1000 = mpcalc.smooth_n_point(height1000, 9, 5)
mslp = mslp_var[0, :, :].squeeze()
smooth_mslp_Pa = mpcalc.smooth_n_point(mslp, 9, 5)
smooth_mslp_hPa = smooth_mslp_Pa / 100
precip_mm = precip_var[0, :, :].squeeze()
precip_inches = precip_mm / 25.4

thickness_1000_500 = smooth_height500 - smooth_height1000

# Create the Chart
# ------------
#
# Set projection of map display
mapproj = ccrs.LambertConformal(central_latitude=45., central_longitude=-100.)

# Set projection of data
dataproj = ccrs.PlateCarree()

fig = plt.figure(1, figsize=(14, 12))
ax = plt.subplot(111, projection=mapproj)
ax.set_extent([-128.5, -72.5, 20.5, 58], ccrs.PlateCarree())

# Add map features for geographic reference
ax.add_feature(cfeature.COASTLINE.with_scale('50m'), edgecolor='grey', zorder=2)
ax.add_feature(cfeature.LAND.with_scale('50m'), facecolor='none', edgecolor='grey', zorder=2)
ax.add_feature(cfeature.STATES.with_scale('50m'), edgecolor='grey', zorder=2)

# Plot 1000-500 mb thickness with multiple colors
clevs = (np.arange(0, 5400, 60),
         np.array([5400]),
         np.arange(5460, 7000, 60))
colors = ('tab:blue', 'b', 'tab:red')
kw_clabels = {'fontsize': 11, 'inline': True, 'inline_spacing': 5, 'fmt': '%i',
              'rightside_up': True, 'use_clabeltext': True}
for clevthick, color in zip(clevs, colors):
    cs = ax.contour(lons, lats, thickness_1000_500, levels=clevthick, colors=color,
                    linewidths=1.5, linestyles='dashed', transform=dataproj)
    plt.clabel(cs, **kw_clabels)

# Plot MSLP
clevmslp = np.arange(800., 1120., 4)
cs2 = ax.contour(lons, lats, smooth_mslp_hPa, clevmslp, colors='k', linewidths=1.5,
                 linestyles='solid', transform=dataproj)
plt.clabel(cs2, **kw_clabels)

# Use definition to plot H/L symbols
# Use smoothed MSLP directly (in hPa) to find local maxima/minima
# Remove units before processing
mslp_data = smooth_mslp_hPa.magnitude

plot_maxmin_points(
    ax, lons, lats, mslp_data,
    'max', 50,
    symbol='H',
    color='blue',
    transform=ccrs.PlateCarree()
)

plot_maxmin_points(
    ax, lons, lats, mslp_data,
    'min', 25,
    symbol='L',
    color='red',
    transform=ccrs.PlateCarree()
)

# Plot 3-hour precipitation
nws_precip_colors = [
#"#b2fffe",
#"#04e9e7",
#"#019ff4",
#"#0300f4",
"#02fd02",
"#01c501",
"#008e00",
"#fdf802",
"#e5bc00",
"#fd9500",
"#e84a00",
"#d40000",
"#a90000",
"#f800fd",
"#c82ae2",
"#9854c6"
]
precip_colormap = ListedColormap(nws_precip_colors)
levels = np.array([0.01, 0.10, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 2.00, 2.50, 3.00, 4.00, 5.00])
norm = BoundaryNorm(boundaries=levels, ncolors=12)

# Plot with contour fill using NWS Reflectivity colormap
precip_contourfill = ax.contourf(lons, lats, precip_inches, levels, norm=norm, cmap=precip_colormap, transform=ccrs.PlateCarree())

# Colorbar
def format_tick(x, pos):
    return f"{x:.3f}".rstrip('0').rstrip('.') if '.' in f"{x:.3f}" else f"{x:.3f}"
cbar = plt.colorbar(precip_contourfill, ticks=levels, orientation='horizontal', pad=0.02, shrink=0.5, fraction=0.075)
cbar.ax.xaxis.set_major_formatter(FuncFormatter(format_tick))

# Put on some titles
#plt.title('NAM MSLP (mb) and 1000–500 mb Thickness (m)', loc='left')
plt.title('GFS MSLP (mb), 1000–500 mb Thickness (m), and 3-hr Precipitation (in)', loc='left')
plt.title('VALID: ' + str(fcst_18hr_day) + ' ' + str(fcst_18hr_month_name) + ' ' + str(fcst_18hr_year) + ' ' + str(fcst_18hr_hour) + '00 UTC', loc='right')

plot_forecast_city_marker(ax)
plt.savefig('MSLP_1000-500mb_Thickness_Chart_GFS_18hr_Forecast.png', dpi=500, bbox_inches='tight')
plt.close()
print('Chart 26 complete')


################################################################################################################

# CHART 27: MSLP/1000–500 mb Thickness Chart (30-hr Forecast)

# Gridded Data
# ------------
#
# Obtain NAM gridded output for contour plotting. Specifically,
# geopotential height and temperature data for the given level and subset
# for over North America. Data are smoothed for aesthetic reasons.
#

# Create NCSS object to access the NetcdfSubset
url = 'https://thredds.ucar.edu/thredds/ncss/grid/grib/NCEP/GFS/Global_0p5deg/GFS_Global_0p5deg_' + str(master_date_06Z) + '.grib2'
ncss = NCSS(url)

# Create lat/lon box for location you want to get data for
query_500hPa = ncss.query()
query_500hPa.lonlat_box(north=90, south=0, east=-30, west=-150)
query_500hPa.time(datetime(fcst_30hr_year, fcst_30hr_month, fcst_30hr_day, int(fcst_30hr_hour)))

# Request data for geopotential height
query_500hPa.variables('Geopotential_height_isobaric')
query_500hPa.vertical_level(50000)
data_query_500hPa = ncss.get_data(query_500hPa)

# Create lat/lon box for location you want to get data for
query_1000hPa = ncss.query()
query_1000hPa.lonlat_box(north=90, south=0, east=-30, west=-150)
query_1000hPa.time(datetime(fcst_30hr_year, fcst_30hr_month, fcst_30hr_day, int(fcst_30hr_hour)))

# Request data for geopotential height
query_1000hPa.variables('Geopotential_height_isobaric')
query_1000hPa.vertical_level(100000)
data_query_1000hPa = ncss.get_data(query_1000hPa)

# Create lat/lon box for location you want to get data for
query_mslp = ncss.query()
query_mslp.lonlat_box(north=90, south=0, east=-30, west=-150)
query_mslp.time(datetime(fcst_30hr_year, fcst_30hr_month, fcst_30hr_day, int(fcst_30hr_hour)))

# Request data for MSLP
query_mslp.variables('Pressure_reduced_to_MSL_msl')
data_query_mslp = ncss.get_data(query_mslp)

# Create lat/lon box for location you want to get data for
query_precip = ncss.query()
query_precip.lonlat_box(north=90, south=0, east=-30, west=-150)
query_precip.time(datetime(fcst_30hr_year, fcst_30hr_month, fcst_30hr_day, int(fcst_30hr_hour)))

# Request data for MSLP
query_precip.variables('Total_precipitation_surface_Mixed_intervals_Accumulation')
data_query_precip = ncss.get_data(query_precip)

# Pull out variables you want to use
height_500hPa_var = data_query_500hPa.variables['Geopotential_height_isobaric']
height_1000hPa_var = data_query_1000hPa.variables['Geopotential_height_isobaric']
mslp_var = data_query_mslp.variables['Pressure_reduced_to_MSL_msl']
precip_var = data_query_precip.variables['Total_precipitation_surface_Mixed_intervals_Accumulation']

dlat = data_query_500hPa.variables['Geopotential_height_isobaric'].dimensions[2]
dlon = data_query_500hPa.variables['Geopotential_height_isobaric'].dimensions[3]
lat = data_query_500hPa.variables[dlat][:]
lon = data_query_500hPa.variables[dlon][:]
lons, lats = np.meshgrid(lon, lat)

height500 = height_500hPa_var[0, 0, :, :].squeeze()
smooth_height500 = mpcalc.smooth_n_point(height500, 9, 5)
height1000 = height_1000hPa_var[0, 0, :, :].squeeze()
smooth_height1000 = mpcalc.smooth_n_point(height1000, 9, 5)
mslp = mslp_var[0, :, :].squeeze()
smooth_mslp_Pa = mpcalc.smooth_n_point(mslp, 9, 5)
smooth_mslp_hPa = smooth_mslp_Pa / 100
precip_mm = precip_var[0, :, :].squeeze()
precip_inches = precip_mm / 25.4

thickness_1000_500 = smooth_height500 - smooth_height1000

# Create the Chart
# ------------
#
# Set projection of map display
mapproj = ccrs.LambertConformal(central_latitude=45., central_longitude=-100.)

# Set projection of data
dataproj = ccrs.PlateCarree()

fig = plt.figure(1, figsize=(14, 12))
ax = plt.subplot(111, projection=mapproj)
ax.set_extent([-128.5, -72.5, 20.5, 58], ccrs.PlateCarree())

# Add map features for geographic reference
ax.add_feature(cfeature.COASTLINE.with_scale('50m'), edgecolor='grey', zorder=2)
ax.add_feature(cfeature.LAND.with_scale('50m'), facecolor='none', edgecolor='grey', zorder=2)
ax.add_feature(cfeature.STATES.with_scale('50m'), edgecolor='grey', zorder=2)

# Plot 1000-500 mb thickness with multiple colors
clevs = (np.arange(0, 5400, 60),
         np.array([5400]),
         np.arange(5460, 7000, 60))
colors = ('tab:blue', 'b', 'tab:red')
kw_clabels = {'fontsize': 11, 'inline': True, 'inline_spacing': 5, 'fmt': '%i',
              'rightside_up': True, 'use_clabeltext': True}
for clevthick, color in zip(clevs, colors):
    cs = ax.contour(lons, lats, thickness_1000_500, levels=clevthick, colors=color,
                    linewidths=1.5, linestyles='dashed', transform=dataproj)
    plt.clabel(cs, **kw_clabels)

# Plot MSLP
clevmslp = np.arange(800., 1120., 4)
cs2 = ax.contour(lons, lats, smooth_mslp_hPa, clevmslp, colors='k', linewidths=1.5,
                 linestyles='solid', transform=dataproj)
plt.clabel(cs2, **kw_clabels)

# Use definition to plot H/L symbols
# Use smoothed MSLP directly (in hPa) to find local maxima/minima
# Remove units before processing
mslp_data = smooth_mslp_hPa.magnitude

plot_maxmin_points(
    ax, lons, lats, mslp_data,
    'max', 50,
    symbol='H',
    color='blue',
    transform=ccrs.PlateCarree()
)

plot_maxmin_points(
    ax, lons, lats, mslp_data,
    'min', 25,
    symbol='L',
    color='red',
    transform=ccrs.PlateCarree()
)

# Plot 3-hour precipitation
nws_precip_colors = [
#"#b2fffe",
#"#04e9e7",
#"#019ff4",
#"#0300f4",
"#02fd02",
"#01c501",
"#008e00",
"#fdf802",
"#e5bc00",
"#fd9500",
"#e84a00",
"#d40000",
"#a90000",
"#f800fd",
"#c82ae2",
"#9854c6"
]
precip_colormap = ListedColormap(nws_precip_colors)
levels = np.array([0.01, 0.10, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 2.00, 2.50, 3.00, 4.00, 5.00])
norm = BoundaryNorm(boundaries=levels, ncolors=12)

# Plot with contour fill using NWS Reflectivity colormap
precip_contourfill = ax.contourf(lons, lats, precip_inches, levels, norm=norm, cmap=precip_colormap, transform=ccrs.PlateCarree())

# Colorbar
def format_tick(x, pos):
    return f"{x:.3f}".rstrip('0').rstrip('.') if '.' in f"{x:.3f}" else f"{x:.3f}"
cbar = plt.colorbar(precip_contourfill, ticks=levels, orientation='horizontal', pad=0.02, shrink=0.5, fraction=0.075)
cbar.ax.xaxis.set_major_formatter(FuncFormatter(format_tick))

# Put on some titles
#plt.title('NAM MSLP (mb) and 1000–500 mb Thickness (m)', loc='left')
plt.title('GFS MSLP (mb), 1000–500 mb Thickness (m), and 3-hr Precipitation (in)', loc='left')
plt.title('VALID: ' + str(fcst_30hr_day) + ' ' + str(fcst_30hr_month_name) + ' ' + str(fcst_30hr_year) + ' ' + str(fcst_30hr_hour) + '00 UTC', loc='right')

plot_forecast_city_marker(ax)
plt.savefig('MSLP_1000-500mb_Thickness_Chart_GFS_30hr_Forecast.png', dpi=500, bbox_inches='tight')
plt.close()
print('Chart 27 complete')

import subprocess

print("All charts completed successfully.")
print("Uploading charts to GitHub...")

subprocess.run(
    ['/home/wiley/upload_synoptic_charts_revised_current.sh'],
    check=True
)

print("Charts successfully uploaded to GitHub.")
