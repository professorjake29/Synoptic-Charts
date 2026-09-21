#!/usr/bin/env python3

from datetime import datetime, timedelta, timezone
import requests

# UCAR THREDDS path for GFS 0.5-degree ANALYSIS files
BASE_URL = (
    "https://thredds.ucar.edu/thredds/ncss/grid/grib/"
    "NCEP/GFS/Global_0p5deg_ana/"
)

VALID_CYCLES = (0, 6, 12, 18)
CYCLES_TO_TEST = 8
TIMEOUT = 20


def floor_cycle(dt):
    """Return the most recent standard GFS cycle at or before dt."""
    hour = max(h for h in VALID_CYCLES if h <= dt.hour)
    return dt.replace(hour=hour, minute=0, second=0, microsecond=0)


def dataset_url(cycle):
    """Build the UCAR THREDDS URL for a GFS analysis cycle."""
    stamp = cycle.strftime("%Y%m%d_%H%M")
    return f"{BASE_URL}GFS_Global_0p5deg_ana_{stamp}.grib2"


def check_cycle(cycle):
    """Check whether dataset.xml exists for a given analysis cycle."""
    url = dataset_url(cycle)
    metadata_url = url + "/dataset.xml"

    try:
        response = requests.get(metadata_url, timeout=TIMEOUT)
        return response.status_code == 200, response.status_code, url
    except requests.RequestException as exc:
        return False, str(exc), url


def main():
    now = datetime.now(timezone.utc)
    cycle = floor_cycle(now)

    print("=" * 76)
    print(" UCAR THREDDS GFS 0.5-degree ANALYSIS Availability Check")
    print("=" * 76)
    print(f"Current UTC time: {now:%Y-%m-%d %H:%M UTC}")
    print()

    available = []

    for _ in range(CYCLES_TO_TEST):
        ok, status, url = check_cycle(cycle)
        label = cycle.strftime("%Y-%m-%d %HZ")

        if ok:
            print(f"[AVAILABLE] {label}   HTTP {status}")
            available.append(cycle)
        else:
            print(f"[NOT READY] {label}   {status}")

        cycle -= timedelta(hours=6)

    print()
    print("-" * 76)

    if available:
        newest = available[0]

        print(
            f"Newest available GFS analysis cycle: "
            f"{newest:%Y-%m-%d %HZ}"
        )
        print()
        print("Dataset URL:")
        print(dataset_url(newest))

        # Also give explicit status for the most recent 00Z and 12Z cycles,
        # since those are the analysis cycles used by the synoptic-chart script.
        print()
        print("00Z / 12Z briefing-chart analysis status:")

        for target_hour in (12, 0):
            candidate = now.replace(
                hour=target_hour,
                minute=0,
                second=0,
                microsecond=0
            )

            if candidate > now:
                candidate -= timedelta(days=1)

            ok, status, _ = check_cycle(candidate)
            state = "AVAILABLE" if ok else "NOT READY"
            print(f"  {candidate:%Y-%m-%d %HZ}: {state} ({status})")

    else:
        print("No tested GFS analysis cycles were available.")

    print("-" * 76)


if __name__ == "__main__":
    main()
