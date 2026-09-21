#!/usr/bin/env python3
from datetime import datetime, timedelta, timezone
import requests

BASE_URL = "https://thredds.ucar.edu/thredds/ncss/grid/grib/NCEP/GFS/Global_0p5deg/"
VALID_CYCLES = (0, 6, 12, 18)
CYCLES_TO_TEST = 8
TIMEOUT = 20

def floor_cycle(dt):
    hour = max(h for h in VALID_CYCLES if h <= dt.hour)
    return dt.replace(hour=hour, minute=0, second=0, microsecond=0)

def dataset_url(cycle):
    stamp = cycle.strftime("%Y%m%d_%H%M")
    return f"{BASE_URL}GFS_Global_0p5deg_{stamp}.grib2"

def check_cycle(cycle):
    url = dataset_url(cycle)
    try:
        r = requests.get(url + "/dataset.xml", timeout=TIMEOUT)
        return r.status_code == 200, r.status_code, url
    except requests.RequestException as e:
        return False, str(e), url

def main():
    now = datetime.now(timezone.utc)
    cycle = floor_cycle(now)

    print("=" * 72)
    print("UCAR THREDDS GFS 0.5-degree Forecast Availability Check")
    print("=" * 72)
    print(f"Current UTC time: {now:%Y-%m-%d %H:%M UTC}\n")

    available = []

    for _ in range(CYCLES_TO_TEST):
        ok, status, url = check_cycle(cycle)
        label = cycle.strftime("%Y-%m-%d %HZ")
        if ok:
            print(f"[AVAILABLE] {label}  HTTP {status}")
            available.append(cycle)
        else:
            print(f"[NOT READY] {label}  {status}")
        cycle -= timedelta(hours=6)

    print("\n" + "-" * 72)
    if available:
        newest = available[0]
        print(f"Newest available cycle: {newest:%Y-%m-%d %HZ}")
        print(dataset_url(newest))
    else:
        print("No tested cycles were available.")
    print("-" * 72)

if __name__ == "__main__":
    main()
