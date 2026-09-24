# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 05:54:26 2026

@author: jakew
"""

import requests

station_id = "KBMX"

url = "https://www.ncei.noaa.gov/access/homr/services/station/search"
response = requests.get(
    url,
    params={
        "qid": f"ICAO:{station_id}",
        "platform": "UPPERAIR",
        "definitions": "false",
        "phrData": "false",
    },
    timeout=30,
)
response.raise_for_status()

stations = response.json().get("stationCollection", {}).get("stations", [])

wmo_ids = {
    identifier["id"]
    for station in stations
    for identifier in station.get("identifiers", [])
    if identifier.get("idType") == "WMO"
    and len(identifier.get("id", "")) == 5
}

if wmo_ids:
    print(f"{station_id} WMO station number(s): {', '.join(sorted(wmo_ids))}")
else:
    print(f"No WMO number found in HOMR for {station_id}.")