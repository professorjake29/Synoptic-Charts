# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 05:48:12 2026

@author: jakew
"""

from datetime import datetime, timedelta, timezone
import requests

station_number = "71845"
url = "https://weather.uwyo.edu/wsgi/sounding"

now = datetime.now(timezone.utc)
cycle = now.replace(
    hour=12 if now.hour >= 12 else 0,
    minute=0,
    second=0,
    microsecond=0,
)

latest_available = None

for offset in range(4):
    sounding_time = cycle - timedelta(hours=12 * offset)

    with requests.get(
        url,
        params={
            "datetime": sounding_time.strftime("%Y-%m-%d %H:%M:%S"),
            "id": station_number,
            "src": "UNKNOWN",
            "type": "TEXT:CSV",
        },
        timeout=30,
        stream=True,
    ) as response:
        if response.status_code == 400:
            if not response.text.startswith("Unable to retrieve the data"):
                response.raise_for_status()
            available = False
        else:
            response.raise_for_status()
            lines = response.iter_lines(decode_unicode=True)
            next(lines, None)                   # CSV headings
            available = bool(next(lines, None))  # First data row

    print(
        f"{sounding_time:%Y-%m-%d %HZ}: "
        f"{'available' if available else 'not available'}"
    )

    if available and latest_available is None:
        latest_available = sounding_time

if latest_available is not None:
    print(
        f"\nLatest available {station_number} sounding: "
        f"{latest_available:%Y-%m-%d %HZ}"
    )
else:
    print(f"\nNo {station_number} sounding found in the last four cycles.")