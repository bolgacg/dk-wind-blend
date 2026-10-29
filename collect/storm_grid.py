"""Gridded 100 m wind and pressure over Denmark for the storm pages (laptop, Wi-Fi). 0.5 degree grid, 231 points.
ERA5 = what happened; IFS, AIFS, KNMI HARMONIE = the forecast issued the day before (Open-Meteo previous_day1)."""
import json, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dl import get

OUT = Path(__file__).resolve().parent.parent / "data" / "clean" / "storm"
LATS, LONS = np.arange(53.5, 58.51, 0.5), np.arange(6.0, 16.01, 0.5)
STORMS = {"easter2026": ("2026-04-04", "2026-04-06"), "jan2025": ("2025-01-06", "2025-01-08")}
SOURCES = {"era5": "era5", "ifs": "ecmwf_ifs025", "aifs": "ecmwf_aifs025_single", "knmi": "knmi_harmonie_arome_europe"}


def fetch(src, model, a, b, lat, lon):
    if src == "era5":
        url = (f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&start_date={a}&end_date={b}"
               "&hourly=wind_speed_100m,wind_direction_100m,pressure_msl&models=era5&wind_speed_unit=ms&timezone=GMT")
        keys = ("wind_speed_100m", "wind_direction_100m", "pressure_msl")
    else:
        url = (f"https://previous-runs-api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&start_date={a}&end_date={b}"
               "&hourly=wind_speed_100m_previous_day1,wind_direction_100m_previous_day1,pressure_msl_previous_day1"
               f"&models={model}&wind_speed_unit=ms&timezone=GMT")
        keys = ("wind_speed_100m_previous_day1", "wind_direction_100m_previous_day1", "pressure_msl_previous_day1")
    return json.loads(get(url)), keys


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    grid = [(la, lo) for la in LATS for lo in LONS]
    for storm, (a, b) in STORMS.items():
        for src, model in SOURCES.items():
            f = OUT / f"{storm}_{src}.npz"
            if f.exists():
                continue
            ws, wd, p, times = [], [], [], None
            for i in range(0, len(grid), 60):
                chunk = grid[i:i + 60]
                d, keys = fetch(src, model, a, b, ",".join(f"{x[0]:.2f}" for x in chunk), ",".join(f"{x[1]:.2f}" for x in chunk))
                d = d if isinstance(d, list) else [d]
                for loc in d:
                    h = loc["hourly"]; times = h["time"]
                    ws.append(h[keys[0]]); wd.append(h[keys[1]]); p.append(h[keys[2]])
                time.sleep(3)
            shp = (len(LATS), len(LONS), len(times))
            ws = np.array(ws, dtype=float).reshape(shp); wd = np.array(wd, dtype=float).reshape(shp)
            p = np.array(p, dtype=float).reshape(shp)
            u, v = -ws * np.sin(np.radians(wd)), -ws * np.cos(np.radians(wd))
            np.savez_compressed(f, time=np.array(times), lat=LATS, lon=LONS, ws=ws, u=u, v=v, mslp=p)
            print(storm, src, ws.shape, f"nan {np.isnan(ws).mean():.2f}", flush=True)


if __name__ == "__main__":
    main()
