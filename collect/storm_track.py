"""Sea level pressure on a wide 1.5 degree grid (50 to 65 N, 6 W to 24 E, 231 points) so the low centre can be tracked
outside the Danish storm box. ERA5 = what happened; IFS, AIFS, KNMI HARMONIE = forecast issued the day before."""
import json, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dl import get
from storm_grid import STORMS, SOURCES

OUT = Path(__file__).resolve().parent.parent / "data" / "clean" / "track"
LATS, LONS = np.arange(50.0, 65.01, 1.5), np.arange(-6.0, 24.01, 1.5)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    grid = [(la, lo) for la in LATS for lo in LONS]
    for storm, (a, b) in STORMS.items():
        for src, model in SOURCES.items():
            if storm == "jan2025" and src == "aifs":
                continue
            f = OUT / f"{storm}_{src}.npz"
            if f.exists():
                continue
            p, times = [], None
            for i in range(0, len(grid), 60):
                chunk = grid[i:i + 60]
                la, lo = ",".join(f"{x[0]:.2f}" for x in chunk), ",".join(f"{x[1]:.2f}" for x in chunk)
                if src == "era5":
                    url = (f"https://archive-api.open-meteo.com/v1/archive?latitude={la}&longitude={lo}&start_date={a}&end_date={b}"
                           "&hourly=pressure_msl&models=era5&timezone=GMT"); key = "pressure_msl"
                else:
                    url = (f"https://previous-runs-api.open-meteo.com/v1/forecast?latitude={la}&longitude={lo}&start_date={a}&end_date={b}"
                           f"&hourly=pressure_msl_previous_day1&models={model}&timezone=GMT"); key = "pressure_msl_previous_day1"
                d = json.loads(get(url)); d = d if isinstance(d, list) else [d]
                for loc in d:
                    times = loc["hourly"]["time"]; p.append(loc["hourly"][key])
                time.sleep(8)
            p = np.array([[np.nan if v is None else v for v in r] for r in p], dtype=float).reshape(len(LATS), len(LONS), len(times))
            np.savez_compressed(f, time=np.array(times), lat=LATS, lon=LONS, mslp=p)
            print(storm, src, p.shape, f"nan {np.isnan(p).mean():.2f}", flush=True)
            time.sleep(20)


if __name__ == "__main__":
    main()
