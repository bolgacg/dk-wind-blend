"""Save today's IFS ENS and AIFS ENS point forecasts for Danish wind sites (Open-Meteo Ensemble API).
One gzipped JSON per model per day; skips files that already exist, so it is safe to run many times a day."""
import gzip, json, sys, time, urllib.request
from datetime import datetime, timezone
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "data" / "ens"
SITES = {  # name: (lat, lon)
    "horns_rev": (55.53, 7.91), "thor": (56.35, 7.70), "anholt": (56.60, 11.21),
    "kriegers_flak": (55.03, 12.92), "w_jutland": (56.10, 8.40), "n_jutland": (57.10, 9.30),
    "lolland": (54.80, 11.40), "zealand": (55.50, 11.80),
}
VARS = "wind_speed_100m,wind_direction_100m,wind_speed_10m,wind_gusts_10m,temperature_2m,pressure_msl"
MODELS = ["ecmwf_ifs025", "ecmwf_aifs025"]

def main():
    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    OUT.mkdir(parents=True, exist_ok=True)
    lats = ",".join(str(v[0]) for v in SITES.values()); lons = ",".join(str(v[1]) for v in SITES.values())
    failed = 0
    for m in MODELS:
        f = OUT / f"{m}_{day}.json.gz"
        if f.exists():
            continue
        url = (f"https://ensemble-api.open-meteo.com/v1/ensemble?latitude={lats}&longitude={lons}"
               f"&hourly={VARS}&models={m}&forecast_days=7&wind_speed_unit=ms&timezone=GMT")
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                data = json.load(r)
        except Exception as e:
            print(f"{day} {m} FAILED {e}", file=sys.stderr); failed += 1; continue
        payload = {"fetched_utc": datetime.now(timezone.utc).isoformat(), "sites": list(SITES), "data": data}
        with gzip.open(f, "wt") as g:
            json.dump(payload, g)
        print(f"{day} {m} saved {f.stat().st_size // 1024} KB")
        time.sleep(2)
    sys.exit(1 if failed else 0)

if __name__ == "__main__":
    main()
