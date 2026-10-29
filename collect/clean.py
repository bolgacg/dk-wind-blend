"""Raw downloads into compact Parquet tables in data/clean. Safe to rerun on partial data."""
import gzip, json, glob, urllib.request
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW, OUT = ROOT / "data" / "raw", ROOT / "data" / "clean"
OUT.mkdir(parents=True, exist_ok=True)
SHORT = {"wind_speed_100m": "ws100", "wind_direction_100m": "wd100", "wind_speed_10m": "ws10",
         "wind_direction_10m": "wd10", "wind_gusts_10m": "gust", "temperature_2m": "t2m", "pressure_msl": "mslp"}


def load(f):
    with gzip.open(f) as g:
        return json.load(g)


def forecasts(only_complete=True):
    """One Parquet per model (data/clean/fc/<model>.parquet), built one model at a time to fit the laptop's 3 GB."""
    (OUT / "fc").mkdir(exist_ok=True)
    for mdir in sorted((RAW / "openmeteo").iterdir()):
        files = sorted(mdir.glob("*.json.gz"))
        target = OUT / "fc" / f"{mdir.name}.parquet"
        if (only_complete and not (mdir / ".complete").exists() and len(files) < 102) or (target.exists() and target.stat().st_mtime > max(f.stat().st_mtime for f in files)):
            continue
        parts = []
        for f in files:
            d = load(f)
            if "hourly" not in d:
                continue
            h = d["hourly"]
            for lead in (1, 2):
                df = pd.DataFrame({SHORT[v]: np.array(h.get(f"{v}_previous_day{lead}") or [np.nan] * len(h["time"]), dtype="float32")
                                   for v in SHORT})
                df["time"] = pd.to_datetime(h["time"])
                df["point"], df["lead"] = f.name.rsplit("_", 1)[0], np.int8(lead)
                parts.append(df.dropna(subset=["ws10", "ws100"], how="all"))
        fc = pd.concat(parts, ignore_index=True)
        fc["hub_from_10m"] = fc["ws100"].isna()
        # Models without 100 m wind: power law from 10 m (alpha 0.11); the blend learns the rest.
        fc["ws100"] = fc["ws100"].fillna(fc["ws10"] * np.float32(10 ** 0.11))
        fc["wd100"] = fc["wd100"].fillna(fc["wd10"])
        fc["point"] = fc["point"].astype("category")
        fc.to_parquet(target, compression="zstd")
        print("forecasts", mdir.name, fc.shape, f"{target.stat().st_size // 1024} KB", flush=True)
        del parts, fc


def era5():
    parts = []
    for f in sorted(glob.glob(str(RAW / "era5" / "*.json.gz"))):
        h = load(f)["hourly"]
        df = pd.DataFrame({SHORT[v]: h[v] for v in SHORT if v in h})
        df["time"] = pd.to_datetime(h["time"]); df["point"] = Path(f).name.rsplit("_", 1)[0]
        parts.append(df)
    e = pd.concat(parts, ignore_index=True).drop_duplicates(["time", "point"])
    e.to_parquet(OUT / "era5.parquet", compression="zstd"); print("era5", e.shape)


def dmi():
    st = load(RAW / "dmi" / "stations.json.gz")
    geo = json.load(urllib.request.urlopen(
        "https://opendataapi.dmi.dk/v2/metObs/collections/station/items?status=Active&limit=1000", timeout=60))
    coords = {}
    for ft in geo["features"]:
        coords.setdefault(ft["properties"]["stationId"], ft["geometry"]["coordinates"])
    pd.DataFrame([{"station": s["stationId"], "name": s["name"], "lon": coords.get(s["stationId"], [None, None])[0],
                   "lat": coords.get(s["stationId"], [None, None])[1]} for s in st]).to_parquet(OUT / "stations.parquet")
    names = {"wind_speed_past1h": "ws", "wind_dir_past1h": "wd", "wind_gust_always_past1h": "gust", "pressure_at_sea": "mslp"}
    frames = []
    for par, short in names.items():
        rows = []
        for f in glob.glob(str(RAW / "dmi" / par / "*.json.gz")):
            sid = Path(f).name.split("_")[0]
            rows += [(sid, t, v) for t, v in load(f)]
        df = pd.DataFrame(rows, columns=["station", "time", short])
        df["time"] = pd.to_datetime(df["time"]).dt.tz_localize(None)
        frames.append(df.drop_duplicates(["station", "time"]).set_index(["station", "time"]))
    o = pd.concat(frames, axis=1).reset_index()
    o.to_parquet(OUT / "obs.parquet", compression="zstd"); print("obs", o.shape)


def energinet():
    def recs(ds):
        r = []
        for f in sorted(glob.glob(str(RAW / "energinet" / ds / "*.json.gz"))):
            r += load(f)["records"]
        return pd.DataFrame(r)
    p = recs("ProductionConsumptionSettlement")
    p = p[p["PriceArea"].isin(["DK1", "DK2"])]
    prod = pd.DataFrame({"time": pd.to_datetime(p["HourUTC"]), "area": p["PriceArea"],
                         "offshore": p["OffshoreWindLt100MW_MWh"].fillna(0) + p["OffshoreWindGe100MW_MWh"].fillna(0),
                         "onshore": p["OnshoreWindLt50kW_MWh"].fillna(0) + p["OnshoreWindGe50kW_MWh"].fillna(0)})
    prod = prod.groupby(["time", "area"], as_index=False).sum()
    prod["wind"] = prod["offshore"] + prod["onshore"]
    prod.to_parquet(OUT / "production.parquet"); print("production", prod.shape)
    fh = recs("Forecasts_Hour")
    fh = fh[fh["ForecastType"].isin(["Offshore Wind", "Onshore Wind"])]
    efc = fh.pivot_table(index=[pd.to_datetime(fh["HourUTC"]), "PriceArea"], columns="ForecastType",
                         values=["ForecastDayAhead", "ForecastIntraday", "Forecast1Hour"], aggfunc="last")
    efc.columns = [f"{a}_{b.split()[0].lower()}" for a, b in efc.columns]
    efc = efc.reset_index().rename(columns={"HourUTC": "time", "PriceArea": "area"})
    efc.to_parquet(OUT / "energinet_forecasts.parquet"); print("energinet forecasts", efc.shape)
    for ds in ["ImbalancePrice", "RegulatingBalancePowerdata", "DayAheadPrices", "ElspotPrices"]:
        d = recs(ds)
        if len(d):
            d.to_parquet(OUT / f"{ds}.parquet"); print(ds, d.shape)


if __name__ == "__main__":
    import sys
    for step in (sys.argv[1:] or ["forecasts", "era5", "dmi", "energinet"]):
        globals()[step]()
