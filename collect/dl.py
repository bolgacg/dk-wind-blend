"""Bulk download for dk-wind-blend, run on the laptop over Wi-Fi. Resumable: every chunk is its own
gzipped file and existing files are skipped. Usage: python3 collect/dl.py energinet|dmi|openmeteo|era5|all"""
import gzip, json, re, sys, time, urllib.error, urllib.parse, urllib.request
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "data" / "raw"
START, END = date(2024, 2, 10), date(2026, 10, 27)

# About 35 points: offshore farms, plus onshore cluster centres in DK1 and DK2 (hand-picked, refine with ENS register).
POINTS = {
    "horns_rev1": (55.48, 7.84), "horns_rev2": (55.60, 7.58), "horns_rev3": (55.70, 7.68), "thor": (56.35, 7.70),
    "vesterhav_nord": (56.45, 8.08), "vesterhav_syd": (55.95, 8.05), "anholt": (56.60, 11.21),
    "kriegers_flak": (55.03, 12.92), "rodsand": (54.56, 11.62), "samso": (55.72, 10.58), "sprogo": (55.33, 10.97),
    "frederikshavn": (57.44, 10.56), "middelgrunden": (55.69, 12.67),
    "ringkobing": (56.05, 8.35), "lemvig": (56.55, 8.30), "thisted": (56.95, 8.70), "hjorring": (57.45, 9.98),
    "herning": (56.25, 8.80), "varde": (55.62, 8.48), "tonder": (54.95, 8.86), "vesthimmerland": (56.80, 9.40),
    "djursland": (56.40, 10.70), "aabenraa": (55.04, 9.42), "skive": (56.57, 9.03), "jammerbugt": (57.15, 9.55),
    "fyn": (55.30, 10.40), "viborg": (56.45, 9.40), "horsens": (55.85, 9.85),
    "lolland": (54.80, 11.40), "falster": (54.80, 11.90), "w_zealand": (55.45, 11.25), "s_zealand": (55.25, 11.75),
    "n_zealand": (55.95, 12.00), "bornholm": (55.10, 14.90),
}
OM_MODELS = ["ecmwf_ifs025", "ecmwf_aifs025_single", "icon_eu", "icon_d2", "gfs_global",
             "ukmo_global_deterministic_10km", "dmi_harmonie_arome_europe", "knmi_harmonie_arome_europe", "metno_nordic"]
OM_VARS = ["wind_speed_100m", "wind_direction_100m", "wind_speed_10m", "wind_direction_10m", "wind_gusts_10m",
           "temperature_2m", "pressure_msl"]


def get(url, tries=8):
    i = 0
    while i < tries:
        i += 1
        try:
            req = urllib.request.Request(url, headers={"Accept-Encoding": "gzip", "User-Agent": "dk-wind-blend research"})
            with urllib.request.urlopen(req, timeout=180) as r:
                raw = r.read()
                return gzip.decompress(raw) if r.headers.get("Content-Encoding") == "gzip" else raw
        except urllib.error.HTTPError as e:
            raw = e.read()
            try:
                raw = gzip.decompress(raw)
            except Exception:
                pass
            body = raw.decode("utf-8", "replace")[:300]
            if e.code == 429:
                tries += 1  # rate limits never count as a failed try: keep waiting
                if "previous-runs" in url or "archive-api" in url:
                    print(f"  429 open-meteo, waiting 900s: {body[:120]}", flush=True); time.sleep(900); continue
                m = re.search(r"(\d+) seconds", body)
                wait = int(m.group(1)) + 5 if m else 60 * (i + 1)
                if "daily" in body.lower() or "Daily" in body:
                    wait = 3600
                print(f"  429, waiting {wait}s: {body[:120]}", flush=True)
                time.sleep(wait)
                continue
            if e.code == 400:
                raise ValueError(body)
            print(f"  HTTP {e.code}: {body[:120]}", flush=True)
        except Exception as e:
            print(f"  error {e}", flush=True)
        time.sleep(15 * (i + 1))
    raise RuntimeError(f"gave up on {url}")


def save(path, raw):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with gzip.open(tmp, "wb") as g:
        g.write(raw)
    tmp.rename(path)


def months():
    d = START.replace(day=1)
    while d <= END:
        nxt = (d + timedelta(days=32)).replace(day=1)
        yield d, nxt
        d = nxt


def energinet():
    for ds in ["ProductionConsumptionSettlement", "Forecasts_Hour", "ImbalancePrice", "RegulatingBalancePowerdata",
               "DayAheadPrices", "ElspotPrices"]:
        for a, b in months():
            f = ROOT / "energinet" / ds / f"{a:%Y-%m}.json.gz"
            if f.exists():
                continue
            try:
                raw = get(f"https://api.energidataservice.dk/dataset/{ds}?start={a}&end={b}&limit=0")
            except ValueError as e:
                print(f"{ds} {a:%Y-%m} skipped: {e}", flush=True)
                continue
            save(f, raw)
            print(f"energinet {ds} {a:%Y-%m} {len(raw) // 1024} KB", flush=True)
            time.sleep(3)
    f = ROOT / "energinet" / "CapacityPerMunicipality.json.gz"
    if not f.exists():
        save(f, get("https://api.energidataservice.dk/dataset/CapacityPerMunicipality?limit=0"))


def dmi():
    st = json.loads(get("https://opendataapi.dmi.dk/v2/metObs/collections/station/items?status=Active&limit=1000"))
    keep = [f["properties"] for f in st["features"] if f["properties"].get("country") == "DNK"
            and f["properties"].get("type") == "Synop"
            and "wind_speed_past1h" in (f["properties"].get("parameterId") or [])]
    seen, stations = set(), []
    for p in keep:
        if p["stationId"] not in seen:
            seen.add(p["stationId"]); stations.append(p)
    save(ROOT / "dmi" / "stations.json.gz", json.dumps(stations).encode())
    print(f"dmi: {len(stations)} synop wind stations", flush=True)
    params = ["wind_speed_past1h", "wind_dir_past1h", "wind_gust_always_past1h", "pressure_at_sea"]
    for p in stations:
        for par in params:
            for y in range(START.year, END.year + 1):
                f = ROOT / "dmi" / par / f"{p['stationId']}_{y}.json.gz"
                if f.exists():
                    continue
                a, b = max(START, date(y, 1, 1)), min(END, date(y, 12, 31))
                q = urllib.parse.urlencode({"stationId": p["stationId"], "parameterId": par, "limit": 300000,
                                            "datetime": f"{a}T00:00:00Z/{b}T23:59:59Z"})
                raw = get(f"https://opendataapi.dmi.dk/v2/metObs/collections/observation/items?{q}")
                # keep only time and value to save disk
                feats = json.loads(raw).get("features", [])
                slim = [(x["properties"]["observed"], x["properties"]["value"]) for x in feats]
                save(f, json.dumps(slim).encode())
                print(f"dmi {p['stationId']} {par} {y} {len(slim)} rows", flush=True)
                time.sleep(1)


def openmeteo():
    full = ",".join(f"{v}_previous_day{d}" for d in (1, 2) for v in OM_VARS)
    # After the 29 Oct rate-limit stall: models after IFS and AIFS fetch day-ahead only, from 2025 (the AIFS window).
    light = ",".join(f"{v}_previous_day1" for v in OM_VARS)
    order = ["ecmwf_ifs025", "ecmwf_aifs025_single", "icon_eu", "dmi_harmonie_arome_europe", "gfs_global",
             "knmi_harmonie_arome_europe", "ukmo_global_deterministic_10km", "metno_nordic", "icon_d2"]
    for model in order:
        is_light = model not in ("ecmwf_ifs025", "ecmwf_aifs025_single")
        hourly = light if is_light else full
        for name, (lat, lon) in POINTS.items():
            for y in range(2025 if is_light else START.year, END.year + 1):
                f = ROOT / "openmeteo" / model / f"{name}_{y}.json.gz"
                if f.exists():
                    continue
                a, b = max(START, date(y, 1, 1)), min(END, date(y, 12, 31))
                url = (f"https://previous-runs-api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
                       f"&start_date={a}&end_date={b}&hourly={hourly}&models={model}&wind_speed_unit=ms&timezone=GMT")
                try:
                    raw = get(url)
                except ValueError as e:
                    print(f"openmeteo {model} {name} {y} skipped: {e}", flush=True)
                    save(f, json.dumps({"skipped": str(e)}).encode())
                    continue
                save(f, raw)
                print(f"openmeteo {model} {name} {y} {len(raw) // 1024} KB", flush=True)
                time.sleep(8)
        (ROOT / "openmeteo" / model / ".complete").touch()


def era5():
    hourly = ",".join(OM_VARS)
    for name, (lat, lon) in POINTS.items():
        for y in range(START.year, END.year + 1):
            f = ROOT / "era5" / f"{name}_{y}.json.gz"
            if f.exists():
                continue
            a, b = max(START, date(y, 1, 1)), min(END, date(y, 12, 31))
            raw = get(f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&start_date={a}"
                      f"&end_date={b}&hourly={hourly}&models=era5&wind_speed_unit=ms&timezone=GMT")
            save(f, raw)
            print(f"era5 {name} {y} {len(raw) // 1024} KB", flush=True)
            time.sleep(4)


if __name__ == "__main__":
    jobs = {"energinet": energinet, "dmi": dmi, "openmeteo": openmeteo, "era5": era5}
    for j in (jobs if sys.argv[1] == "all" else [sys.argv[1]]):
        jobs[j]()
    print("DONE", sys.argv[1], flush=True)
