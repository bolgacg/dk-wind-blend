"""Build the site's data files (site/data/*.json) and the PDF report's figures (report/fig_*.png) from results."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
C, R = ROOT / "data" / "clean", ROOT / "experiments" / "results"
D, F = ROOT / "site" / "data", ROOT / "report"
D.mkdir(parents=True, exist_ok=True); F.mkdir(exist_ok=True)
sys.path.insert(0, str(ROOT / "collect")); from dl import POINTS
BBOX = (5.0, 53.0, 17.0, 59.5)
NAMES = {"ecmwf_ifs025": "ECMWF IFS", "ecmwf_aifs025_single": "ECMWF AIFS (AI)", "icon_eu": "DWD ICON-EU",
         "icon_d2": "DWD ICON-D2", "gfs_global": "NOAA GFS", "ukmo_global_deterministic_10km": "UK Met Office",
         "dmi_harmonie_arome_europe": "DMI HARMONIE", "knmi_harmonie_arome_europe": "KNMI HARMONIE", "metno_nordic": "MET Nordic"}
INK, MUTED, ACCENT, AI = "#1f2933", "#7b8794", "#1d5fa8", "#c2410c"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK})


def clip_ring(ring, x0, y0, x1, y1):  # Sutherland-Hodgman against the rectangle
    def clip(pts, inside, cut):
        out = []
        for i in range(len(pts)):
            a, b = pts[i - 1], pts[i]
            if inside(b):
                if not inside(a): out.append(cut(a, b))
                out.append(b)
            elif inside(a):
                out.append(cut(a, b))
        return out
    def xcut(x):
        return lambda a, b: (x, a[1] + (b[1] - a[1]) * (x - a[0]) / (b[0] - a[0]))
    def ycut(y):
        return lambda a, b: (a[0] + (b[0] - a[0]) * (y - a[1]) / (b[1] - a[1]), y)
    p = [tuple(q) for q in ring]
    for inside, cut in [(lambda q: q[0] >= x0, xcut(x0)), (lambda q: q[0] <= x1, xcut(x1)),
                        (lambda q: q[1] >= y0, ycut(y0)), (lambda q: q[1] <= y1, ycut(y1))]:
        if not p: break
        p = clip(p, inside, cut)
    return p


def coast():
    g = json.load(open(ROOT / "data" / "raw" / "geo" / "ne_10m_land.geojson"))
    rings = []
    for ft in g["features"]:
        geom = ft["geometry"]; polys = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
        for poly in polys:
            r = clip_ring(poly[0], *BBOX)
            if len(r) > 3:
                rings.append([[round(x, 3), round(y, 3)] for x, y in r])
    json.dump(rings, open(D / "coast.json", "w"), separators=(",", ":"))
    return rings


def storms():
    out = {}
    for f in sorted((C / "storm").glob("*.npz")):
        z = np.load(f); storm, src = f.stem.split("_", 1)
        r1 = lambda a: np.round(np.nan_to_num(a), 1).transpose(2, 0, 1).reshape(len(z["time"]), -1).tolist()
        out.setdefault(storm, {})[src] = dict(u=r1(z["u"]), v=r1(z["v"]), p=r1(z["mslp"]))
        out[storm]["meta"] = dict(time=[str(t) for t in z["time"]], lat=z["lat"].tolist(), lon=z["lon"].tolist())
    for storm, d in out.items():
        json.dump(d, open(D / f"storm_{storm}.json", "w"), separators=(",", ":"))
    return out


def station_scores():
    st = pd.read_parquet(C / "stations.parquet").dropna(subset=["lat"])
    pts = pd.DataFrame([(k, *v) for k, v in POINTS.items()], columns=["point", "plat", "plon"])
    pairs = []
    for s in st.itertuples():
        dist = 111 * np.sqrt((s.lat - pts.plat) ** 2 + ((s.lon - pts.plon) * np.cos(np.radians(s.lat))) ** 2)
        i = int(dist.values.argmin())
        if dist.iloc[i] < 30: pairs.append((s.station, s.name, s.lat, s.lon, pts.point.iloc[i]))
    pairs = pd.DataFrame(pairs, columns=["station", "name", "lat", "lon", "point"])
    obs = pd.read_parquet(C / "obs.parquet", columns=["station", "time", "ws"]).merge(pairs[["station", "point"]], on="station")
    obs = obs[obs.time >= "2025-03-01"]
    rows = []
    for f in sorted((C / "fc").glob("*.parquet")):
        d = pd.read_parquet(f, columns=["time", "point", "lead", "ws10"]); d = d[d.lead == 1]; d["point"] = d.point.astype(str)
        m = obs.merge(d, on=["time", "point"]).dropna()
        g = m.assign(e=(m.ws10 - m.ws).abs()).groupby("station").e.mean()
        rows += [(s, f.stem, e) for s, e in g.items()]
    sc = pd.DataFrame(rows, columns=["station", "model", "mae"])
    best = sc.loc[sc.groupby("station").mae.idxmin()].merge(pairs, on="station")
    return best, sc


def main():
    rings = coast(); st = storms()
    era = pd.read_csv(R / "scorecard_era5.csv"); era = era[(era.kind == "all") & (era.lead == 1)].set_index("model")
    sta = pd.read_csv(R / "scorecard_stations.csv").set_index("model") if (R / "scorecard_stations.csv").exists() else None
    best, sc = station_scores()
    if sta is None:
        sta = sc.groupby("model").mae.mean().rename("rmse").to_frame()
    power = pd.read_csv(R / "power.csv"); gale = pd.read_csv(R / "gale.csv"); emos = pd.read_csv(R / "emos.csv")
    drn = pd.read_csv(R / "drn_bqn.csv")
    # Easter timeline
    pr = pd.read_parquet(C / "DayAheadPrices.parquet"); pr["time"] = pd.to_datetime(pr.TimeUTC).dt.floor("h")
    price = pr.groupby(["time", "PriceArea"]).DayAheadPriceEUR.mean()
    tl = {}
    for area in ("DK1", "DK2"):
        t = pd.read_parquet(R / f"power_test_{area}.parquet").loc["2026-04-03":"2026-04-07"]
        tl[area] = dict(time=[str(x) for x in t.index], y=t.y.round(3).tolist(), blend=t["stack"].round(3).tolist(),
                        q10=t["q0.1"].round(3).tolist(), q90=t["q0.9"].round(3).tolist(),
                        aifs=t.p_ecmwf_aifs025_single.round(3).tolist(), ifs=t.p_ecmwf_ifs025.round(3).tolist(),
                        price=price.xs(area, level="PriceArea").reindex(t.index).round(1).tolist())
    # spread-skill deciles
    ss = {}
    for area in ("DK1", "DK2"):
        t = pd.read_parquet(R / f"power_test_{area}.parquet")
        q = pd.qcut(t.spread, 10, labels=False); e = (t["stack"] - t.y).abs()
        ss[area] = (100 * e.groupby(q).mean()).round(2).tolist()
    summary = dict(
        names=NAMES, points=[dict(name=k, lat=v[0], lon=v[1]) for k, v in POINTS.items()],
        era5={m: round(float(r.rmse), 2) for m, r in era.iterrows()},
        stations={m: round(float(r.rmse), 2) for m, r in sta.iterrows()},
        station_best=best[["name", "lat", "lon", "model", "mae"]].round(3).to_dict("records"),
        power={a: power[(power.area == a) & power.nmae_pct.notna()][["forecast", "nmae_pct"]].to_dict("records") for a in ("DK1", "DK2")},
        gale=gale.to_dict("records"), emos=emos.to_dict("records"), drn=drn.to_dict("records"),
        timeline=tl, spread_skill=ss)
    json.dump(summary, open(D / "summary.json", "w"), separators=(",", ":"), default=float)
    figures(rings, st, summary, best)
    print("site data and figures written")


def draw_coast(ax, rings):
    for r in rings:
        a = np.array(r); ax.fill(a[:, 0], a[:, 1], color="#f4f1ea", zorder=1); ax.plot(a[:, 0], a[:, 1], color="#3e4c59", lw=0.7, zorder=6)
    ax.set_xlim(BBOX[0] + 1, BBOX[2] - 1); ax.set_ylim(BBOX[1] + 0.5, BBOX[3] - 1); ax.set_aspect(1 / np.cos(np.radians(56)))
    ax.set_xticks([]); ax.set_yticks([]); ax.set_facecolor("#e6eef5")
    for s in ax.spines.values(): s.set_visible(False)


def figures(rings, st, S, best):
    # 1. Storm maps: ERA5 vs AIFS vs IFS vs KNMI at the Easter peak (05 Apr 06 UTC)
    e = st["easter2026"]; meta = e["meta"]; i = meta["time"].index("2026-04-05T06:00")
    lon, lat = np.array(meta["lon"]), np.array(meta["lat"]); X, Y = np.meshgrid(lon, lat)
    fig, axs = plt.subplots(2, 2, figsize=(9, 7.6))
    for ax, (src, title) in zip(axs.ravel(), [("era5", "What happened (ERA5)"), ("aifs", "AIFS, day before"), ("ifs", "IFS, day before"), ("knmi", "KNMI HARMONIE, day before")]):
        u = np.array(e[src]["u"][i]).reshape(X.shape); v = np.array(e[src]["v"][i]).reshape(X.shape); p = np.array(e[src]["p"][i]).reshape(X.shape)
        ws = np.hypot(u, v); draw_coast(ax, rings)
        cf = ax.contourf(X, Y, ws, levels=np.arange(0, 30, 2), cmap="YlGnBu", alpha=0.75, zorder=2)
        cs = ax.contour(X, Y, p, levels=np.arange(950, 1040, 4), colors=INK, linewidths=0.6, zorder=4)
        ax.clabel(cs, fontsize=6, fmt="%d")
        ax.streamplot(X, Y, u, v, color="#243b53", density=1.1, linewidth=0.6, arrowsize=0.6, zorder=5)
        ax.set_title(title, fontsize=9, color=INK)
    cb = fig.colorbar(cf, ax=axs.ravel().tolist(), shrink=0.7, pad=0.02); cb.set_label("100 m wind (m/s)")
    fig.savefig(F / "fig_storm_maps.png", dpi=200, bbox_inches="tight"); plt.close(fig)
    # 2. Two truths slope chart
    ms = [m for m in S["era5"] if m in S["stations"]]
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    ra = {m: k for k, m in enumerate(sorted(ms, key=lambda m: S["era5"][m]))}; rs = {m: k for k, m in enumerate(sorted(ms, key=lambda m: S["stations"][m]))}
    for m in ms:
        c = AI if "aifs" in m else (ACCENT if m == "icon_eu" else "#bcccdc")
        ax.plot([0, 1], [ra[m], rs[m]], color=c, lw=2.2 if c != "#bcccdc" else 1.2, marker="o")
        ax.text(-0.04, ra[m], f"{NAMES[m]}  {S['era5'][m]:.2f}", ha="right", va="center", fontsize=8, color=INK)
        ax.text(1.04, rs[m], f"{S['stations'][m]:.2f}  {NAMES[m]}", ha="left", va="center", fontsize=8, color=INK)
    ax.set_xlim(-0.9, 1.9); ax.invert_yaxis(); ax.axis("off")
    ax.text(-0.3, -1.1, "Ranked against ERA5\n(100 m, smooth reanalysis)", ha="center", fontsize=8, color=MUTED)
    ax.text(1.3, -1.1, "Ranked against 36 DMI stations\n(10 m, real anemometers)", ha="center", fontsize=8, color=MUTED)
    fig.savefig(F / "fig_two_truths.png", dpi=200, bbox_inches="tight"); plt.close(fig)
    # 3. Station map: best model per station
    fig, ax = plt.subplots(figsize=(5.2, 4.2)); draw_coast(ax, rings)
    models_here = sorted(best.model.unique()); cmap = plt.get_cmap("tab10")
    for k, m in enumerate(models_here):
        b = best[best.model == m]; ax.scatter(b.lon, b.lat, s=36, color=cmap(k), edgecolor="white", lw=0.6, zorder=6, label=NAMES[m])
    ax.legend(fontsize=7, frameon=False, loc="upper right")
    fig.savefig(F / "fig_station_map.png", dpi=200, bbox_inches="tight"); plt.close(fig)
    # 4. Easter timeline, DK2
    for area in ("DK1", "DK2"):
        t = S["timeline"][area]; tt = pd.to_datetime(t["time"])
        fig, ax = plt.subplots(figsize=(7.5, 2.8))
        neg = np.array([p is not None and p == p and p <= 0 for p in t["price"]])
        for k in np.where(neg)[0]:
            ax.axvspan(tt[k], tt[k] + pd.Timedelta("1h"), color="#fde8d8", lw=0, zorder=0)
        ax.fill_between(tt, np.array(t["q10"]) * 100, np.array(t["q90"]) * 100, color=ACCENT, alpha=0.15, lw=0, label="blend 10 to 90% band")
        ax.plot(tt, np.array(t["blend"]) * 100, color=ACCENT, lw=1.6, label="blend of 8 models")
        ax.plot(tt, np.array(t["aifs"]) * 100, color=AI, lw=1, ls="--", label="AIFS alone")
        ax.plot(tt, np.array(t["y"]) * 100, color=INK, lw=1.8, label="actual output")
        ax.set_ylabel("% of capacity"); ax.legend(fontsize=7, frameon=False, ncol=4, loc="upper left")
        ax.text(0.99, 0.02, "shaded: negative day-ahead price", transform=ax.transAxes, ha="right", fontsize=7, color=AI)
        fig.savefig(F / f"fig_easter_{area}.png", dpi=200, bbox_inches="tight"); plt.close(fig)
    # 5. Spread-skill
    fig, ax = plt.subplots(figsize=(4.6, 2.8))
    for area, c in (("DK1", ACCENT), ("DK2", INK)):
        ax.plot(range(1, 11), S["spread_skill"][area], marker="o", color=c, label=area)
    ax.set_xlabel("model disagreement, decile (1 = agree most)"); ax.set_ylabel("blend error, % of capacity"); ax.legend(frameon=False)
    fig.savefig(F / "fig_spread_skill.png", dpi=200, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    main()
