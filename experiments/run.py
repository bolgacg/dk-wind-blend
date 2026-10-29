"""Experiments on gene (16 GB). Usage: python run.py scorecard|power|emos|cases|all
Reads data/clean, writes experiments/results/<name>.csv and appends key lines to experiments/results/RESULTS.md.
Train 2025-03-01 to 2025-12-31, test 2026-01-01 onward (AIFS archive starts 25 Feb 2025)."""
import sys, os, time
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats, optimize

ROOT = Path(__file__).resolve().parent.parent
C, R = ROOT / "data" / "clean", ROOT / "experiments" / "results"
R.mkdir(parents=True, exist_ok=True)
TRAIN = ("2025-03-01", "2026-01-01"); TEST = ("2026-01-01", "2027-01-01")
DK2 = {"kriegers_flak", "middelgrunden", "lolland", "falster", "w_zealand", "s_zealand", "n_zealand", "bornholm",
       "rodsand", "sprogo"}
OFFSHORE = {"horns_rev1", "horns_rev2", "horns_rev3", "thor", "vesterhav_nord", "vesterhav_syd", "anholt",
            "kriegers_flak", "rodsand", "samso", "sprogo", "frederikshavn", "middelgrunden"}


def note(line):
    with open(R / "RESULTS.md", "a") as f:
        f.write(f"- {time.strftime('%d %b %H:%M')} {line}\n")
    print(line, flush=True)


def models():
    return sorted(p.stem for p in (C / "fc").glob("*.parquet"))


def fc(model, lead=1, cols=("ws100", "wd100", "ws10", "gust", "t2m", "mslp")):
    d = pd.read_parquet(C / "fc" / f"{model}.parquet", columns=["time", "point", "lead", *cols])
    d = d[d["lead"] == lead].drop(columns="lead")
    d["point"] = d["point"].astype(str)
    return d


def crps_normal(y, mu, sd):
    z = (y - mu) / sd
    return sd * (z * (2 * stats.norm.cdf(z) - 1) + 2 * stats.norm.pdf(z) - 1 / np.sqrt(np.pi))


def crps_ens(y, ens):  # empirical CRPS, ens shape (n, m)
    m = ens.shape[1]
    t1 = np.abs(ens - y[:, None]).mean(1)
    s = np.sort(ens, 1); i = np.arange(1, m + 1)
    t2 = ((2 * i - m - 1) * s).sum(1) / (m * m)
    return t1 - t2


def crps_quantiles(y, q, taus):  # approx CRPS from quantile forecasts = 2 * mean pinball
    e = y[:, None] - q
    return 2 * np.mean(np.maximum(taus * e, (taus - 1) * e), axis=1)


# 1. Scorecard: every model against ERA5 100 m and DMI 10 m stations.
def scorecard():
    era = pd.read_parquet(C / "era5.parquet", columns=["time", "point", "ws100"]).rename(columns={"ws100": "era"})
    rows = []
    for m in models():
        for lead in (1, 2):
            d = fc(m, lead, ("ws100",)).merge(era, on=["time", "point"])
            d = d[d["time"] >= "2025-03-01"]  # common window for all models
            d["kind"] = np.where(d["point"].isin(OFFSHORE), "offshore", "onshore")
            for kind, g in [("all", d), *d.groupby("kind")]:
                e = g["ws100"] - g["era"]
                rows.append(dict(model=m, lead=lead, kind=kind, n=len(g), bias=e.mean(), mae=e.abs().mean(),
                                 rmse=np.sqrt((e ** 2).mean()), corr=g[["ws100", "era"]].corr().iloc[0, 1]))
    sc = pd.DataFrame(rows).round(3); sc.to_csv(R / "scorecard_era5.csv", index=False)
    a = sc[(sc.kind == "all") & (sc.lead == 1)].sort_values("rmse")
    note("SCORECARD vs ERA5 100 m, day-ahead, since Mar 2025 (RMSE m/s): " +
         ", ".join(f"{r.model} {r.rmse:.2f} (bias {r.bias:+.2f})" for r in a.itertuples()))
    # stations: 10 m wind, nearest model point within 30 km
    st = pd.read_parquet(C / "stations.parquet").dropna(subset=["lat"])
    sys.path.insert(0, str(ROOT / "collect")); from dl import POINTS
    pts = pd.DataFrame([(k, *v) for k, v in POINTS.items()], columns=["point", "plat", "plon"])
    def km(a1, o1, a2, o2):
        return 111 * np.sqrt((a1 - a2) ** 2 + ((o1 - o2) * np.cos(np.radians(a1))) ** 2)
    pairs = []
    for s in st.itertuples():
        dist = km(s.lat, s.lon, pts.plat.values, pts.plon.values); i = dist.argmin()
        if dist[i] < 30:
            pairs.append((s.station, pts.point.iloc[i], round(dist[i], 1)))
    pairs = pd.DataFrame(pairs, columns=["station", "point", "km"])
    obs = pd.read_parquet(C / "obs.parquet", columns=["station", "time", "ws"]).merge(pairs, on="station")
    obs = obs[obs["time"] >= "2025-03-01"]
    rows = []
    for m in models():
        d = fc(m, 1, ("ws10",)).merge(obs, on=["time", "point"]).dropna(subset=["ws10", "ws"])
        e = d["ws10"] - d["ws"]
        rows.append(dict(model=m, n=len(d), stations=d.station.nunique(), bias=e.mean(), mae=e.abs().mean(),
                         rmse=np.sqrt((e ** 2).mean()), corr=d[["ws10", "ws"]].corr().iloc[0, 1]))
    ss = pd.DataFrame(rows).round(3).sort_values("rmse"); ss.to_csv(R / "scorecard_stations.csv", index=False)
    note(f"SCORECARD vs {pairs.station.nunique()} DMI stations, 10 m day-ahead (RMSE m/s): " +
         ", ".join(f"{r.model} {r.rmse:.2f}" for r in ss.itertuples()))


# 2. Power: one gradient-boosting mapping per model and area; then blends, quantiles, spread-as-risk.
def power_frame(model, area):
    d = fc(model, 1, ("ws100", "wd100"))
    d = d[d["point"].isin(DK2) == (area == "DK2")]
    d["ws3"] = np.clip(d["ws100"], 0, 25) ** 3
    d["u"] = -d["ws100"] * np.sin(np.radians(d["wd100"])); d["v"] = -d["ws100"] * np.cos(np.radians(d["wd100"]))
    w = d.pivot_table(index="time", columns="point", values=["ws100", "ws3", "u", "v"])
    w.columns = [f"{model}|{a}|{b}" for a, b in w.columns]
    return w


def target(area):
    p = pd.read_parquet(C / "production.parquet")
    p = p[p.area == area].set_index("time").sort_index()
    cap = p["wind"].rolling("60D", min_periods=24).max()  # trailing capacity proxy (Thor commissioning)
    return (p["wind"] / cap).rename("cf"), cap


def gbm(q=None):
    from sklearn.ensemble import HistGradientBoostingRegressor
    kw = dict(max_iter=400, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=1.0)
    return HistGradientBoostingRegressor(loss="quantile", quantile=q, **kw) if q else HistGradientBoostingRegressor(**kw)


def power():
    taus = np.array([0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95])
    out = []
    efc = pd.read_parquet(C / "energinet_forecasts.parquet")
    for area in ("DK1", "DK2"):
        cf, cap = target(area)
        frames = {m: power_frame(m, area) for m in models()}
        common = cf.index
        for w in frames.values():
            common = common.intersection(w.dropna().index)
        tr = common[(common >= TRAIN[0]) & (common < TRAIN[1])]; te = common[common >= TEST[0]]
        cal = lambda X: X.assign(hour=X.index.hour, doy_s=np.sin(2 * np.pi * X.index.dayofyear / 365.25),
                                 doy_c=np.cos(2 * np.pi * X.index.dayofyear / 365.25))
        preds = {}
        for m, w in frames.items():
            X = cal(w.loc[common]); g = gbm().fit(X.loc[tr], cf.loc[tr])
            preds[m] = pd.Series(np.clip(g.predict(X.loc[te]), 0, 1.1), te)
        y = cf.loc[te]
        res = {m: (p - y).abs().mean() for m, p in preds.items()}
        res["blend_equal"] = (pd.concat(preds, axis=1).mean(1) - y).abs().mean()
        Xall = cal(pd.concat(frames.values(), axis=1).loc[common])
        g = gbm().fit(Xall.loc[tr], cf.loc[tr]); stack = pd.Series(np.clip(g.predict(Xall.loc[te]), 0, 1.1), te)
        res["blend_stacked"] = (stack - y).abs().mean()
        # Energinet day-ahead, recalibrated on train (their DK1 offshore misses Thor)
        e = efc[efc.area == area].set_index("time")
        e = (e["ForecastDayAhead_offshore"].fillna(0) + e["ForecastDayAhead_onshore"].fillna(0)) / cap
        e = e.reindex(common)
        ok_tr, ok_te = e.loc[tr].dropna().index, e.loc[te].dropna().index
        a, b = np.polyfit(e.loc[ok_tr], cf.loc[ok_tr], 1)
        res["energinet_dayahead_raw"] = (e.loc[ok_te] - y.loc[ok_te]).abs().mean()
        res["energinet_dayahead_recal"] = (a * e.loc[ok_te] + b - y.loc[ok_te]).abs().mean()
        for k, v in res.items():
            out.append(dict(area=area, forecast=k, nmae_pct=round(100 * v, 2), n_test=len(te)))
        # probabilistic: quantile GBM, IFS only vs all models
        ifs = cal(frames["ecmwf_ifs025"].loc[common])
        def qrun(X):
            Q = np.column_stack([gbm(t).fit(X.loc[tr], cf.loc[tr]).predict(X.loc[te]) for t in taus])
            return np.sort(Q, 1)
        Qi, Qa = qrun(ifs), qrun(Xall)
        for name, Q in [("quantile_ifs_only", Qi), ("quantile_all_models", Qa)]:
            cov = np.mean((y.values >= Q[:, 1]) & (y.values <= Q[:, 5]))
            out.append(dict(area=area, forecast=name, crps_pct=round(100 * crps_quantiles(y.values, Q, taus).mean(), 2),
                            cover_10_90=round(cov, 3), n_test=len(te)))
        # conformalized quantile regression (Romano et al. 2019): fit 10/90 on Mar-Oct 2025, calibrate on Nov-Dec 2025
        fit_i, cal_i = tr[tr < pd.Timestamp("2025-11-01")], tr[tr >= pd.Timestamp("2025-11-01")]
        lo, hi = (gbm(t).fit(Xall.loc[fit_i], cf.loc[fit_i]) for t in (0.1, 0.9))
        s = np.maximum(lo.predict(Xall.loc[cal_i]) - cf.loc[cal_i], cf.loc[cal_i] - hi.predict(Xall.loc[cal_i]))
        qhat = np.quantile(s, min(1, 0.8 * (1 + 1 / len(s))))
        L, H = lo.predict(Xall.loc[te]) - qhat, hi.predict(Xall.loc[te]) + qhat
        out.append(dict(area=area, forecast="conformal_10_90_all_models", cover_10_90=round(np.mean((y.values >= L) & (y.values <= H)), 3),
                        width_pct=round(100 * np.mean(H - L), 1), n_test=len(te)))
        note(f"POWER {area} conformal 10-90 band: coverage {out[-1]['cover_10_90']} (target 0.8), mean width {out[-1]['width_pct']}% of capacity")
        # model disagreement as a risk signal
        P = pd.concat(preds, axis=1); spread = P.std(1); err = (stack - y).abs()
        rho = stats.spearmanr(spread, err).correlation
        top = err[spread >= spread.quantile(0.9)].mean() / err[spread <= spread.quantile(0.5)].mean()
        out.append(dict(area=area, forecast="spread_vs_error", spearman=round(rho, 3), err_ratio_top10_vs_bottom50=round(top, 2)))
        pd.DataFrame({"y": y, "stack": stack, "spread": spread, **{f"p_{m}": p for m, p in preds.items()},
                      **{f"q{t}": Qa[:, i] for i, t in enumerate(taus)}}).to_parquet(R / f"power_test_{area}.parquet")
    df = pd.DataFrame(out); df.to_csv(R / "power.csv", index=False)
    for area in ("DK1", "DK2"):
        s = df[(df.area == area) & df.nmae_pct.notna()].sort_values("nmae_pct")
        note(f"POWER {area} test 2026 nMAE % of capacity: " + ", ".join(f"{r.forecast} {r.nmae_pct}" for r in s.itertuples()))
        q = df[(df.area == area) & df.crps_pct.notna()]
        note(f"POWER {area} probabilistic CRPS %: " + ", ".join(f"{r.forecast} {r.crps_pct} (10-90 cover {r.cover_10_90})" for r in q.itertuples()))
        sp = df[(df.area == area) & (df.forecast == "spread_vs_error")].iloc[0]
        note(f"POWER {area} model disagreement vs blend error: Spearman {sp.spearman}, error when spread is top 10% is {sp.err_ratio_top10_vs_bottom50}x the error when spread is below median")


# 3. EMOS at each point: multi-model mean and spread into a calibrated normal for ERA5 100 m wind.
def emos():
    era = pd.read_parquet(C / "era5.parquet", columns=["time", "point", "ws100"]).rename(columns={"ws100": "y"})
    W = None
    for m in models():
        d = fc(m, 1, ("ws100",)).rename(columns={"ws100": m})
        W = d if W is None else W.merge(d, on=["time", "point"])
    W = W.merge(era, on=["time", "point"]).dropna()
    ms = models(); ens = W[ms].values
    W["mean"], W["sd"] = ens.mean(1), ens.std(1)
    tr = W[(W.time >= TRAIN[0]) & (W.time < TRAIN[1])]; te = W[W.time >= TEST[0]]
    def fit(d):
        def loss(p):
            mu = p[0] + p[1] * d["mean"].values; sd = np.sqrt(np.exp(p[2]) + np.exp(p[3]) * d["sd"].values ** 2)
            return crps_normal(d["y"].values, mu, sd).mean()
        return optimize.minimize(loss, [0, 1, 0, 0], method="Nelder-Mead", options={"maxiter": 2000}).x
    rows = []
    p = fit(tr)
    mu = p[0] + p[1] * te["mean"].values; sd = np.sqrt(np.exp(p[2]) + np.exp(p[3]) * te["sd"].values ** 2)
    rows.append(("emos_global", crps_normal(te.y.values, mu, sd).mean()))
    # per-point EMOS
    cr = []
    for pt, g in te.groupby("point"):
        pp = fit(tr[tr.point == pt])
        mu = pp[0] + pp[1] * g["mean"].values; sd = np.sqrt(np.exp(pp[2]) + np.exp(pp[3]) * g["sd"].values ** 2)
        cr.append(crps_normal(g.y.values, mu, sd))
    rows.append(("emos_per_point", np.concatenate(cr).mean()))
    rows.append(("raw_multimodel_ensemble", crps_ens(te.y.values, te[ms].values).mean()))
    for m in ms:
        rows.append((f"single_{m}_mae", (te[m] - te.y).abs().mean()))
    z = (te.y.values - mu.mean()) if False else None
    df = pd.DataFrame(rows, columns=["forecast", "crps_ms"]).round(3).sort_values("crps_ms")
    df.to_csv(R / "emos.csv", index=False)
    W.to_parquet(R / "emos_frame.parquet")
    note("EMOS 100 m wind vs ERA5, test 2026, CRPS m/s (single models: MAE, which equals CRPS for a point forecast): " +
         ", ".join(f"{r.forecast} {r.crps_ms}" for r in df.itertuples()))


# 4. Case studies: worst blend days and storm cut-out hours; per-model timing error of the wind peak.
def cases():
    era = pd.read_parquet(C / "era5.parquet", columns=["time", "point", "ws100", "mslp"])
    rows = []
    for area in ("DK1", "DK2"):
        t = pd.read_parquet(R / f"power_test_{area}.parquet")
        day = (t["stack"] - t["y"]).abs().groupby(t.index.date).mean().sort_values(ascending=False)
        for d, e in day.head(5).items():
            rows.append(dict(kind="worst_day", area=area, date=str(d), blend_nmae_pct=round(100 * e, 1)))
    # storms: hours where ERA5 100 m wind at offshore points averages above 22 m/s
    off = era[era.point.isin(OFFSHORE)].groupby("time")[["ws100", "mslp"]].mean()
    storm_days = sorted(set(off[off.ws100 > 22].index.date))
    for d in storm_days:
        win = off.loc[str(d)]
        rows.append(dict(kind="storm_day", date=str(d), era_max_ws100=round(win.ws100.max(), 1), era_min_mslp=round(win.mslp.min(), 1)))
    # timing: for each storm day, lag (h) of each model's offshore-mean wind peak vs ERA5 within a 48 h window
    for m in models():
        f = fc(m, 1, ("ws100",)); f = f[f.point.isin(OFFSHORE)].groupby("time")["ws100"].mean()
        for d in storm_days:
            a, b = pd.Timestamp(d) - pd.Timedelta("24h"), pd.Timestamp(d) + pd.Timedelta("24h")
            fe, ee = f.loc[a:b], off.ws100.loc[a:b]
            if len(fe) > 30 and len(ee) > 30:
                rows.append(dict(kind="peak_timing", model=m, date=str(d),
                                 lag_h=(fe.idxmax() - ee.idxmax()) / pd.Timedelta("1h"),
                                 peak_err=round(fe.max() - ee.max(), 1)))
    df = pd.DataFrame(rows); df.to_csv(R / "cases.csv", index=False)
    pt = df[df.kind == "peak_timing"]
    if len(pt):
        s = pt.groupby("model").agg(mean_abs_lag=("lag_h", lambda x: x.abs().mean()), peak_err=("peak_err", "mean")).round(2)
        note(f"CASES {len(storm_days)} storm days (ERA5 offshore mean 100 m > 22 m/s). Peak timing error (h) and peak wind error (m/s) per model: " +
             ", ".join(f"{i} {r.mean_abs_lag}h/{r.peak_err:+}" for i, r in s.sort_values('mean_abs_lag').iterrows()))
    note("CASES worst blend days: " + ", ".join(f"{r.area} {r.date} ({r.blend_nmae_pct}%)" for r in df[df.kind == 'worst_day'].itertuples()))


# 5. Weather regimes and ramps: which model wins in which weather, and who catches fast swings in output.
def regimes():
    era = pd.read_parquet(C / "era5.parquet", columns=["time", "point", "ws100", "wd100", "mslp"])
    c = era[era.point == "herning"].set_index("time")
    sector = pd.cut((c.wd100 + 22.5) % 360, np.arange(0, 361, 45), labels=["N", "NE", "E", "SE", "S", "SW", "W", "NW"], right=False)
    speed = pd.cut(c.ws100, [0, 5, 10, 15, 99], labels=["calm <5", "moderate 5-10", "strong 10-15", "gale >15"])
    rows = []
    for area in ("DK1", "DK2"):
        t = pd.read_parquet(R / f"power_test_{area}.parquet")
        cols = [k for k in t.columns if k.startswith("p_")] + ["stack"]
        t["sector"], t["speed"] = sector.reindex(t.index).values, speed.reindex(t.index).values
        for by in ("sector", "speed"):
            for g, d in t.groupby(by, observed=True):
                for k in cols:
                    rows.append(dict(area=area, by=by, group=g, forecast=k.replace("p_", ""), n=len(d),
                                     nmae_pct=round(100 * (d[k] - d.y).abs().mean(), 2)))
        # ramps: actual change over 3 h above 15% of capacity; a forecast "catches" it if its own 3 h change has the same sign and is above 7.5%
        dy = t.y.diff(3)
        ramp = dy.abs() > 0.15
        for k in cols:
            df_ = t[k].diff(3)
            hit = ((np.sign(df_) == np.sign(dy)) & (df_.abs() > 0.075))[ramp].mean()
            false = ((df_.abs() > 0.15) & ~(dy.abs() > 0.075)).sum() / max(1, (df_.abs() > 0.15).sum())
            rows.append(dict(area=area, by="ramp", group="3h>15%", forecast=k.replace("p_", ""), n=int(ramp.sum()),
                             hit_rate=round(hit, 3), false_alarm=round(false, 3)))
    df = pd.DataFrame(rows); df.to_csv(R / "regimes.csv", index=False)
    for area in ("DK1", "DK2"):
        d = df[(df.area == area) & (df.by == "sector")]
        win = d.loc[d.groupby("group", observed=True).nmae_pct.idxmin()]
        note(f"REGIMES {area} best forecast per wind direction (nMAE %): " + ", ".join(f"{r.group} {r.forecast} {r.nmae_pct} (n {r.n})" for r in win.itertuples()))
        d = df[(df.area == area) & (df.by == "speed") & df.forecast.isin(["ecmwf_ifs025", "ecmwf_aifs025_single", "stack"])]
        note(f"REGIMES {area} by wind speed, IFS / AIFS / blend nMAE %: " + "; ".join(
            f"{g} " + "/".join(str(x) for x in d[d.group == g].sort_values("forecast", key=lambda s: s.map({"ecmwf_ifs025": 0, "ecmwf_aifs025_single": 1, "stack": 2})).nmae_pct)
            for g in d.group.unique()))
        r = df[(df.area == area) & (df.by == "ramp")].sort_values("hit_rate", ascending=False)
        note(f"RAMPS {area} ({int(r.n.iloc[0])} ramps of >15% capacity in 3 h): hit rate / false alarm: " + ", ".join(f"{x.forecast} {x.hit_rate}/{x.false_alarm}" for x in r.itertuples()))


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT / "experiments"))
    for step in (["scorecard", "power", "emos", "cases", "regimes"] if sys.argv[1] == "all" else sys.argv[1:]):
        t0 = time.time()
        try:
            globals()[step]()
        except Exception as e:
            import traceback; traceback.print_exc(); note(f"FAILED {step}: {e!r}")
        print(f"{step} took {time.time() - t0:.0f}s", flush=True)
