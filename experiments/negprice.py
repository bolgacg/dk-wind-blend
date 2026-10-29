"""Pre-auction negative-price risk, using only what a desk knows at 10:30 the day before delivery:
day-ahead weather forecasts (wind at 34 DK sites from 8 models, solar and wind in DK and Germany), calendar and
holidays, and the day-ahead prices for today (set at yesterday's auction). Then feed the predicted probability to
the power blend and check whether the gale and curtailment errors fall without look-ahead."""
import gzip, json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run import C, R, ROOT, TRAIN, TEST, DK2, models, fc, target, gbm, note
from gale import frame

HOLIDAYS = pd.to_datetime([  # Danish public holidays 2025-2026 (incl. Christmas Eve, New Year's Eve)
    "2025-01-01", "2025-04-17", "2025-04-18", "2025-04-20", "2025-04-21", "2025-05-29", "2025-06-08", "2025-06-09",
    "2025-06-05", "2025-12-24", "2025-12-25", "2025-12-26", "2025-12-31",
    "2026-01-01", "2026-04-02", "2026-04-03", "2026-04-05", "2026-04-06", "2026-05-14", "2026-05-24", "2026-05-25",
    "2026-06-05", "2026-12-24", "2026-12-25", "2026-12-26", "2026-12-31"]).date


def solar():
    out = {}
    for f in sorted((ROOT / "data" / "raw" / "solar").glob("*.json.gz")):
        h = json.load(gzip.open(f))["hourly"]; k = f.name.split(".")[0]
        t = pd.to_datetime(h["time"])
        out[f"sw_{k}"] = pd.Series(h["shortwave_radiation_previous_day1"], t)
        out[f"t2m_{k}"] = pd.Series(h["temperature_2m_previous_day1"], t)
        out[f"ws_{k}"] = pd.Series(h["wind_speed_100m_previous_day1"], t)
    return pd.DataFrame(out)


def main():
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score, brier_score_loss
    pr = pd.read_parquet(C / "DayAheadPrices.parquet"); pr["time"] = pd.to_datetime(pr.TimeUTC).dt.floor("h")
    price = pr.groupby(["time", "PriceArea"]).DayAheadPriceEUR.mean()
    sol = solar()
    era = pd.read_parquet(C / "era5.parquet", columns=["time", "point", "ws100"])
    rows = []
    for area in ("DK1", "DK2"):
        p = price.xs(area, level="PriceArea").sort_index()
        cf, cap = target(area)
        wind = {m: frame(m, area, False) for m in models()}
        mean_ws = pd.concat([w.filter(like="|ws100|").mean(1).rename(m) for m, w in wind.items()], axis=1)
        X = pd.DataFrame(index=mean_ws.dropna(how="all").index)
        X["ws_mean"] = mean_ws.mean(1); X["ws_spread"] = mean_ws.std(1)
        X = X.join(sol, how="left")
        X["hour"], X["dow"] = X.index.hour, X.index.dayofweek
        X["holiday"] = np.isin(X.index.date, HOLIDAYS).astype(int)
        X["doy_s"], X["doy_c"] = np.sin(2 * np.pi * X.index.dayofyear / 365.25), np.cos(2 * np.pi * X.index.dayofyear / 365.25)
        # prices known at 10:30 on D-1: today's prices (D-1), from yesterday's auction. Same hour, and today's minimum.
        X["price_lag24"] = p.reindex(X.index - pd.Timedelta("24h")).values
        dmin = p.groupby(p.index.date).min()
        X["price_min_today"] = pd.Series(pd.to_datetime(X.index.date) - pd.Timedelta("1D")).map(lambda d: dmin.get(d.date(), np.nan)).values
        y = (p.reindex(X.index) <= 0)
        ok = y.notna() & X["price_lag24"].notna()
        X, y = X[ok], y[ok].astype(int)
        tr = X.index[(X.index >= TRAIN[0]) & (X.index < TRAIN[1])]; te = X.index[X.index >= TEST[0]]
        clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=1.0)
        clf.fit(X.loc[tr], y.loc[tr])
        pte = pd.Series(clf.predict_proba(X.loc[te])[:, 1], te)
        base = y.loc[tr].mean()
        auc, bs, bs0 = roc_auc_score(y.loc[te], pte), brier_score_loss(y.loc[te], pte), brier_score_loss(y.loc[te], np.full(len(te), base))
        caught = ((pte > 0.5) & (y.loc[te] == 1)).sum() / max(1, y.loc[te].sum())
        false = ((pte > 0.5) & (y.loc[te] == 0)).sum() / max(1, (pte > 0.5).sum())
        rows.append(dict(area=area, kind="negprice", auc=round(auc, 3), brier=round(bs, 4), brier_climatology=round(bs0, 4),
                         share_negative_test=round(y.loc[te].mean(), 3), caught_at_0_5=round(caught, 3), false_alarm_at_0_5=round(false, 3)))
        # out-of-fold probabilities on train (5 time blocks) so the power model trains on honest p_neg
        p_all = pd.Series(np.nan, X.index)
        blocks = np.array_split(np.arange(len(tr)), 5)
        for b in blocks:
            fit_idx = tr.delete(b)
            m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=1.0).fit(X.loc[fit_idx], y.loc[fit_idx])
            p_all.loc[tr[b]] = m.predict_proba(X.loc[tr[b]])[:, 1]
        p_all.loc[te] = pte
        # power blend with and without the pre-auction p_neg
        common = cf.index.intersection(p_all.dropna().index)
        for w in wind.values():
            common = common.intersection(w.dropna(how="all").index)
        trc = common[(common >= TRAIN[0]) & (common < TRAIN[1])]; tec = common[common >= TEST[0]]
        ws_area = era[era.point.isin(DK2) == (area == "DK2")].groupby("time").ws100.mean()
        for variant in ("baseline", "plus pre-auction negative-price risk"):
            preds = []
            for mname, w in wind.items():
                Xp = w.loc[common].fillna(-1).assign(hour=common.hour)
                if variant != "baseline":
                    Xp["p_neg"] = p_all.loc[common].values; Xp["holiday"] = X.loc[common, "holiday"].values
                preds.append(np.clip(gbm().fit(Xp.loc[trc], cf.loc[trc]).predict(Xp.loc[tec]), 0, 1.1))
            err = pd.Series(np.mean(preds, 0), tec) - cf.loc[tec]
            d = pd.DataFrame({"err": err, "ws": ws_area.reindex(tec).values, "price": p.reindex(tec).values})
            for name, g in [("all", d), ("gale >15", d[d.ws > 15]), ("gale >15, price <= 0", d[(d.ws > 15) & (d.price <= 0)]),
                            ("price <= 0", d[d.price <= 0])]:
                rows.append(dict(area=area, kind="power", model=variant, subset=name, n=len(g),
                                 nmae_pct=round(100 * g.err.abs().mean(), 2), bias_pct=round(100 * g.err.mean(), 2)))
    df = pd.DataFrame(rows); df.to_csv(R / "negprice.csv", index=False)
    for area in ("DK1", "DK2"):
        n = df[(df.area == area) & (df.kind == "negprice")].iloc[0]
        note(f"NEGPRICE {area} at 10:30 the day before: AUC {n.auc}, Brier {n.brier} against {n.brier_climatology} for climatology; "
             f"{n.share_negative_test:.1%} of test hours negative; at p>0.5 caught {n.caught_at_0_5:.0%}, false alarms {n.false_alarm_at_0_5:.0%}")
        s = df[(df.area == area) & (df.kind == "power")]
        note(f"NEGPRICE {area} power, no look-ahead: " + "; ".join(f"{r.model} / {r.subset}: {r.nmae_pct}% (bias {r.bias_pct:+}%, n {r.n})" for r in s.itertuples()))


if __name__ == "__main__":
    main()
