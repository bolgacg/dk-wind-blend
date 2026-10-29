"""Gale fix. Why does the blend fail above 15 m/s? Two suspects: turbines cutting out (weather) and curtailment at
negative prices (market). Test both, then refit the blend with gust, shear and cut-out features."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run import C, R, TRAIN, TEST, DK2, models, fc, target, gbm, note


def frame(model, area, extra):
    d = fc(model, 1, ("ws100", "wd100", "ws10", "gust"))
    d = d[d["point"].isin(DK2) == (area == "DK2")]
    d["ws3"] = np.clip(d["ws100"], 0, 25) ** 3
    d["u"] = -d["ws100"] * np.sin(np.radians(d["wd100"])); d["v"] = -d["ws100"] * np.cos(np.radians(d["wd100"]))
    vals = ["ws100", "ws3", "u", "v"]
    if extra:
        d["shear"] = d["ws100"] / d["ws10"].clip(lower=0.5)      # stability proxy: high shear = stable air
        d["cut"] = (d["gust"].fillna(d["ws100"] * 1.4) > 30).astype(float)  # gusts near turbine cut-out
        vals += ["gust", "shear", "cut"]
    w = d.pivot_table(index="time", columns="point", values=vals)
    w.columns = [f"{model}|{a}|{b}" for a, b in w.columns]
    if extra:
        ws = d.pivot_table(index="time", columns="point", values="ws100")
        w[f"{model}|maxws"] = ws.max(1); w[f"{model}|n_over20"] = (ws > 20).sum(1)
    return w


def main():
    pr = pd.read_parquet(C / "DayAheadPrices.parquet")
    pr["time"] = pd.to_datetime(pr["TimeUTC"]).dt.floor("h")
    price = pr.groupby(["time", "PriceArea"])["DayAheadPriceEUR"].mean()
    era = pd.read_parquet(C / "era5.parquet", columns=["time", "point", "ws100"])
    rows, series = [], {}
    for area in ("DK1", "DK2"):
        cf, cap = target(area)
        wind = era[era.point.isin(DK2) == (area == "DK2")].groupby("time").ws100.mean()
        p = price.xs(area, level="PriceArea")
        for variant in ("baseline", "with gust, shear, cut-out", "after the auction: plus day-ahead price"):
            extra = variant != "baseline"
            frames = {m: frame(m, area, extra) for m in models()}
            common = cf.index
            for w in frames.values():
                common = common.intersection(w.dropna(how="all").index)
            tr = common[(common >= TRAIN[0]) & (common < TRAIN[1])]; te = common[common >= TEST[0]]
            preds = []
            for m, w in frames.items():
                X = w.loc[common].fillna(-1).assign(hour=common.hour)
                if variant.startswith("after"):
                    X["price"] = p.reindex(common).fillna(p.median()).values
                preds.append(np.clip(gbm().fit(X.loc[tr], cf.loc[tr]).predict(X.loc[te]), 0, 1.1))
            blend = pd.Series(np.mean(preds, 0), te); y = cf.loc[te]
            tag = variant
            series[(area, tag)] = blend
            d = pd.DataFrame({"err": blend - y, "ws": wind.reindex(te).values, "price": p.reindex(te).values}, index=te)
            for name, g in [("all", d), ("gale >15", d[d.ws > 15]), ("gale >15, price <= 0", d[(d.ws > 15) & (d.price <= 0)]),
                            ("gale >15, price > 0", d[(d.ws > 15) & (d.price > 0)]), ("price <= 0", d[d.price <= 0])]:
                rows.append(dict(area=area, model=tag, subset=name, n=len(g), nmae_pct=round(100 * g.err.abs().mean(), 2),
                                 bias_pct=round(100 * g.err.mean(), 2)))
    df = pd.DataFrame(rows); df.to_csv(R / "gale.csv", index=False)
    pd.DataFrame({f"{a}|{t}": s for (a, t), s in series.items()}).to_parquet(R / "gale_blends.parquet")
    for area in ("DK1", "DK2"):
        s = df[df.area == area]
        note(f"GALE {area}: " + "; ".join(f"{r.model} / {r.subset}: nMAE {r.nmae_pct}%, bias {r.bias_pct:+}% (n {r.n})" for r in s.itertuples()))


if __name__ == "__main__":
    main()
