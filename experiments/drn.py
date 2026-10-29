"""Neural post-processing on gene's GPU: DRN (Rasp and Lerch 2018) and BQN (Bremnes 2020) for 100 m wind,
target ERA5, inputs every model's ws100/ws10/gust/t2m/mslp/direction plus a point embedding and season.
Same train/test split as run.py. Appends to results/RESULTS.md."""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "experiments"))
from run import C, R, TRAIN, TEST, models, fc, note
torch.set_num_threads(2)
dev = "cuda" if torch.cuda.is_available() else "cpu"
VARS = ("ws100", "ws10", "gust", "t2m", "mslp", "wd100")


def frame():
    W = None
    for m in models():
        d = fc(m, 1, VARS)
        d["s"], d["c"] = np.sin(np.radians(d.wd100)), np.cos(np.radians(d.wd100))
        d = d.drop(columns="wd100").rename(columns={v: f"{m}|{v}" for v in [*VARS, "s", "c"] if v != "wd100"})
        W = d if W is None else W.merge(d, on=["time", "point"])
    era = pd.read_parquet(C / "era5.parquet", columns=["time", "point", "ws100"]).rename(columns={"ws100": "y"})
    W = W.merge(era, on=["time", "point"])
    feats = [c for c in W.columns if "|" in c and W[c].notna().mean() > 0.5]
    W[feats] = W[feats].fillna(W[feats].median())
    W = W.dropna(subset=["y"])
    W["doy_s"], W["doy_c"] = np.sin(2 * np.pi * W.time.dt.dayofyear / 365.25), np.cos(2 * np.pi * W.time.dt.dayofyear / 365.25)
    W["hs"], W["hc"] = np.sin(2 * np.pi * W.time.dt.hour / 24), np.cos(2 * np.pi * W.time.dt.hour / 24)
    return W, feats + ["doy_s", "doy_c", "hs", "hc"]


class Net(nn.Module):
    def __init__(self, nin, npts, nout):
        super().__init__()
        self.emb = nn.Embedding(npts, 8)
        self.f = nn.Sequential(nn.Linear(nin + 8, 128), nn.ReLU(), nn.Dropout(0.1), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, nout))
    def forward(self, x, p):
        return self.f(torch.cat([x, self.emb(p)], 1))


def crps_normal_t(y, mu, sd):
    z = (y - mu) / sd; n = torch.distributions.Normal(0, 1)
    return sd * (z * (2 * n.cdf(z) - 1) + 2 * torch.exp(n.log_prob(z)) - 1 / np.sqrt(np.pi))


def bern_basis(deg, taus):
    from scipy.special import comb
    k = np.arange(deg + 1)
    return torch.tensor(comb(deg, k) * taus[:, None] ** k * (1 - taus[:, None]) ** (deg - k), dtype=torch.float32)


def train(kind, Xtr, Ptr, ytr, Xva, Pva, yva, npts, epochs=40, deg=12):
    taus = np.linspace(0.01, 0.99, 99); B = bern_basis(deg, taus).to(dev); T = torch.tensor(taus, dtype=torch.float32, device=dev)
    net = Net(Xtr.shape[1], npts, 2 if kind == "drn" else deg + 1).to(dev)
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    def out(x, p):
        o = net(x, p)
        if kind == "drn":
            return o[:, 0], nn.functional.softplus(o[:, 1]) + 1e-3
        # BQN: increasing Bernstein coefficients => monotone quantiles
        coef = torch.cumsum(torch.cat([o[:, :1], nn.functional.softplus(o[:, 1:])], 1), 1)
        return coef @ B.T, None
    def loss(x, p, y):
        a, b = out(x, p)
        if kind == "drn":
            return crps_normal_t(y, a, b).mean()
        e = y[:, None] - a
        return 2 * torch.maximum(T * e, (T - 1) * e).mean()
    best, bad, state = 1e9, 0, None
    n = len(ytr)
    for ep in range(epochs):
        net.train(); perm = torch.randperm(n, device=dev)
        for i in range(0, n, 4096):
            j = perm[i:i + 4096]; opt.zero_grad(); l = loss(Xtr[j], Ptr[j], ytr[j]); l.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            v = np.mean([loss(Xva[i:i + 50000], Pva[i:i + 50000], yva[i:i + 50000]).item() for i in range(0, len(yva), 50000)])
        if v < best - 1e-4:
            best, bad, state = v, 0, {k: t.clone() for k, t in net.state_dict().items()}
        else:
            bad += 1
            if bad >= 5:
                break
    net.load_state_dict(state)
    return net, out, loss


def main():
    t0 = time.time()
    W, feats = frame()
    W["pid"] = W.point.astype("category").cat.codes
    tr = W[(W.time >= TRAIN[0]) & (W.time < "2025-11-15")]; va = W[(W.time >= "2025-11-15") & (W.time < TRAIN[1])]
    te = W[W.time >= TEST[0]]
    mu_, sd_ = tr[feats].mean(), tr[feats].std().fillna(1).replace(0, 1)
    T_ = lambda d: (torch.tensor(((d[feats] - mu_) / sd_).values, dtype=torch.float32, device=dev),
                    torch.tensor(d.pid.values, dtype=torch.long, device=dev),
                    torch.tensor(d.y.values, dtype=torch.float32, device=dev))
    Xtr, Ptr, ytr = T_(tr); Xva, Pva, yva = T_(va); Xte, Pte, yte = T_(te)
    res = []
    for kind in ("drn", "bqn"):
        scores = []
        for seed in range(3):  # small ensemble of seeds, as in the DRN papers
            torch.manual_seed(seed)
            net, out, loss = train(kind, Xtr, Ptr, ytr, Xva, Pva, yva, W.pid.nunique())
            with torch.no_grad():
                scores.append(np.sum([loss(Xte[i:i + 50000], Pte[i:i + 50000], yte[i:i + 50000]).item() * len(yte[i:i + 50000])
                                       for i in range(0, len(yte), 50000)]) / len(yte))
        res.append((kind, float(np.mean(scores)), float(np.std(scores))))
    pd.DataFrame(res, columns=["method", "crps_ms", "seed_sd"]).to_csv(R / "drn_bqn.csv", index=False)
    note(f"NEURAL post-processing of 100 m wind vs ERA5, test 2026, CRPS m/s (3 seeds each, {len(feats)} inputs, {dev}): " +
         ", ".join(f"{k} {c:.3f} (sd {s:.3f})" for k, c, s in res) + f"; took {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
