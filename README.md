# Eight weather models, one wind forecast for Denmark

Which public weather model forecasts Danish wind best one day ahead, what blending them buys, and why the worst forecast day of 2026 was a market event rather than a weather failure.

Page: https://bolgacg.github.io/dk-wind-blend/ · Report (PDF): `docs/denmark-wind-model-study.pdf`

- `collect/` downloads the Open-Meteo forecast archive (8 models, 34 sites), ERA5, DMI station observations and Energinet production and prices, and cleans them into Parquet.
- `experiments/run.py` scores each model against ERA5 and DMI stations, maps winds to power per price area, blends, and builds quantile, conformal and EMOS forecasts, storm cases and weather-regime splits.
- `experiments/drn.py` trains the DRN and Bernstein quantile network post-processing models (PyTorch).
- `experiments/gale.py` tests why the blend fails in gales: turbine cut-out against curtailment at negative prices.
- `site/make_data.py` builds the page data and the report figures.

Train on 2025, test on January to October 2026. Data: Open-Meteo, ERA5, DMI open data, Energinet Energi Data Service, Natural Earth.
