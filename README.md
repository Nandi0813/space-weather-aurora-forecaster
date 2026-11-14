# Space weather & aurora forecaster for Central Europe

Homework #26, AI in Practice (BME VIK).

**Live site:** https://nandi0813.github.io/space-weather-aurora-forecaster/

The site shows:

- live solar wind at L1 (speed, density, IMF Bz/Bt) from NOAA SWPC real-time feeds,
- the planetary Kp index, observed and NOAA's forecast,
- the NOAA OVATION aurora probability map, zoomed on Europe, with Budapest, Vienna, Prague, Berlin and Warsaw marked,
- **our own Kp forecast for the next 3 hours**, computed in the browser from the live solar wind,
- a back-test of the forecaster on past geomagnetic storms, including the May 2024 (Gannon) G5 storm.

## Data

| Source | Used for |
|---|---|
| NOAA SWPC `json/rtsw/rtsw_wind_1m.json`, `rtsw_mag_1m.json` | live solar wind (browser) |
| NOAA SWPC `products/noaa-planetary-k-index-forecast.json` | live Kp, observed and forecast (browser) |
| NOAA SWPC `json/ovation_aurora_latest.json` | aurora probability map (browser) |
| NASA OMNI2 hourly (`spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni/`) | historical solar wind 2010–2025 for training |
| GFZ Potsdam Kp (`kp.gfz.de/app/json/`) | historical Kp 2010–2025, the training target |

All sources are free and need no registration. The SWPC feeds send `Access-Control-Allow-Origin: *`, so the site is
purely static and the browser calls them directly. NOAA keeps only about a week of real-time solar wind, so the model
is trained on the OMNI archive instead of self-logged data. OMNI is the same L1 measurement (ACE/DSCOVR), cleaned and
time-shifted to the bow shock.

## The forecaster (analysis element)

**Task:** predict Kp of the next 3-hour bin `[t, t+3h)` using only what is known at time `t`.

**Features** (`train.py:make_features`), from hourly means over the 3 hours before `t`:

| Feature | Why |
|---|---|
| `kp_prev` | Kp of the previous bin: geomagnetic activity is persistent |
| `v`, `n`, `bt`, `bz`, `bz_min` | solar wind speed, density, IMF magnitude, north-south IMF (mean and most southward hour) |
| `vbs` | `v · max(-Bz, 0)`: the dawn-dusk electric field that drives reconnection |
| `newell` | Newell et al. (2007) coupling function `v^(4/3) Bt^(2/3) sin^(8/3)(θ/2)` |

**Split:** train on 2010–2022 (37 714 bins), test on 2023–2025 (8 666 bins). The split is in time, not random,
because neighbouring 3-hour bins are strongly correlated and a random split would leak.

**Models:**

1. *Persistence* baseline: Kp(next) = Kp(now). This is the baseline to beat.
2. *Ridge regression* on standardized features. This is the live model: 8 coefficients exported to `docs/model.json`
   and evaluated in JavaScript.
3. *Gradient boosting* (`HistGradientBoostingRegressor`) as a non-linear comparison.

Storms are rare: only about 4% of bins have Kp ≥ 5. An unweighted fit regresses to quiet conditions and misses storms.
So training samples are weighted by `exp(Kp/2)`. A Kp 9 bin then weighs about 90× a Kp 0 bin.

**Metrics:** RMSE/MAE in Kp units. For storm detection at the Kp ≥ 5 (G1) threshold: POD (hit rate),
FAR (false-alarm ratio) and the Heidke skill score (HSS; 0 = no better than chance, 1 = perfect).

### Results on the 2023–2025 test set

| Model | RMSE | MAE | RMSE (Kp≥5) | POD | FAR | HSS |
|---|---|---|---|---|---|---|
| Persistence | 0.89 | 0.67 | 1.41 | 0.57 | 0.43 | 0.55 |
| Ridge, storm-weighted | **0.84** | 0.66 | 1.39 | 0.54 | **0.36** | **0.57** |
| Gradient boosting, storm-weighted | **0.84** | 0.66 | **1.36** | 0.49 | **0.34** | 0.55 |

Effect of the storm weighting on Ridge (same test set):

| Weight | RMSE | RMSE (Kp≥5) | POD | HSS |
|---|---|---|---|---|
| none | 0.79 | 1.60 | 0.44 | 0.54 |
| `exp(Kp/2)` | 0.84 | 1.39 | 0.54 | 0.57 |

**How to defend it in one minute:**

- Persistence is a hard baseline at a 3-hour horizon. The models beat it overall, by about 6% RMSE, and on the
  false-alarm ratio.
- The solar wind adds information mainly at storm *onset*. At onset, persistence is always one bin late.
- During the main phase of a storm, persistence is as good as or better than the models. On the May 2024 storm
  the RMSE is 1.25 for persistence and 1.30 for Ridge. Kp is a 3-hour aggregate, and L1 gives only ~30–60 minutes of
  warning, so a 3-hour forecast is mostly "current state + drivers".
- Weighting trades quiet-time accuracy for storm accuracy. That is the right trade for an aurora forecaster.
- The linear model saturates: it cannot extrapolate beyond the extreme drivers seen in training. The prediction is
  clipped to `[0, 9]`.

## Usage

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run python train.py      # downloads ~45 MB into data/, trains, writes docs/model.json + docs/backtest.json
uv run pytest -q            # unit tests: Newell function, no-leakage feature windows, metrics
cd docs && python -m http.server 8000   # open http://localhost:8000
```

GitHub Pages serves `docs/` from the `main` branch. No build step is needed.

## Files

- `train.py`: data download/caching, feature engineering, training, back-test, export
- `test_train.py`: pytest checks
- `docs/index.html`, `docs/app.js`: the static dashboard (Chart.js and Leaflet from cdnjs, with SRI hashes)
- `docs/model.json`, `docs/backtest.json`: generated by `train.py`, committed so the site works without a build

## Experiences and lessons learned

**Tools:** Claude Code (Claude Opus 5.5 model) in the terminal as an agent. It wrote the pipeline and the site,
ran the training, and checked the page in a headless browser (Playwright MCP). PyCharm was the editor.

**Key prompt:** the assignment text plus "implement it, make the commit clean, test it yourself".

**What did not work at first and had to be fixed:**

- The commonly documented SWPC feeds `products/solar-wind/plasma-1-day.json` and `mag-1-day.json` now return 404.
  They were replaced by `json/rtsw/rtsw_wind_1m.json` and `rtsw_mag_1m.json`. These mix several spacecraft, so the
  code filters on `active: true`.
- `kp.gfz-potsdam.de` now redirects to `kp.gfz.de`.
- The first map used CARTO dark tiles, which now need an API key. They were replaced by OpenStreetMap tiles.
- The first unweighted models had the best RMSE but missed more than half of the storms. The storm weighting was
  added after looking at the POD/HSS numbers, not the RMSE.
- The OVATION oval was almost invisible on a quiet day with a linear opacity scale, so a minimum opacity was added.

**Lessons:** check that each endpoint still exists before designing around it. Always compare against a persistence
baseline. A forecast with a lower RMSE can still be a worse storm forecast.

## Limitations

- The live features come from hourly means of the 1-minute real-time feed. Training used OMNI, which is time-shifted
  to the bow shock. The live data is not shifted, so the live forecast is ~30–60 minutes "early".
- If the real-time feed has a gap in the last 3 hours, the site shows "data gap" instead of a forecast.
- Visibility thresholds (Kp ≥ 6 for Berlin/Warsaw, Kp ≥ 7 for Budapest/Vienna) are rules of thumb, not model output.
