# UAC Care Load Forecasting Dashboard

Streamlit app forecasting **Children in HHS Care** (the daily care-load metric)
for the Predictive Forecasting of Care Load & Placement Demand project.

## What's inside

```
app/
├── app.py              # Streamlit dashboard (run this)
├── forecasting.py      # Feature engineering + all 6 models + KPIs (imported by app.py)
├── clean_data.csv       # Pre-cleaned, continuous daily dataset (2023-01-12 → 2025-12-21)
└── requirements.txt     # MUST stay at repo root for Streamlit Cloud
```

There's also a `data/` folder with `clean_data.py` (the cleaning script) and
`raw_data.csv` (the original HHS export) if you want to re-run cleaning on a
refreshed export later — just drop the new raw CSV in and re-run the script,
then copy the new `clean_data.csv` into `app/`.

## Run locally

```bash
cd app
pip install -r requirements.txt
streamlit run app.py
```

## Deploying to Streamlit Cloud (avoiding your past issues)

You hit two recurring problems before: `requirements.txt` not found at the repo
root, and `ModuleNotFoundError: plotly`. To avoid both this time:

1. **Push the contents of the `app/` folder to the ROOT of your GitHub repo** —
   not into a subfolder. Streamlit Cloud looks for `requirements.txt` at the
   repo root by default. So your repo should look like:
   ```
   your-repo/
   ├── app.py
   ├── forecasting.py
   ├── clean_data.csv
   └── requirements.txt
   ```
   If you'd rather keep a subfolder (e.g. `app/app.py`), set the **"Main file
   path"** in Streamlit Cloud's deploy settings to `app/app.py` AND set the
   advanced setting so it still finds `app/requirements.txt` — but the simplest
   fix is just pushing these 4 files to the repo root.

2. **`plotly` and `scipy` are in `requirements.txt` this time** — last time's
   `ModuleNotFoundError: plotly` happened because it was used in code but
   missing from the file. Double check after pushing that GitHub actually
   shows `plotly>=5.18` and `scipy>=1.11` inside `requirements.txt` on the repo
   root — sometimes a stale cached version gets pushed instead.

3. **No absolute or `../` file paths** — `app.py` loads `clean_data.csv` using
   a plain relative path (`"clean_data.csv"`), so as long as that file sits
   next to `app.py` in the deployed repo, there are no path errors like the
   ODS file issue from your EduPro project.

4. After deploying, if Streamlit Cloud still throws a module error, open
   **Manage app → Reboot app** — it sometimes caches an old `requirements.txt`
   from a previous deploy attempt.

## Dashboard features (mapped to project requirements)

- **Future Care Load Forecast Chart** — Tab 1, with confidence-interval bands,
  forecast-horizon slider (7–60 days), and model toggle (multiselect)
- **Discharge Demand / Net Pressure Panel** — Tab 2 (transfers vs discharges,
  net pressure bar chart)
- **Model Selection & Comparison** — Tab 3, walk-forward backtest table (MAE,
  RMSE, MAPE, Accuracy %) + horizon-error curve (short vs medium-term reliability)
- **Confidence Interval Visualization** — shaded bands on the forecast chart
- **KPIs in the header strip** — Forecast Accuracy %, Capacity Breach
  Probability, Surge Lead Time, Forecast Stability Index
- **Scenario comparison** — adjustable capacity threshold in the sidebar drives
  the breach-probability / surge-lead-time KPIs and the red threshold line on
  the chart

## Models implemented

| Category | Models |
|---|---|
| Baseline | Naive Persistence, Moving Average (7-day) |
| Statistical | SARIMA, Exponential Smoothing (Holt-Winters, damped trend, weekly seasonality) |
| Machine Learning | Random Forest, Gradient Boosting (recursive multi-step forecasting on lag/rolling/calendar features) |

All are scored with strict time-based walk-forward validation (rolling-origin,
no random sampling) across 5 historical folds.

## Data cleaning notes

The raw HHS export skips most Fridays and Saturdays. `clean_data.py` reindexes
the series to a **continuous daily calendar** and time-interpolates the gaps
so lag/rolling features and the statistical models (which assume a regular
frequency) work correctly. Each row keeps an `is_observed` flag so you can
distinguish originally-reported days from interpolated ones — used in the
dashboard's "Data & Methodology" tab and an optional toggle on the main chart.
