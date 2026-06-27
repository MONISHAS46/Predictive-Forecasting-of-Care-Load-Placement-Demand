"""
Forecasting engine for HHS Care Load.

Contains:
  - feature engineering (lags, rolling stats, calendar effects)
  - 4 model families: Naive/Moving-Average, ARIMA/SARIMA, Exponential Smoothing,
    Random Forest, Gradient Boosting
  - walk-forward backtest + multi-horizon evaluation
  - KPI calculations (capacity breach probability, surge lead time,
    forecast stability index, forecast accuracy)

Designed to be imported by the Streamlit app. Kept dependency-light
(pandas, numpy, scikit-learn, statsmodels only) to avoid Streamlit Cloud
deployment issues.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings
warnings.filterwarnings("ignore")

TARGET = "hhs_care"


# --------------------------------------------------------------------------
# Feature engineering
# --------------------------------------------------------------------------
def build_features(df: pd.DataFrame, target: str = TARGET) -> pd.DataFrame:
    """Add lag, rolling, flow-signal and calendar features to the dataframe.

    Expects df indexed/sorted by date with at least the target column and
    transfers_to_hhs / discharges if present.
    """
    out = df.copy()
    out = out.sort_values("date").reset_index(drop=True)

    for lag in [1, 7, 14]:
        out[f"{target}_lag{lag}"] = out[target].shift(lag)

    for window in [7, 14]:
        out[f"{target}_roll_mean{window}"] = out[target].shift(1).rolling(window).mean()
        out[f"{target}_roll_std{window}"] = out[target].shift(1).rolling(window).std()

    if "transfers_to_hhs" in out.columns and "discharges" in out.columns:
        out["net_pressure"] = out["transfers_to_hhs"] - out["discharges"]
        out["net_pressure_lag1"] = out["net_pressure"].shift(1)
        out["net_pressure_roll7"] = out["net_pressure"].shift(1).rolling(7).mean()

    out["dow"] = out["date"].dt.dayofweek
    out["month"] = out["date"].dt.month
    out["is_month_start"] = out["date"].dt.is_month_start.astype(int)
    out["is_month_end"] = out["date"].dt.is_month_end.astype(int)

    return out


def feature_columns(df: pd.DataFrame, target: str = TARGET) -> list[str]:
    exclude = {target, "date", "is_observed", "intake_cbp", "cbp_custody",
               "transfers_to_hhs", "discharges"}
    return [c for c in df.columns if c not in exclude and df[c].dtype != "O"]


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def mae(y_true, y_pred):
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def mape(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def all_metrics(y_true, y_pred) -> dict:
    return {"MAE": mae(y_true, y_pred), "RMSE": rmse(y_true, y_pred), "MAPE": mape(y_true, y_pred)}


# --------------------------------------------------------------------------
# Model wrappers — each exposes .fit(train_df) and .forecast(h) -> (mean, lower, upper)
# --------------------------------------------------------------------------
@dataclass
class ForecastResult:
    mean: np.ndarray
    lower: np.ndarray
    upper: np.ndarray


class NaivePersistence:
    name = "Naive Persistence"

    def fit(self, train: pd.DataFrame, target: str = TARGET):
        self.last_value = train[target].iloc[-1]
        resid = train[target].diff().dropna()
        self.resid_std = resid.std() if len(resid) > 1 else 0.0
        return self

    def forecast(self, h: int) -> ForecastResult:
        mean = np.repeat(self.last_value, h)
        spread = self.resid_std * np.sqrt(np.arange(1, h + 1))
        return ForecastResult(mean, mean - 1.96 * spread, mean + 1.96 * spread)


class MovingAverage:
    name = "Moving Average (7d)"

    def __init__(self, window: int = 7):
        self.window = window

    def fit(self, train: pd.DataFrame, target: str = TARGET):
        self.avg = train[target].tail(self.window).mean()
        resid = train[target].tail(30).diff().dropna()
        self.resid_std = resid.std() if len(resid) > 1 else 0.0
        return self

    def forecast(self, h: int) -> ForecastResult:
        mean = np.repeat(self.avg, h)
        spread = self.resid_std * np.sqrt(np.arange(1, h + 1))
        return ForecastResult(mean, mean - 1.96 * spread, mean + 1.96 * spread)


class SarimaModel:
    name = "SARIMA"

    def __init__(self, order=(2, 1, 2), seasonal_order=(1, 0, 1, 7)):
        self.order = order
        self.seasonal_order = seasonal_order

    def fit(self, train: pd.DataFrame, target: str = TARGET):
        series = train[target].astype(float).values
        try:
            self.model = SARIMAX(
                series, order=self.order, seasonal_order=self.seasonal_order,
                enforce_stationarity=False, enforce_invertibility=False,
            ).fit(disp=False)
            self.failed = False
        except Exception:
            self.failed = True
            self.last_value = series[-1]
            self.resid_std = pd.Series(series).diff().dropna().std()
        return self

    def forecast(self, h: int) -> ForecastResult:
        if self.failed:
            mean = np.repeat(self.last_value, h)
            spread = self.resid_std * np.sqrt(np.arange(1, h + 1))
            return ForecastResult(mean, mean - 1.96 * spread, mean + 1.96 * spread)
        pred = self.model.get_forecast(steps=h)
        mean = pred.predicted_mean
        ci = pred.conf_int(alpha=0.05)
        return ForecastResult(np.asarray(mean), np.asarray(ci[:, 0]), np.asarray(ci[:, 1]))


class ExpSmoothingModel:
    name = "Exponential Smoothing"

    def fit(self, train: pd.DataFrame, target: str = TARGET):
        series = train[target].astype(float).values
        try:
            self.model = ExponentialSmoothing(
                series, trend="add", seasonal="add", seasonal_periods=7,
                damped_trend=True,
            ).fit()
            self.failed = False
            resid = self.model.resid
            self.resid_std = np.std(resid) if len(resid) > 1 else 0.0
        except Exception:
            self.failed = True
            self.last_value = series[-1]
            self.resid_std = pd.Series(series).diff().dropna().std()
        return self

    def forecast(self, h: int) -> ForecastResult:
        if self.failed:
            mean = np.repeat(self.last_value, h)
        else:
            mean = np.asarray(self.model.forecast(h))
        spread = self.resid_std * np.sqrt(np.arange(1, h + 1))
        return ForecastResult(mean, mean - 1.96 * spread, mean + 1.96 * spread)


class MLModel:
    """Shared wrapper for RandomForest / GradientBoosting using recursive
    multi-step forecasting over engineered lag/rolling features."""

    def __init__(self, estimator, name: str):
        self.estimator = estimator
        self.name = name

    def fit(self, train_features: pd.DataFrame, feat_cols: list[str], target: str = TARGET):
        data = train_features.dropna(subset=feat_cols + [target])
        self.feat_cols = feat_cols
        self.target = target
        X, y = data[feat_cols], data[target]
        self.estimator.fit(X, y)
        resid = self.estimator.predict(X) - y.values
        self.resid_std = float(np.std(resid))
        self.history = train_features.copy()
        return self

    def forecast(self, h: int) -> ForecastResult:
        hist = self.history.copy()
        preds = []
        last_date = hist["date"].iloc[-1]
        for step in range(h):
            next_date = last_date + pd.Timedelta(days=step + 1)
            row = {"date": next_date}
            tmp = pd.concat([hist, pd.DataFrame([row])], ignore_index=True)
            tmp.loc[tmp.index[-1], self.target] = np.nan
            tmp = _refresh_features_last_row(tmp, self.target)
            x = tmp[self.feat_cols].iloc[[-1]].copy()
            # Fill any remaining NaNs (e.g. std on a short window) using the
            # last valid value of each feature column across full history
            for col in self.feat_cols:
                if x[col].isna().any():
                    fallback = tmp[col].ffill().iloc[-1]
                    x[col] = x[col].fillna(fallback if pd.notna(fallback) else 0.0)
            yhat = float(self.estimator.predict(x)[0])
            tmp.loc[tmp.index[-1], self.target] = yhat
            hist = tmp
            preds.append(yhat)
        mean = np.array(preds)
        spread = self.resid_std * np.sqrt(np.arange(1, h + 1))
        return ForecastResult(mean, mean - 1.96 * spread, mean + 1.96 * spread)


def _refresh_features_last_row(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """Recompute lag/rolling/calendar features only (cheap) for recursive forecasting."""
    df = df.copy()
    for lag in [1, 7, 14]:
        df[f"{target}_lag{lag}"] = df[target].shift(lag)
    for window in [7, 14]:
        df[f"{target}_roll_mean{window}"] = df[target].shift(1).rolling(window).mean()
        df[f"{target}_roll_std{window}"] = df[target].shift(1).rolling(window).std()
    if "net_pressure" in df.columns:
        df["net_pressure_lag1"] = df["net_pressure"].shift(1)
        df["net_pressure_roll7"] = df["net_pressure"].shift(1).rolling(7).mean()
    df["dow"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["is_month_start"] = df["date"].dt.is_month_start.astype(int)
    df["is_month_end"] = df["date"].dt.is_month_end.astype(int)
    return df


MODEL_REGISTRY = {
    "Naive Persistence": lambda: NaivePersistence(),
    "Moving Average (7d)": lambda: MovingAverage(7),
    "SARIMA": lambda: SarimaModel(),
    "Exponential Smoothing": lambda: ExpSmoothingModel(),
    "Random Forest": lambda: MLModel(
        RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42), "Random Forest"
    ),
    "Gradient Boosting": lambda: MLModel(
        GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42),
        "Gradient Boosting",
    ),
}

ML_MODELS = {"Random Forest", "Gradient Boosting"}


def fit_and_forecast(model_name: str, train_df: pd.DataFrame, horizon: int, target: str = TARGET) -> ForecastResult:
    """Single entry point used by the Streamlit app."""
    factory = MODEL_REGISTRY[model_name]
    model = factory()
    if model_name in ML_MODELS:
        feats = build_features(train_df, target)
        cols = feature_columns(feats, target)
        model.fit(feats, cols, target)
    else:
        model.fit(train_df, target)
    return model.forecast(horizon)


# --------------------------------------------------------------------------
# Walk-forward backtest for model comparison
# --------------------------------------------------------------------------
def walk_forward_backtest(df: pd.DataFrame, model_name: str, horizon: int = 7,
                           n_folds: int = 5, min_train: int = 180, target: str = TARGET) -> dict:
    """Rolling-origin backtest: for each fold, train on data up to a cutoff,
    forecast `horizon` days ahead, score against actuals. Returns averaged metrics
    and per-horizon error breakdown."""
    df = df.sort_values("date").reset_index(drop=True)
    n = len(df)
    fold_starts = np.linspace(min_train, n - horizon - 1, n_folds, dtype=int)
    fold_starts = sorted(set(fold_starts))

    all_metrics_list = []
    horizon_errors = {h: [] for h in range(1, horizon + 1)}

    for cutoff in fold_starts:
        train = df.iloc[:cutoff]
        test = df.iloc[cutoff:cutoff + horizon]
        if len(test) < horizon:
            continue
        try:
            result = fit_and_forecast(model_name, train, horizon, target)
        except Exception:
            continue
        y_true = test[target].values
        y_pred = result.mean[:len(y_true)]
        all_metrics_list.append(all_metrics(y_true, y_pred))
        for i, (t, p) in enumerate(zip(y_true, y_pred), start=1):
            horizon_errors[i].append(abs(t - p))

    if not all_metrics_list:
        return {"MAE": np.nan, "RMSE": np.nan, "MAPE": np.nan, "horizon_errors": {}}

    avg_metrics = {
        k: float(np.mean([m[k] for m in all_metrics_list])) for k in ["MAE", "RMSE", "MAPE"]
    }
    avg_horizon_errors = {h: float(np.mean(v)) for h, v in horizon_errors.items() if v}
    avg_metrics["horizon_errors"] = avg_horizon_errors
    return avg_metrics


# --------------------------------------------------------------------------
# KPIs
# --------------------------------------------------------------------------
def forecast_accuracy_pct(mape_value: float) -> float:
    return max(0.0, 100.0 - mape_value)


def capacity_breach_probability(result: ForecastResult, capacity_threshold: float) -> float:
    """Probability (fraction of forecast horizon days) the upper/likely range
    breaches the stated capacity threshold, assuming normal error distribution
    around the mean forecast using the CI width as ~95% band."""
    sigma = (result.upper - result.lower) / (2 * 1.96)
    sigma = np.where(sigma <= 0, 1e-6, sigma)
    from scipy.stats import norm
    probs = 1 - norm.cdf(capacity_threshold, loc=result.mean, scale=sigma)
    return float(np.mean(probs) * 100)


def surge_lead_time(result: ForecastResult, capacity_threshold: float) -> int | None:
    """First forecast day index (1-indexed) where mean forecast crosses threshold."""
    over = np.where(result.mean >= capacity_threshold)[0]
    return int(over[0] + 1) if len(over) else None


def forecast_stability_index(result: ForecastResult) -> float:
    """Lower = more stable. Defined as the coefficient of variation of the
    forecast's day-over-day changes, normalized 0-100 (100 = very stable)."""
    diffs = np.diff(result.mean)
    if len(diffs) == 0 or np.mean(np.abs(result.mean)) == 0:
        return 100.0
    volatility = np.std(diffs) / (np.mean(np.abs(result.mean)) + 1e-6)
    score = max(0.0, 100.0 - volatility * 100)
    return float(score)
