"""
CellGuardAI_fixed.py
A cleaned and defensive variant of the user's CellGuardAI app.
This file focuses on correctness of helper functions, ordering, and testability.
It is intended to be used as a Streamlit app (streamlit run ...) but here we
ensure helper functions are safely defined and unit-testable.
"""

import logging
import numpy as np
import pandas as pd

# Import streamlit lazily to avoid import-time failures during unit tests.
# When running the app normally, this will succeed.
try:
    import streamlit as st
except Exception:
    st = None  # tests can still import module without streamlit installed

from sklearn.ensemble import IsolationForest
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression

# -- Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CellGuardAI_fixed")

# ----------------------
# Simple data generator
# ----------------------
def gen_sample_data(n=800, seed=42, scenario="Generic"):
    np.random.seed(seed)
    t = np.arange(n)
    base_v = 3.7
    base_i = 1.5
    base_temp = 30.0
    soc_base = 80.0

    if scenario == "Generic":
        v = base_v + 0.05 * np.sin(t / 50) + np.random.normal(0, 0.005, n)
        i = base_i + 0.3 * np.sin(t / 30) + np.random.normal(0, 0.05, n)
        temp = base_temp + 3 * np.sin(t / 60) + np.random.normal(0, 0.3, n)
        soc = np.clip(soc_base + 10 * np.sin(t / 80) + np.random.normal(0, 1, n), 0, 100)
        cycle = t // 50
        idx = np.random.choice(n, size=18, replace=False)
        v[idx] -= np.random.uniform(0.03, 0.08, size=len(idx))
        temp[idx] += np.random.uniform(2, 5, size=len(idx))

    elif scenario == "EV":
        v = base_v + 0.03 * np.sin(t / 40) - 0.0005 * t / n + np.random.normal(0, 0.008, n)
        i = 2.5 + 0.4 * np.sin(t / 20) + np.random.normal(0, 0.07, n)
        temp = base_temp + 4 * np.sin(t / 120) + 0.01 * (t / n) * 10 + np.random.normal(0, 0.5, n)
        soc = np.clip(90 - 20 * (t / n) + np.random.normal(0, 1.5, n), 0, 100)
        cycle = t // 10
        idx = np.random.choice(n, size=35, replace=False)
        v[idx] -= np.random.uniform(0.04, 0.12, size=len(idx))
        temp[idx] += np.random.uniform(3, 8, size=len(idx))

    elif scenario == "Drone":
        v = base_v + 0.04 * np.sin(t / 30) + np.random.normal(0, 0.006, n)
        i = base_i + 0.6 * np.sin(t / 10) + np.random.normal(0, 0.2, n)
        temp = base_temp + 2 * np.sin(t / 80) + np.random.normal(0, 0.4, n)
        soc = np.clip(85 + 6 * np.sin(t / 40) + np.random.normal(0, 2, n), 0, 100)
        cycle = t // 30
        spikes = np.random.choice(n, size=60, replace=False)
        i[spikes] += np.random.uniform(2.0, 6.0, size=len(spikes))
        dips = np.random.choice(n, size=30, replace=False)
        v[dips] -= np.random.uniform(0.06, 0.18, size=len(dips))

    elif scenario == "Phone":
        v = base_v + 0.02 * np.sin(t / 80) + np.random.normal(0, 0.002, n)
        i = 0.8 + 0.1 * np.sin(t / 60) + np.random.normal(0, 0.02, n)
        temp = base_temp + 1.5 * np.sin(t / 120) + np.random.normal(0, 0.15, n)
        soc = np.clip(95 + 3 * np.sin(t / 160) + np.random.normal(0, 0.5, n), 0, 100)
        cycle = t // 200
        idx = np.random.choice(n, size=6, replace=False)
        v[idx] -= np.random.uniform(0.01, 0.03, size=len(idx))

    else:
        return gen_sample_data(n=n, seed=seed, scenario="Generic")

    df = pd.DataFrame({
        "time": t,
        "voltage": v,
        "current": i,
        "temperature": temp,
        "soc": soc,
        "cycle": cycle
    })
    return df

# ----------------------
# Helpers: normalize and ensure columns
# ----------------------
def normalize_cols(df):
    df = df.copy()
    simple = {c: "".join(ch for ch in str(c).lower() if ch.isalnum()) for c in df.columns}
    patt = {
        "voltage": ["volt", "vcell", "cellv", "packv"],
        "current": ["curr", "amp", "amps", "ichg", "idis", "current"],
        "temperature": ["temp", "temperature", "celltemp", "packtemp"],
        "soc": ["soc", "stateofcharge"],
        "cycle": ["cycle", "cyclecount", "chargecycle"],
        "time": ["time", "timestamp", "t", "index"]
    }
    cmap = {}
    used = set()
    for target, keys in patt.items():
        for orig, simplified in simple.items():
            if orig in used:
                continue
            if any(k in simplified for k in keys):
                cmap[target] = orig
                used.add(orig)
                break
    rename = {orig: targ for targ, orig in cmap.items()}
    df = df.rename(columns=rename)
    return df, cmap

def ensure_cols_exist(df, needed):
    for c in needed:
        if c not in df.columns:
            df[c] = np.nan
    return df

# ----------------------
# Feature engineering
# ----------------------
def make_features(df, window=10):
    df = df.copy()
    df = ensure_cols_exist(df, ["voltage", "current", "temperature", "soc", "cycle", "time"])
    # voltage features
    if df["voltage"].notna().sum() > 0:
        df["voltage_ma"] = df["voltage"].rolling(window, min_periods=1).mean()
        df["voltage_roc"] = df["voltage"].diff().fillna(0)
        df["voltage_var"] = df["voltage"].rolling(window, min_periods=1).var().fillna(0)
    else:
        df["voltage_ma"] = np.nan
        df["voltage_roc"] = np.nan
        df["voltage_var"] = np.nan

    # temperature features
    if df["temperature"].notna().sum() > 0:
        df["temp_ma"] = df["temperature"].rolling(window, min_periods=1).mean()
        df["temp_roc"] = df["temperature"].diff().fillna(0)
    else:
        df["temp_ma"] = np.nan
        df["temp_roc"] = np.nan

    # soc features
    if df["soc"].notna().sum() > 0:
        df["soc_ma"] = df["soc"].rolling(window, min_periods=1).mean()
        df["soc_roc"] = df["soc"].diff().fillna(0)
    else:
        df["soc_ma"] = np.nan
        df["soc_roc"] = np.nan

    # risk label: simple rules (voltage drop, temp spike, soc drop)
    if df["voltage"].notna().sum() > 0:
        volt_drop_thresh = -0.03
        cond = pd.Series(False, index=df.index)
        if df["temperature"].notna().sum() > 0:
            tmean = df["temperature"].mean()
            tstd = df["temperature"].std()
            tth = tmean + 2 * tstd if not np.isnan(tmean) and not np.isnan(tstd) else np.nan
            if not np.isnan(tth):
                cond = cond | (df["temperature"] > tth)
        if "voltage_roc" in df.columns:
            cond = cond | (df["voltage_roc"] < volt_drop_thresh)
        if "soc_roc" in df.columns:
            cond = cond | (df["soc_roc"] < -5)
        df["risk_label"] = np.where(cond, 1, 0)
    else:
        df["risk_label"] = 0

    return df

# ----------------------
# Models + scoring
# ----------------------
def run_models(df, contamination=0.05):
    df = df.copy()
    possible = ["voltage", "current", "temperature", "soc", "voltage_ma", "voltage_roc", "soc_roc", "voltage_var", "temp_ma", "cycle"]
    features = [f for f in possible if f in df.columns and df[f].notna().sum() > 0]

    df["anomaly_flag"] = 0
    df["risk_pred"] = 0
    df["battery_health_score"] = 50.0

    if len(features) >= 2 and df[features].dropna().shape[0] >= 30:
        try:
            iso = IsolationForest(n_estimators=100, contamination=contamination, random_state=42)
            X = df[features].fillna(df[features].median())
            iso.fit(X)
            df["anomaly_flag"] = iso.predict(X).map({1: 0, -1: 1})
        except Exception:
            logger.exception("IsolationForest failed during run_models")

    if "risk_label" in df.columns and df["risk_label"].nunique() > 1:
        clf_feats = [f for f in features if f in df.columns]
        if len(clf_feats) >= 2:
            try:
                Xc = df[clf_feats].fillna(df[clf_feats].median())
                yc = df["risk_label"]
                tree = DecisionTreeClassifier(max_depth=4, random_state=42)
                tree.fit(Xc, yc)
                df["risk_pred"] = tree.predict(Xc)
            except Exception:
                logger.exception("DecisionTree training failed in run_models")
                df["risk_pred"] = df["risk_label"]
        else:
            df["risk_pred"] = df["risk_label"]
    else:
        tseries = df.get("temperature", pd.Series(np.nan, index=df.index))
        tmean = tseries.mean() if hasattr(tseries, "mean") else np.nan
        tstd = tseries.std() if hasattr(tseries, "std") else np.nan
        tth = tmean + 2 * tstd if not np.isnan(tmean) and not np.isnan(tstd) else np.nan
        cond_temp = (tseries > tth) if not np.isnan(tth) else False
        df["risk_pred"] = np.where((df.get("anomaly_flag", 0) == 1) | cond_temp, 1, 0)

    base = pd.Series(0.0, index=df.index)
    if "voltage_ma" in df.columns and df["voltage_ma"].notna().sum() > 0:
        vm = df["voltage_ma"].fillna(method="ffill").fillna(df["voltage"].median() if "voltage" in df.columns else 3.7)
        base += (vm.max() - vm)
    elif "voltage" in df.columns:
        v = df["voltage"].fillna(df["voltage"].median())
        base += (v.max() - v)
    else:
        base += 0.5

    if "temperature" in df.columns and df["temperature"].notna().sum() > 0:
        t = df["temperature"].fillna(df["temperature"].median())
        base += (t - t.min()) / 10.0

    base = base + df.get("anomaly_flag", 0)*1.0 + df.get("risk_pred", 0)*0.8

    trend_feats = [f for f in ["voltage_ma", "voltage_var", "temp_ma", "cycle", "anomaly_flag"] if f in df.columns]
    if len(trend_feats) >= 2 and df[trend_feats].dropna().shape[0] >= 20:
        try:
            Xtr = df[trend_feats].fillna(0)
            reg = LinearRegression()
            reg.fit(Xtr, base)
            hp = reg.predict(Xtr)
        except Exception:
            logger.exception("LinearRegression failed in run_models")
            hp = base.values
    else:
        hp = base.values

    hp = np.array(hp, dtype=float)
    hp_norm = (hp - hp.min()) / (hp.max() - hp.min() + 1e-9)
    health_comp = 1 - hp_norm

    score = (0.6 * health_comp) + (0.25 * (1 - df.get("risk_pred", 0))) + (0.15 * (1 - df.get("anomaly_flag", 0)))
    df["battery_health_score"] = (score * 100).clip(0, 100)

    return df

# ----------------------
# Simple recommendations & labels
# ----------------------
def simple_recommend(row):
    sc = row.get("battery_health_score", 50)
    rp = row.get("risk_pred", 0)
    an = row.get("anomaly_flag", 0)
    if sc > 85 and rp == 0 and an == 0:
        return "Healthy — normal operation."
    elif 70 < sc <= 85:
        return "Watch — avoid deep discharge & fast-charge this cycle."
    elif 50 < sc <= 70:
        return "Caution — restrict fast charging; allow cooling intervals."
    else:
        return "Critical — reduce load, stop fast charging, schedule inspection."

def pack_label(score):
    if score >= 85:
        return "HEALTHY", "green"
    elif score >= 60:
        return "WATCH", "orange"
    else:
        return "CRITICAL", "red"