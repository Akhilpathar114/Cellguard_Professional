
# CellGuardAI - Professional Full UI
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="CellGuardAI Pro", layout="wide", initial_sidebar_state="expanded")

def clean(df):
    df = df.rename(columns=lambda x: x.strip().lower().replace(" ","_"))
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="ignore")
    cells = [c for c in df.columns if c.startswith("cell")]
    temps = [c for c in df.columns if c.startswith("temp")]
    return df, cells, temps

def feats(df,cells,temps):
    df["voltage_ma"] = df["voltage"].rolling(10,min_periods=1).mean()
    df["voltage_roc"] = df["voltage"].diff().fillna(0)
    df["temp_avg"] = df[temps].mean(axis=1) if temps else np.nan
    df["temp_diff"] = df[temps].max(axis=1)-df[temps].min(axis=1) if temps else np.nan
    df["cell_diff"] = df[cells].max(axis=1)-df[cells].min(axis=1) if cells else np.nan
    return df

def ai(df):
    f=["voltage_ma","voltage_roc","temp_avg","temp_diff","cell_diff"]
    df["anomaly"]=0
    try:
        iso=IsolationForest(contamination=0.05)
        df["anomaly"]=iso.fit_predict(df[f].fillna(0))
        df["anomaly"]=df["anomaly"].map({1:0,-1:1})
    except:
        pass
    df["health"]=100-(df["cell_diff"]/df["cell_diff"].max()*70 if df["cell_diff"].max()!=0 else 0)
    df["health"]=df["health"].clip(0,100)
    return df

def gauge(score):
    fig=go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        gauge={"axis":{"range":[0,100]}},
        title={"text":"Health Score"}))
    fig.update_layout(height=300)
    return fig

def main():
    st.title("⚡ CellGuardAI — Professional Dashboard")
    file=st.file_uploader("Upload EV Bench CSV")
    if not file:
        return
    df=pd.read_csv(file)
    df,cells,temps=clean(df)
    df=feats(df,cells,temps)
    df=ai(df)

    c1,c2,c3=st.columns(3)
    with c1:
        st.plotly_chart(gauge(df["health"].mean()),use_container_width=True)
    with c2:
        st.metric("Avg Voltage",f"{df['voltage'].mean():.2f}")
        st.metric("Avg Temp",f"{df['temp_avg'].mean():.2f}")
    with c3:
        st.metric("Cell Imbalance",f"{df['cell_diff'].iloc[-1]:.2f}")
        st.metric("Anomaly %",f"{df['anomaly'].mean()*100:.2f}%")

    tab1,tab2,tab3,tab4=st.tabs(["Health Trend","Cells","Temperature","Data"])
    with tab1:
        st.line_chart(df["health"])
    with tab2:
        if cells:
            st.plotly_chart(px.bar(df[cells].iloc[-1]),use_container_width=True)
    with tab3:
        for t in temps:
            st.line_chart(df[t])
    with tab4:
        st.dataframe(df)

main()
