
# Compact Enhanced CellGuardAI - dynamic cell support + info buttons + heatmap + SOH + PDF update
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
import io, re
# --- HOTFIX: ensure ensure_cols_exist exists (temporary) ---
import numpy as np
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CellGuardAI_hotfix")
logger.info("Applying hotfix for ensure_cols_exist (temporary)")

if "ensure_cols_exist" not in globals():
    def ensure_cols_exist(df, needed):
        """
        Defensive fallback: add missing columns with NaN.
        Temporary — restore real implementation later.
        """
        for c in needed:
            if c not in df.columns:
                df[c] = np.nan
        return df

# Optional quick debug (comment out if noisy)
try:
    import streamlit as st
    st.write("DEBUG: ensure_cols_exist present:", "ensure_cols_exist" in globals())
except Exception:
    print("DEBUG: ensure_cols_exist present:", "ensure_cols_exist" in globals())

st.set_page_config(page_title="CellGuard.Ai - Dashboard", layout="wide")

def gen_sample_data(n=400, seed=42):
    np.random.seed(seed)
    t = np.arange(n)
    v = 3.7 + 0.05 * np.sin(t/50) + np.random.normal(0,0.005,n)
    i = 1.5 + 0.3 * np.sin(t/30) + np.random.normal(0,0.05,n)
    temp = 30 + 3*np.sin(t/60) + np.random.normal(0,0.3,n)
    soc = np.clip(80 + 10*np.sin(t/80) + np.random.normal(0,1,n), 0, 100)
    cycle = t//50
    df = pd.DataFrame({'time':t,'voltage':v,'current':i,'temperature':temp,'soc':soc,'cycle':cycle})
    # add sample cell columns for demo
    for c in range(1,13):
        df[f'Cell{c}'] = (v/12.0) + np.random.normal(0,0.005,n)
    return df

def normalize_cols(df):
    df = df.copy()
    simple = {c: "".join(ch for ch in c.lower() if ch.isalnum()) for c in df.columns}
    patt = {'voltage':['volt','vcell','cellv','packv'],'current':['curr','amp','amps','current'],'temperature':['temp','temperature'],'soc':['soc'],'cycle':['cycle'],'time':['time','timestamp']}
    cmap = {}
    used=set()
    for target, keys in patt.items():
        for orig, simplified in simple.items():
            if orig in used: continue
            if any(k in simplified for k in keys):
                cmap[target]=orig; used.add(orig); break
    rename = {orig:targ for targ,orig in cmap.items()}
    df = df.rename(columns=rename)
    return df, cmap

def detect_cell_columns(df):
    pattern = re.compile(r'(?i)\b(?:cell|vcell|c)(?:[_\-\s]*)?0*([0-9]{1,4})\b')
    found=[]; unknowns=[]
    for i,col in enumerate(df.columns):
        lowered=str(col).lower()
        m=pattern.search(lowered)
        if m:
            idx=int(m.group(1)); found.append((col,idx,i))
        else:
            if 'cell' in lowered:
                unknowns.append((col,None,i))
    found_sorted=sorted(found,key=lambda x:(x[1],x[2]))
    for col,_,pos in unknowns:
        found_sorted.append((col,100000+pos,pos))
    return [c for c,_,_ in found_sorted]

def compute_cell_metrics(df, cell_cols, pack_voltage_col='voltage'):
    if not cell_cols: return df
    cells = df[cell_cols].apply(pd.to_numeric, errors='coerce')
    mean_cell = cells.mean(skipna=True).mean()
    if not np.isnan(mean_cell) and mean_cell > 10:
        cells = cells/1000.0
    df['pack_from_cells']=cells.sum(axis=1)
    df['C_Low']=cells.min(axis=1); df['C_High']=cells.max(axis=1); df['C_Diff']=df['C_High']-df['C_Low']
    df['C_N_Low']=cells.idxmin(axis=1); df['C_N_High']=cells.idxmax(axis=1)
    df.attrs['per_cell_var']=cells.var(skipna=True); df.attrs['cell_columns']=list(cells.columns)
    if pack_voltage_col not in df.columns or df[pack_voltage_col].isna().all():
        df[pack_voltage_col]=df['pack_from_cells']
    with np.errstate(divide='ignore', invalid='ignore'):
        df['imbalance_index']=df['C_Diff']/df['C_High'].replace(0,np.nan)
    return df

def make_features(df, window=10):
    df = df.copy()
    df = df.fillna(np.nan)
    if df.get('voltage').notna().sum()>0:
        df['voltage_ma']=df['voltage'].rolling(window, min_periods=1).mean()
        df['voltage_roc']=df['voltage'].diff().fillna(0)
        df['voltage_var']=df['voltage'].rolling(window, min_periods=1).var().fillna(0)
    if 'temperature' in df.columns and df['temperature'].notna().sum()>0:
        df['temp_ma']=df['temperature'].rolling(window,min_periods=1).mean()
    if 'soc' in df.columns and df['soc'].notna().sum()>0:
        df['soc_ma']=df['soc'].rolling(window,min_periods=1).mean(); df['soc_roc']=df['soc'].diff().fillna(0)
    if 'C_Diff' in df.columns:
        df['c_diff_ma']=df['C_Diff'].rolling(window,min_periods=1).mean()
    # simple risk label
    cond = pd.Series(False, index=df.index)
    if 'temperature' in df.columns:
        tmean=df['temperature'].mean(); tstd=df['temperature'].std()
        if not np.isnan(tmean) and not np.isnan(tstd):
            cond = cond | (df['temperature'] > tmean + 2*tstd)
    if 'voltage_roc' in df.columns:
        cond = cond | (df['voltage_roc'] < -0.01)
    if 'C_Diff' in df.columns:
        cond = cond | (df['C_Diff'] > 0.1)
    df['risk_label']=np.where(cond,1,0)
    return df

def run_models(df, contamination=0.05):
    df = df.copy()
    possible=["voltage","current","temperature","soc","pack_from_cells","C_Diff","C_Low","C_High","voltage_ma","voltage_roc","soc_roc","voltage_var","temp_ma","cycle","c_diff_ma"]
    features=[f for f in possible if f in df.columns and df[f].notna().sum()>0]
    df['anomaly_flag']=0; df['risk_pred']=0; df['battery_health_score']=50.0
    if len(features)>=2 and df[features].dropna().shape[0]>=30:
        try:
            iso=IsolationForest(n_estimators=100, contamination=contamination, random_state=42)
            X=df[features].fillna(df[features].median())
            iso.fit(X); df['anomaly_flag']=iso.predict(X).map({1:0,-1:1})
        except Exception:
            df['anomaly_flag']=0
    if 'risk_label' in df.columns and df['risk_label'].nunique()>1:
        clf_feats=[f for f in features if f in df.columns]
        if len(clf_feats)>=2:
            try:
                Xc=df[clf_feats].fillna(df[clf_feats].median()); yc=df['risk_label']
                tree=DecisionTreeClassifier(max_depth=4, random_state=42); tree.fit(Xc,yc); df['risk_pred']=tree.predict(Xc)
            except Exception:
                df['risk_pred']=df['risk_label']
        else:
            df['risk_pred']=df['risk_label']
    else:
        tseries=df.get('temperature', pd.Series(np.nan,index=df.index)); tmean=tseries.mean(); tstd=tseries.std()
        tth = tmean + 2*tstd if not np.isnan(tmean) and not np.isnan(tstd) else np.nan
        cond_temp = (tseries > tth) if not np.isnan(tth) else False
        df['risk_pred']=np.where((df.get('anomaly_flag',0)==1) | cond_temp,1,0)
    base=pd.Series(0.0, index=df.index)
    if 'voltage_ma' in df.columns and df['voltage_ma'].notna().sum()>0:
        vm=df['voltage_ma'].fillna(method='ffill').fillna(df['voltage'].median() if 'voltage' in df.columns else 3.7); base += (vm.max() - vm)
    elif 'voltage' in df.columns:
        v=df['voltage'].fillna(df['voltage'].median()); base += (v.max() - v)
    else:
        base += 0.5
    if 'temperature' in df.columns and df['temperature'].notna().sum()>0:
        t=df['temperature'].fillna(df['temperature'].median()); base += (t - t.min())/10.0
    base = base + df.get('anomaly_flag',0)*1.0 + df.get('risk_pred',0)*0.8
    hp = base.values
    hp = np.array(hp,dtype=float); hp_norm = (hp - hp.min())/(hp.max()-hp.min()+1e-9); health_comp = 1 - hp_norm
    score = (0.6*health_comp) + (0.25*(1-df.get('risk_pred',0))) + (0.15*(1-df.get('anomaly_flag',0)))
    df['battery_health_score'] = (score*100).clip(0,100)
    return df

def simple_recommend(row):
    sc=row.get('battery_health_score',50); rp=row.get('risk_pred',0); an=row.get('anomaly_flag',0)
    if sc>85 and rp==0 and an==0: return "Healthy — normal operation."
    if 70<sc<=85: return "Watch — avoid deep discharge & fast-charge this cycle."
    if 50<sc<=70: return "Caution — restrict fast charging; allow cooling intervals."
    return "Critical — reduce load, stop fast charging, schedule inspection."

def make_pdf(df_out, avg_score, anomaly_pct, alerts, recs, verdict_text):
    # minimal PDF using text (keeps dependencies light); fallback if reportlab not present
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas
        from reportlab.lib.units import mm
        buf=io.BytesIO(); c=canvas.Canvas(buf,pagesize=A4); w,h=A4; x=20*mm; y=h-20*mm
        c.setFont('Helvetica-Bold',14); c.drawString(x,y,'CellGuard.Ai - Report'); y-=10*mm
        c.setFont('Helvetica',10); c.drawString(x,y,f'Avg Health Score: {avg_score:.1f}/100'); y-=6*mm
        c.drawString(x,y,f'Anomaly Rate: {anomaly_pct:.2f}%'); y-=6*mm
        c.drawString(x,y,verdict_text); y-=8*mm
        if alerts:
            c.drawString(x,y,'Top Alerts:'); y-=6*mm
            for a in alerts[:6]:
                c.drawString(x+6*mm,y,f"- {a['title']}: {a['detail']}"); y-=5*mm
        buf.seek(0); return buf.read()
    except Exception:
        return b""

def info_button(label, text, key):
    col1, col2 = st.columns([0.95, 0.05])
    with col1: st.write(label)
    with col2:
        if st.button("i", key=key):
            st.info(text)

def strong_sanitize(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    # drop fully empty cols
    df = df.dropna(axis=1, how='all')
    # remove repeated header rows
    def looks_like_header(row):
        cnt=0
        for col in df.columns:
            try:
                if str(row.get(col,'')).strip().lower() == str(col).strip().lower():
                    cnt+=1
            except Exception:
                pass
        return cnt >= max(2, len(df.columns)//4)
    try:
        header_mask = df.apply(looks_like_header, axis=1)
        if header_mask.any(): df = df.loc[~header_mask].reset_index(drop=True)
    except Exception:
        pass
    # coerce numeric-like columns if obvious
    return df

# ----------------------
# Main UI
# ----------------------
def main():
    st.title("CellGuard.Ai")
    st.write("Predictive battery intelligence — drop your CSV (college bench logs supported).")
    st.sidebar.header("Config")
    data_mode = st.sidebar.radio("Data source", ["Sample data","Upload CSV"])
    contamination = st.sidebar.slider("Anomaly sensitivity", 0.01, 0.15, 0.05, 0.01)
    window = st.sidebar.slider("Rolling window", 5, 30, 10)
    if data_mode=="Sample data":
        df_raw = gen_sample_data(n=400)
        st.sidebar.success("Using sample data")
    else:
        uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])
        if uploaded is None: st.warning("Upload CSV or choose sample."); st.stop()
        try:
            df_raw = pd.read_csv(uploaded)
        except Exception:
            df_raw = pd.read_csv(uploaded, encoding='latin1')
        df_raw = strong_sanitize(df_raw)
    df_raw, col_map = normalize_cols(df_raw)
    cell_cols = detect_cell_columns(df_raw)
    if cell_cols:
        st.sidebar.success(f"Detected {len(cell_cols)} cell columns (e.g. {cell_cols[:6]})")
    df_raw = compute_cell_metrics(df_raw, cell_cols, pack_voltage_col='voltage')
    # detect RemAh & FullCap for SOH
    rem_col=None; full_col=None
    for c in df_raw.columns:
        if re.search(r'(?i)rem', c) and re.search(r'(?i)ah', c): rem_col=c
        if re.search(r'(?i)full', c) and re.search(r'(?i)cap', c): full_col=c
    if rem_col and full_col:
        try:
            df_raw['SOH'] = pd.to_numeric(df_raw[rem_col], errors='coerce') / pd.to_numeric(df_raw[full_col], errors='coerce') * 100
        except Exception:
            df_raw['SOH']=np.nan
    df_raw = ensure_cols_exist(df_raw, ["voltage","current","temperature","soc","cycle","time"])
    df_feat = make_features(df_raw, window=window)
    df_out = run_models(df_feat, contamination=contamination)
    df_out['recommendation'] = df_out.apply(simple_recommend, axis=1)
    avg_score = float(df_out['battery_health_score'].mean()) if not df_out['battery_health_score'].isnull().all() else 50.0
    anomaly_pct = float(df_out['anomaly_flag'].mean()*100) if 'anomaly_flag' in df_out.columns else 0.0
    label,color = ('HEALTHY','#2ecc71') if avg_score>=85 else (('WATCH','#f39c12') if avg_score>=60 else ('CRITICAL','#e74c3c'))
    alerts = basic_alerts(df_out)
    recs = top_recs_from_df(df_out, n=8)
    pdf_bytes = make_pdf(df_out, avg_score, anomaly_pct, alerts, recs, "Auto-generated verdict")
    left,mid,right = st.columns([1.4,1.4,1])
    with left:
        st.markdown("### Battery Health"); safe_plot(make_gauge_figure(avg_score),"gauge_health")
    with mid:
        st.markdown("### Pack Status")
        st.markdown(f"<span style='background:{color};color:#fff;padding:6px 10px;border-radius:8px;font-weight:600'>{label}</span>", unsafe_allow_html=True)
        st.metric("Avg Health Score", f"{avg_score:.1f}/100", delta=f"{(avg_score-85):.1f} vs ideal")
        st.write(f"- Anomalies: **{anomaly_pct:.1f}%**")
        st.write(f"- Data points: **{len(df_out)}**")
        st.write(f"- Detected cell columns: {len(cell_cols)}")
        st.write(f"- Mapped cols: {', '.join(list(col_map.keys())) if col_map else 'none'}")
    with right:
        st.download_button("⬇️ Download processed CSV", df_out.to_csv(index=False).encode('utf-8'), "CellGuardAI_Output.csv", "text/csv")
    st.subheader("Combined Verdict and PDF")
    if avg_score<60: st.error("Combined verdict: Immediate action required.")
    elif avg_score<75: st.warning("Combined verdict: Monitor closely.")
    else: st.success("Combined verdict: Pack is healthy.")
    st.download_button("📄 Download PDF Report", data=pdf_bytes, file_name="CellGuardAI_Report.pdf", mime="application/pdf")
    st.markdown("---")
    # tabs with heatmap and worst cells
    tab1,tab2,tab3 = st.tabs(["AI Insights","Traditional BMS","Data"])
    with tab1:
        st.subheader("AI-Based Insights"); st.write("Health Score Over Time"); safe_plot(px.line(df_out, x='time', y='battery_health_score'), "h1")
        cell_columns = df_out.attrs.get('cell_columns', [])
        if cell_columns:
            st.write("Cell voltage heatmap (downsampled for performance)")
            sample = df_out[cell_columns].copy()
            if sample.shape[0]>400: sample = sample.iloc[::max(1, sample.shape[0]//400), :]
            try:
                fig = px.imshow(sample.T, labels={'x':'Time','y':'Cell','color':'Voltage'}, x=sample.index, y=sample.columns, aspect='auto')
                safe_plot(fig, "cell_heatmap")
                pcv = df_out.attrs.get('per_cell_var', None)
                if pcv is not None:
                    worst = pcv.sort_values(ascending=False).head(8)
                    safe_plot(px.bar(x=worst.index, y=worst.values, labels={'x':'Cell','y':'Var'}), "worst_cells")
            except Exception as e:
                st.write("Heatmap error:",e)
    with tab2:
        st.subheader("Traditional BMS"); safe_plot(px.line(df_out, x='time', y='voltage'), "trad_v")
    with tab3:
        st.header("Processed Data"); st.dataframe(df_out.head(300))
    st.caption("CellGuard.Ai — flexible CSV ingestion for college bench logs.")

if __name__ == '__main__':
    main()
