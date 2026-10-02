"""Smart-factory efficiency dashboard.   Run:  streamlit run app.py
Needs models/efficiency_model.joblib (created by:  python train.py)"""
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from core import (load_raw, add_features, model_matrix, NUMERIC_FULL, CLASS_ORDER, TARGET,
                  SENSOR, NETWORK, QUALITY, PRODUCTION, MODES, RENAME)

st.set_page_config(page_title="Factory Efficiency AI", page_icon="🏭", layout="wide")
COLORS = {"Low": "#d9534f", "Medium": "#f0ad4e", "High": "#5cb85c"}
ICON = {"Low": "🔴", "Medium": "🟠", "High": "🟢"}

LABELS = {"temperature": "Temperature (°C)", "vibration": "Vibration (Hz)", "power": "Power (kW)",
          "latency": "Network latency (ms)", "packet_loss": "Packet loss (%)", "defect_rate": "Defect rate (%)",
          "speed": "Production speed (units/hr)", "maint_score": "Maintenance score", "error_rate": "Error rate (%)"}


# ------------------------------------------------------------------ loading
@st.cache_resource
def load_model():
    return joblib.load("models/efficiency_model.joblib")


@st.cache_data
def load_data():
    bundle = load_model()
    d = add_features(load_raw())
    X = model_matrix(d, NUMERIC_FULL).reindex(columns=bundle["columns"])
    proba = bundle["model"].predict_proba(X)
    classes = list(bundle["classes"])
    d["Predicted"] = [classes[i] for i in proba.argmax(1)]
    d["Confidence"] = proba.max(1)
    for i, c in enumerate(classes):
        d[f"p_{c}"] = proba[:, i]
    d["date"] = d["datetime"].dt.date
    d["Network quality"] = pd.qcut(d["network_reliability"], 3, labels=["Poor", "Fair", "Good"])
    return d


def predict_df(rows):
    """rows: DataFrame with raw columns + Operation_Mode. Returns (label, confidence, probs dict)."""
    bundle = load_model()
    r = add_features(rows)
    X = model_matrix(r, NUMERIC_FULL).reindex(columns=bundle["columns"])
    p = bundle["model"].predict_proba(X)[0]
    classes = list(bundle["classes"])
    return classes[int(p.argmax())], float(p.max()), dict(zip(classes, p))


def explain_record(row, top=8):
    """Occlusion explanation: replace one input at a time by the typical (median) value and see how much the
    probability of the predicted class changes. Positive bar = this value pushes TOWARDS the predicted class."""
    bundle = load_model(); med = bundle["medians"]
    base_label, _, base_p = predict_df(row)
    out = {}
    for col in RENAME.values():
        r2 = row.copy(); r2[col] = med.get(col, r2[col].iloc[0])
        _, _, p2 = predict_df(r2)
        out[LABELS[col]] = base_p[base_label] - p2[base_label]
    s = pd.Series(out).sort_values(key=abs, ascending=False).head(top)
    return base_label, s


def rule_text(speed, err):
    """Plain-language reason, based on the thresholds the model discovered (see paper)."""
    if speed >= 400 and err <= 2:
        return f"Speed {speed:.0f} is ≥ 400 and error rate {err:.2f}% is ≤ 2% → **High**."
    if speed >= 200 and err <= 5:
        why = []
        if speed < 400: why.append(f"speed {speed:.0f} is below 400")
        if err > 2: why.append(f"error rate {err:.2f}% is above 2%")
        return f"Speed ≥ 200 and error ≤ 5%, but {' and '.join(why)} → **Medium**."
    why = []
    if speed < 200: why.append(f"speed {speed:.0f} is below 200")
    if err > 5: why.append(f"error rate {err:.2f}% is above 5%")
    return f"{' and '.join(why).capitalize()} → **Low**."


try:
    df = load_data()
except FileNotFoundError:
    st.error("Model file not found. Run `python train.py` first, then restart the app.")
    st.stop()

# ------------------------------------------------------------------ sidebar filters
st.sidebar.title("🏭 Filters")
machines = sorted(df["Machine_ID"].unique())
machine_sel = st.sidebar.multiselect("Machine(s)", machines, default=[], placeholder="All machines")
mode_sel = st.sidebar.selectbox("Operation mode", ["All"] + MODES)
net_sel = st.sidebar.selectbox("Network quality", ["All", "Good", "Fair", "Poor"],
                               help="Terciles of the network reliability score (latency + packet loss).")
dmin, dmax = df["date"].min(), df["date"].max()
win = st.sidebar.date_input("Time window", (dmin, dmax), min_value=dmin, max_value=dmax)
if isinstance(win, tuple) and len(win) == 2:
    d0, d1 = win
else:
    d0, d1 = dmin, dmax

f = df[(df["date"] >= d0) & (df["date"] <= d1)]
if machine_sel: f = f[f["Machine_ID"].isin(machine_sel)]
if mode_sel != "All": f = f[f["Operation_Mode"] == mode_sel]
if net_sel != "All": f = f[f["Network quality"] == net_sel]
st.sidebar.caption(f"{len(f):,} of {len(df):,} records selected")

st.title("AI-Based Manufacturing Efficiency Classification")
st.caption("Real-time High / Medium / Low efficiency labels with confidence scores and explanations.")
if f.empty:
    st.warning("No records match the current filters. Widen the filters in the sidebar.")
    st.stop()

tab1, tab2, tab3, tab4 = st.tabs(["⚡ Prediction", "🔧 Machine insights", "🔍 Explainability", "📊 Operational monitoring"])

# ------------------------------------------------------------------ TAB 1
with tab1:
    st.subheader("Efficiency prediction")
    c1, c2, c3, c4 = st.columns(4)
    mix = f["Predicted"].value_counts(normalize=True)
    c1.metric("Records", f"{len(f):,}")
    c2.metric("🟢 High", f"{mix.get('High', 0):.1%}")
    c3.metric("🟠 Medium", f"{mix.get('Medium', 0):.1%}")
    c4.metric("🔴 Low", f"{mix.get('Low', 0):.1%}")

    src = st.radio("Input source", ["A record from the data", "Enter values manually"], horizontal=True)
    if src == "A record from the data":
        recs = f.sort_values("datetime").reset_index(drop=True)
        i = st.slider("Pick a record (time-ordered)", 0, len(recs) - 1, len(recs) - 1)
        rec = recs.iloc[i]
        st.write(f"**{rec['datetime']}** · Machine **{int(rec['Machine_ID'])}** · Mode **{rec['Operation_Mode']}** · "
                 f"Recorded label: **{rec[TARGET]}**")
        base = rec[list(RENAME.values())].astype(float).to_frame().T
        base["Operation_Mode"] = rec["Operation_Mode"]
    else:
        cols = st.columns(5)
        defaults = {"temperature": 60.0, "vibration": 2.5, "power": 5.5, "latency": 25.0, "packet_loss": 2.5,
                    "defect_rate": 5.0, "speed": 275.0, "maint_score": 0.5, "error_rate": 7.5}
        maxes = {"temperature": 90.0, "vibration": 5.0, "power": 10.0, "latency": 50.0, "packet_loss": 5.0,
                 "defect_rate": 10.0, "speed": 500.0, "maint_score": 1.0, "error_rate": 15.0}
        vals = {}
        for idx, k in enumerate(defaults):
            col = cols[idx % 5]
            vals[k] = col.number_input(LABELS[k], 0.0, maxes[k] * 1.5, defaults[k], step=maxes[k] / 100)
        mode = st.selectbox("Operation mode of this reading", MODES)
        base = pd.DataFrame([vals]); base["Operation_Mode"] = mode

    st.markdown("**Metric sensitivity sliders** – change a metric by ±% and watch the prediction react")
    sc = st.columns(6)
    sens = {}
    for col, k in zip(sc, ["speed", "error_rate", "temperature", "vibration", "latency", "packet_loss"]):
        sens[k] = col.slider(LABELS[k].split(" (")[0], -50, 50, 0, 5, format="%d%%", key=f"s_{k}")
    adj = base.copy()
    for k, pct in sens.items():
        adj[k] = adj[k].astype(float) * (1 + pct / 100)

    lab0, conf0, p0 = predict_df(base)
    lab1, conf1, p1 = predict_df(adj)
    a, b = st.columns(2)
    with a:
        st.markdown("#### Original reading")
        st.markdown(f"## {ICON[lab0]} {lab0}")
        st.progress(conf0, text=f"Confidence {conf0:.0%}")
    with b:
        st.markdown("#### After sensitivity changes")
        st.markdown(f"## {ICON[lab1]} {lab1}")
        st.progress(conf1, text=f"Confidence {conf1:.0%}")
    st.bar_chart(pd.DataFrame({"Original": p0, "Adjusted": p1}).loc[CLASS_ORDER], y_label="Probability")
    if conf1 < 0.7:
        st.info("Confidence is below 70% – the reading sits close to a class boundary. Treat the label with care.")

    st.markdown("#### Confidence across the selected records")
    hist = np.histogram(f["Confidence"], bins=np.linspace(0.3, 1.0, 15))
    st.bar_chart(pd.DataFrame({"records": hist[0]}, index=[f"{x:.2f}" for x in hist[1][:-1]]), y_label="Records")

# ------------------------------------------------------------------ TAB 2
with tab2:
    st.subheader("Machine-level insights")
    prof = (f.groupby("Machine_ID")
              .agg(records=("Predicted", "size"),
                   High=("Predicted", lambda s: (s == "High").mean()),
                   Medium=("Predicted", lambda s: (s == "Medium").mean()),
                   Low=("Predicted", lambda s: (s == "Low").mean()),
                   avg_speed=("speed", "mean"), avg_error=("error_rate", "mean"), avg_conf=("Confidence", "mean"))
              .sort_values("High", ascending=False))
    left, right = st.columns([3, 2])
    with left:
        st.markdown("**Machine efficiency profile** (share of time in each class)")
        st.bar_chart(prof[CLASS_ORDER], color=[COLORS[c] for c in CLASS_ORDER], y_label="Share of records", height=320)
    with right:
        st.markdown("**Best / worst machines by High share**")
        show = prof[["records", "High", "Low", "avg_speed", "avg_error"]].copy()
        st.dataframe(pd.concat([show.head(5), show.tail(5)]).style.format(
            {"High": "{:.1%}", "Low": "{:.1%}", "avg_speed": "{:.0f}", "avg_error": "{:.2f}"}), height=320)

    st.markdown("**Efficiency trend over time** (daily share of predicted class)")
    trend = f.groupby(["date", "Predicted"]).size().unstack(fill_value=0).reindex(columns=CLASS_ORDER, fill_value=0)
    trend = trend.div(trend.sum(1), axis=0)
    st.line_chart(trend, color=[COLORS[c] for c in CLASS_ORDER], y_label="Share of records")

    st.markdown("**Historical classification pattern – hour of day**")
    hr = f.assign(hour=f["datetime"].dt.hour).groupby(["hour", "Predicted"]).size().unstack(fill_value=0)
    hr = hr.reindex(columns=CLASS_ORDER, fill_value=0); hr = hr.div(hr.sum(1), axis=0)
    st.bar_chart(hr, color=[COLORS[c] for c in CLASS_ORDER], y_label="Share of records")
    st.caption("Tip: pick one machine in the sidebar to see that machine's own trend.")

# ------------------------------------------------------------------ TAB 3
with tab3:
    st.subheader("Why is a record Low / Medium / High?")
    try:
        metrics = json.load(open("models/metrics.json"))
    except FileNotFoundError:
        metrics = None
    if metrics:
        imp = pd.Series(metrics["importanceA"]).sort_values(ascending=True)
        st.markdown("**Global feature importance** (how much accuracy drops if a feature is scrambled)")
        st.bar_chart(imp, horizontal=True, height=330, x_label="Drop in macro-F1")
        st.caption("Production speed and error rate (and ratios built from them) dominate. Sensor and network "
                   "features carry almost no signal in this dataset.")

    st.markdown("**Explain one record**")
    recs = f.sort_values("datetime").reset_index(drop=True)
    j = st.slider("Record", 0, len(recs) - 1, len(recs) - 1, key="exp_rec")
    r = recs.iloc[j]
    row = r[list(RENAME.values())].astype(float).to_frame().T
    row["Operation_Mode"] = r["Operation_Mode"]
    lab, contrib = explain_record(row)
    st.markdown(f"### {ICON[lab]} Predicted **{lab}** (confidence {r['Confidence']:.0%})")
    st.markdown(rule_text(float(r["speed"]), float(r["error_rate"])))
    st.bar_chart(contrib, horizontal=True, height=300, x_label=f"Pushes toward '{lab}' (+) / away (−)")
    st.caption("Method: each input is replaced by its typical value; the bar is the resulting change in the predicted "
               "class probability. No extra libraries needed.")

    st.markdown("**What changed between two moments?**")
    k1, k2 = st.columns(2)
    ia = k1.number_input("Earlier record #", 0, len(recs) - 1, max(0, j - 10))
    ib = k2.number_input("Later record #", 0, len(recs) - 1, j)
    ra, rb = recs.iloc[int(ia)], recs.iloc[int(ib)]
    chg = pd.DataFrame({"Earlier": [ra[c] for c in RENAME.values()], "Later": [rb[c] for c in RENAME.values()]},
                       index=[LABELS[c] for c in RENAME.values()])
    chg["Change"] = chg["Later"] - chg["Earlier"]
    st.write(f"Earlier: **{ra['Predicted']}** → Later: **{rb['Predicted']}**")
    st.dataframe(chg.style.format("{:.2f}"))

# ------------------------------------------------------------------ TAB 4
with tab4:
    st.subheader("Operational monitoring")
    a, b = st.columns(2)
    with a:
        st.markdown("**Efficiency by operation mode**")
        m = f.groupby(["Operation_Mode", "Predicted"]).size().unstack(fill_value=0).reindex(columns=CLASS_ORDER, fill_value=0)
        st.bar_chart(m.div(m.sum(1), axis=0), color=[COLORS[c] for c in CLASS_ORDER], y_label="Share of records")
    with b:
        st.markdown("**Efficiency by network quality**")
        n = f.groupby(["Network quality", "Predicted"], observed=True).size().unstack(fill_value=0).reindex(columns=CLASS_ORDER, fill_value=0)
        st.bar_chart(n.div(n.sum(1), axis=0), color=[COLORS[c] for c in CLASS_ORDER], y_label="Share of records")

    st.markdown("**Network vs sensor vs production impact**")
    if metrics:
        g = pd.Series(metrics["groupA"]).sort_values()
        st.bar_chart(g, horizontal=True, height=260, x_label="Drop in macro-F1 when this group is scrambled")
        gb = metrics["scenarioB"]["Random Forest"]
        st.info(f"Without speed and error rate, a Random Forest using only sensors + network + quality scores "
                f"macro-F1 {gb['macro_f1']:.2f} (1.00 = perfect) and its predictions are spread almost randomly "
                f"across the three classes. Sensor and network data do not explain efficiency in this dataset.")

    st.markdown("**Average readings by predicted class**")
    show_cols = list(RENAME.values())
    st.dataframe(f.groupby("Predicted")[show_cols].mean().reindex(CLASS_ORDER)
                 .rename(columns=LABELS).T.style.format("{:.2f}"))
