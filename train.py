"""Run once:  python train.py
Trains the models, writes metrics + figures + the deployed model."""
import json, warnings
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.inspection import permutation_importance
from core import *

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")
RS = 42

df = add_features(load_raw())
df_roll = add_rolling_stability(df)
y = df[TARGET]

# ---------- EDA figures ----------
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
y.value_counts()[CLASS_ORDER].plot.bar(ax=ax[0], color=["#d9534f", "#f0ad4e", "#5cb85c"]); ax[0].set_title("Class distribution"); ax[0].set_xlabel("")
for c, col in zip(CLASS_ORDER, ["#d9534f", "#f0ad4e", "#5cb85c"]):
    sns.kdeplot(df.loc[y == c, "error_rate"], ax=ax[1], label=c, color=col, fill=True, alpha=.3)
ax[1].set_title("Error rate by efficiency class"); ax[1].legend()
for c, col in zip(CLASS_ORDER, ["#d9534f", "#f0ad4e", "#5cb85c"]):
    sns.kdeplot(df.loc[y == c, "speed"], ax=ax[2], label=c, color=col, fill=True, alpha=.3)
ax[2].set_title("Production speed by efficiency class"); ax[2].legend()
plt.tight_layout(); plt.savefig("figures/eda_overview.png", dpi=130); plt.close()

plt.figure(figsize=(7, 5.5))
sns.scatterplot(data=df.sample(6000, random_state=RS), x="speed", y="error_rate", hue=TARGET, hue_order=CLASS_ORDER,
                palette={"Low": "#d9534f", "Medium": "#f0ad4e", "High": "#5cb85c"}, s=12, alpha=.7)
plt.axvline(400, ls="--", c="grey"); plt.axvline(200, ls="--", c="grey"); plt.axhline(2, ls="--", c="grey"); plt.axhline(5, ls="--", c="grey")
plt.title("The label is a rule on speed and error rate"); plt.tight_layout(); plt.savefig("figures/rule_scatter.png", dpi=130); plt.close()

plt.figure(figsize=(9, 7))
corr_cols = SENSOR + NETWORK + QUALITY + PRODUCTION
sns.heatmap(df[corr_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, annot_kws={"size": 8})
plt.title("Correlation of raw features"); plt.tight_layout(); plt.savefig("figures/correlation.png", dpi=130); plt.close()

mode_tab = pd.crosstab(df["Operation_Mode"], y, normalize="index")[CLASS_ORDER] * 100
mode_tab.plot.bar(stacked=True, figsize=(6.5, 4), color=["#d9534f", "#f0ad4e", "#5cb85c"])
plt.ylabel("% of records"); plt.title("Efficiency mix by operation mode"); plt.xticks(rotation=0)
plt.tight_layout(); plt.savefig("figures/mode_mix.png", dpi=130); plt.close()

# ---------- time-ordered split (first 80% train, last 20% test) ----------
cut = int(len(df) * 0.8)
tr, te = df.iloc[:cut], df.iloc[cut:]
ytr, yte = y.iloc[:cut], y.iloc[cut:]

def models(light=False):
    rf_kw = dict(n_estimators=40, min_samples_leaf=300, max_depth=8) if light else dict(n_estimators=80, min_samples_leaf=10)
    return {
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")),
        "Random Forest": RandomForestClassifier(**rf_kw, class_weight="balanced", n_jobs=-1, random_state=RS),
        "Gradient Boosting": HistGradientBoostingClassifier(max_iter=100, learning_rate=0.1, class_weight="balanced", random_state=RS),
    }

def evaluate(numeric, label, light=False):
    Xtr, Xte = model_matrix(tr, numeric), model_matrix(te, numeric)
    Xall = model_matrix(df, numeric)
    out, fitted = {}, {}
    for name, m in models(light).items():
        m.fit(Xtr, ytr)
        p = m.predict(Xte)
        cv = cross_val_score(models(light)[name], Xall, y, cv=TimeSeriesSplit(3), scoring="f1_macro", n_jobs=1)
        out[name] = dict(accuracy=accuracy_score(yte, p), balanced_accuracy=balanced_accuracy_score(yte, p),
                         macro_f1=f1_score(yte, p, average="macro"),
                         cv_macro_f1_mean=float(cv.mean()), cv_macro_f1_std=float(cv.std()),
                         confusion=confusion_matrix(yte, p, labels=CLASS_ORDER).tolist())
        fitted[name] = m
        print(label, name, {k: round(v, 4) for k, v in out[name].items() if k != "confusion"})
    return out, fitted

resA, fitA = evaluate(NUMERIC_FULL, "A-all")
resB, fitB = evaluate(NUMERIC_NO_PROD, "B-noprod", light=True)

majority = float((yte == "Low").mean())

# Extension: do rolling stability indicators add anything to scenario B?
dr = df_roll
cut_r = int(len(dr) * 0.8)
colsR = NUMERIC_NO_PROD + ["temp_roll_std", "vib_roll_std"]
rfR = RandomForestClassifier(n_estimators=40, min_samples_leaf=300, max_depth=8, class_weight="balanced", n_jobs=-1, random_state=RS)
rfR.fit(model_matrix(dr.iloc[:cut_r], colsR), dr[TARGET].iloc[:cut_r])
pR = rfR.predict(model_matrix(dr.iloc[cut_r:], colsR))
rolling_f1 = f1_score(dr[TARGET].iloc[cut_r:], pR, average="macro")
rolling_acc = accuracy_score(dr[TARGET].iloc[cut_r:], pR)

# confusion matrices figure for scenario A random forest and scenario B random forest
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
for a, (title, res) in zip(ax, [("A: all features (Random Forest)", resA), ("B: no speed/error (Random Forest)", resB)]):
    sns.heatmap(np.array(res["Random Forest"]["confusion"]), annot=True, fmt="d", cmap="Blues", ax=a,
                xticklabels=CLASS_ORDER, yticklabels=CLASS_ORDER); a.set_title(title); a.set_xlabel("Predicted"); a.set_ylabel("Actual")
plt.tight_layout(); plt.savefig("figures/confusion.png", dpi=130); plt.close()

# ---------- feature importance (permutation, on test data) ----------
def perm(model, numeric, n=8000):
    Xte = model_matrix(te, numeric).sample(n, random_state=RS)
    r = permutation_importance(model, Xte, yte.loc[Xte.index], n_repeats=3, scoring="f1_macro", random_state=RS, n_jobs=-1)
    return pd.Series(r.importances_mean, index=Xte.columns).sort_values(ascending=False)

impA = perm(fitA["Random Forest"], NUMERIC_FULL)
impB = perm(fitB["Random Forest"], NUMERIC_NO_PROD)
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
impA.head(10)[::-1].plot.barh(ax=ax[0], color="#2b6cb0"); ax[0].set_title("A: top drivers (all features)")
impB.head(10)[::-1].plot.barh(ax=ax[1], color="#718096"); ax[1].set_title("B: top drivers (no speed/error)")
for a in ax: a.set_xlabel("Drop in macro-F1 when shuffled")
plt.tight_layout(); plt.savefig("figures/importance.png", dpi=130); plt.close()

# group impact: network vs sensor vs quality vs production (scenario A, permuting a whole group)
def group_impact(model, numeric):
    Xte = model_matrix(te, numeric).sample(8000, random_state=RS); yt = yte.loc[Xte.index]
    base = f1_score(yt, model.predict(Xte), average="macro"); rng = np.random.default_rng(RS); res = {}
    groups = {"Production (speed, error rate)": PRODUCTION, "Quality (defect rate)": QUALITY,
              "Sensors (temp, vibration, power, maint.)": SENSOR, "6G network (latency, packet loss)": NETWORK}
    for g, cols in groups.items():
        cols = [c for c in cols if c in Xte.columns]
        if not cols: continue
        drops = []
        for _ in range(3):
            Xp = Xte.copy(); idx = rng.permutation(len(Xp)); Xp[cols] = Xp[cols].values[idx]
            drops.append(base - f1_score(yt, model.predict(Xp), average="macro"))
        res[g] = float(np.mean(drops))
    return res
groupA = group_impact(fitA["Random Forest"], NUMERIC_FULL)
groupB = group_impact(fitB["Random Forest"], NUMERIC_NO_PROD)
pd.Series(groupA).sort_values().plot.barh(figsize=(8, 3.5), color="#2b6cb0")
plt.xlabel("Drop in macro-F1 when group is shuffled"); plt.title("Impact by feature group (scenario A)")
plt.tight_layout(); plt.savefig("figures/group_impact.png", dpi=130); plt.close()

# ---------- deploy: Random Forest, scenario A ----------
final = fitA["Random Forest"]
X_full = model_matrix(df, NUMERIC_FULL)
final.fit(X_full, y)                       # refit on all data for deployment
joblib.dump({"model": final, "columns": list(X_full.columns), "medians": X_full.median().to_dict(),
             "classes": list(final.classes_)}, "models/efficiency_model.joblib")

def jclean(o):
    if isinstance(o, dict): return {k: jclean(v) for k, v in o.items()}
    if isinstance(o, (np.floating,)): return float(o)
    return o
json.dump(jclean({
    "n_rows": len(df), "n_machines": int(df.Machine_ID.nunique()), "date_min": str(df.datetime.min()), "date_max": str(df.datetime.max()),
    "class_counts": y.value_counts().to_dict(), "majority_baseline_test": majority,
    "scenarioA": resA, "scenarioB": resB, "rolling_extension": {"macro_f1": rolling_f1, "accuracy": rolling_acc},
    "importanceA": impA.head(10).to_dict(), "importanceB": impB.head(10).to_dict(),
    "groupA": groupA, "groupB": groupB,
    "mode_mix": mode_tab.round(2).to_dict(),
    "rule_check": classification_report(yte, fitA["Random Forest"].predict(model_matrix(te, NUMERIC_FULL)), output_dict=True),
}), open("models/metrics.json", "w"), indent=1)
print("DONE")
