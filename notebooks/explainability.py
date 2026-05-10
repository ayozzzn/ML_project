import numpy as np
import pandas as pd
import shap
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import label_binarize
from pathlib import Path

PROCESSED_DIR = Path("../processed")
MODELS_DIR = Path("../models")
MODELS_DIR.mkdir(exist_ok=True)

df = pd.read_csv(PROCESSED_DIR / "features_ml.csv")
scaler = joblib.load(MODELS_DIR / "scaler.pkl")
label_encoder = joblib.load(MODELS_DIR / "label_encoder.pkl")

feature_cols = [c for c in df.columns if c not in ["emotion", "language", "dataset"]]
X = scaler.transform(df[feature_cols].values.astype(np.float32))
y = label_encoder.transform(df["emotion"].values)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# RandomForest — SHAP поддерживает multiclass RF
print("Training Random Forest surrogate model for SHAP...")
rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
acc = accuracy_score(y_test, rf.predict(X_test))
print(f"Random Forest accuracy: {acc:.4f}")

# Сохраняем как gbm_for_shap.pkl — именно это имя ждёт inference.py
joblib.dump(rf, MODELS_DIR / "gbm_for_shap.pkl")
print("✅ Saved gbm_for_shap.pkl")

print("\nComputing SHAP values (may take 2-3 min)...")
explainer = shap.TreeExplainer(rf)

N_SHAP = min(200, len(X_test))
X_shap = X_test[:N_SHAP]

shap_values = explainer.shap_values(X_shap)

joblib.dump(explainer, MODELS_DIR / "shap_explainer.pkl")
print("✅ SHAP explainer saved")

if isinstance(shap_values, list):
    mean_importance = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
elif shap_values.ndim == 3:
    mean_importance = np.abs(shap_values).mean(axis=(0, 2))
else:
    mean_importance = np.abs(shap_values).mean(axis=0)

print(f"mean_importance shape: {mean_importance.shape}")  # должно быть (n_features,)
feat_importance = pd.Series(mean_importance, index=feature_cols).sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(10, 8))
feat_importance.head(20).plot(kind="barh", ax=ax, color="steelblue")
ax.invert_yaxis()
ax.set_title("Top 20 Most Important Acoustic Features (SHAP)", fontsize=13)
ax.set_xlabel("Mean |SHAP value|")
ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig(MODELS_DIR / "shap_feature_importance.png", dpi=150)
plt.show()
print("💾 Saved feature importance plot")

top_features = feat_importance.head(15).index.tolist()
top_indices  = [feature_cols.index(f) for f in top_features]
classes      = list(label_encoder.classes_)
target_class = "anxiety" if "anxiety" in classes else classes[0]
class_idx    = classes.index(target_class)

if isinstance(shap_values, list):
    shap_for_class = shap_values[class_idx][:, top_indices]
elif shap_values.ndim == 3:
    shap_for_class = shap_values[:, top_indices, class_idx]
else:
    shap_for_class = shap_values[:, top_indices]

plt.figure(figsize=(10, 6))
shap.summary_plot(shap_for_class, X_shap[:, top_indices],
                  feature_names=top_features, show=False)
plt.title(f"SHAP Summary — {target_class} class")
plt.tight_layout()
plt.savefig(MODELS_DIR / "shap_summary_plot.png", dpi=150)
plt.show()
print("💾 Saved SHAP summary plot")

y_pred = rf.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
cm_normalized = cm.astype(float) / cm.sum(axis=1, keepdims=True)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_, ax=axes[0])
axes[0].set_title("Confusion Matrix (counts)")
axes[0].set_ylabel("True label"); axes[0].set_xlabel("Predicted label")

sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="YlOrRd",
            xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_, ax=axes[1])
axes[1].set_title("Confusion Matrix (normalized)")
axes[1].set_ylabel("True label"); axes[1].set_xlabel("Predicted label")

plt.tight_layout()
plt.savefig(MODELS_DIR / "confusion_matrix.png", dpi=150)
plt.show()
print("💾 Saved confusion matrix")

print("\nPer-class results:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

print("\n📊 Generating per-class SHAP waterfall charts...")
n_classes = len(label_encoder.classes_)
n_cols = 3
n_rows = (n_classes + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
axes = axes.flatten()

for ci, class_name in enumerate(label_encoder.classes_):
    true_mask    = y_test[:N_SHAP] == ci
    correct_mask = rf.predict(X_shap) == y_test[:N_SHAP]
    candidates   = np.where(true_mask & correct_mask)[0]
    if len(candidates) == 0:
        candidates = np.where(true_mask)[0]
    if len(candidates) == 0:
        axes[ci].text(0.5, 0.5, f"No samples\nfor {class_name}",
                      ha="center", va="center", transform=axes[ci].transAxes)
        axes[ci].set_title(class_name); continue

    sample_idx = candidates[0]
    if isinstance(shap_values, list):
        sv = shap_values[ci][sample_idx]
    elif shap_values.ndim == 3:
        sv = shap_values[sample_idx, :, ci]
    else:
        sv = shap_values[sample_idx]

    top_n = 8
    top_feat_idx   = np.argsort(np.abs(sv))[::-1][:top_n]
    top_feat_names = [feature_cols[i] for i in top_feat_idx]
    top_feat_vals  = sv[top_feat_idx]
    colors = ["#d73027" if v > 0 else "#4575b4" for v in top_feat_vals]
    y_pos  = np.arange(len(top_feat_names))
    axes[ci].barh(y_pos, top_feat_vals, color=colors, alpha=0.8)
    axes[ci].set_yticks(y_pos)
    axes[ci].set_yticklabels(top_feat_names, fontsize=9)
    axes[ci].axvline(x=0, color="black", linewidth=0.8)
    axes[ci].set_title(f"SHAP: {class_name}", fontsize=11)
    axes[ci].set_xlabel("SHAP value")
    axes[ci].grid(axis="x", alpha=0.3)

for idx in range(n_classes, len(axes)):
    axes[idx].set_visible(False)

plt.suptitle("Per-Class Feature Contributions (SHAP Waterfall)", fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(MODELS_DIR / "shap_per_class_waterfall.png", dpi=150, bbox_inches="tight")
plt.show()
print("💾 Saved per-class SHAP waterfall chart")

print("\n📊 Computing calibration curves...")
y_proba    = rf.predict_proba(X_test)
y_test_bin = label_binarize(y_test, classes=range(n_classes))

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfectly calibrated")
for ci, cname in enumerate(label_encoder.classes_):
    try:
        prob_true, prob_pred = calibration_curve(y_test_bin[:, ci], y_proba[:, ci], n_bins=8)
        ax.plot(prob_pred, prob_true, marker="o", label=cname, linewidth=1.5, markersize=5)
    except Exception as e:
        print(f"  Skipping calibration for {cname}: {e}")
ax.set_xlabel("Mean predicted probability"); ax.set_ylabel("Fraction of positives")
ax.set_title("Calibration Curves (Reliability Diagram)")
ax.legend(loc="upper left", fontsize=9); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(MODELS_DIR / "calibration_curves.png", dpi=150)
plt.show()
print("💾 Saved calibration curves")

if "language" in df.columns:
    print("\n🌍 Per-language performance breakdown:")
    from sklearn.metrics import f1_score
    for lang in df["language"].unique():
        mask   = df["language"] == lang
        X_lang = scaler.transform(df[mask][feature_cols].values.astype(np.float32))
        y_lang = label_encoder.transform(df[mask]["emotion"].values)
        if len(y_lang) < 10: continue
        preds = rf.predict(X_lang)
        print(f"  {lang.upper()}: Accuracy={accuracy_score(y_lang, preds):.4f}, "
              f"Macro F1={f1_score(y_lang, preds, average='macro', zero_division=0):.4f} (n={len(y_lang)})")

FEATURE_NAMES_HUMAN = {
    "pitch_mean": "average pitch", "pitch_std": "pitch variability",
    "pitch_range": "pitch range", "energy_mean": "voice energy (loudness)",
    "energy_std": "energy variability", "zcr_mean": "voice clarity",
    "spectral_centroid_mean": "brightness of voice", "speaking_rate": "speaking speed",
    "pause_ratio": "pause duration", "num_pauses": "number of pauses",
    "chroma_mean": "tonal quality", "mel_mean": "spectral character",
}
for i in range(40):
    FEATURE_NAMES_HUMAN[f"mfcc_{i}_mean"] = f"vocal texture #{i+1}"
    FEATURE_NAMES_HUMAN[f"mfcc_{i}_std"]  = f"vocal texture var #{i+1}"
for i in range(7):
    FEATURE_NAMES_HUMAN[f"spectral_contrast_{i}_mean"] = f"spectral contrast #{i+1}"

joblib.dump(FEATURE_NAMES_HUMAN, MODELS_DIR / "feature_names_human.pkl")
print("\n✅ feature_names_human.pkl saved")

sample_feat = X_test[0]
proba       = rf.predict_proba(sample_feat.reshape(1, -1))[0]
predicted   = label_encoder.classes_[proba.argmax()]

sv_test = explainer.shap_values(sample_feat.reshape(1, -1))
ci      = list(label_encoder.classes_).index(predicted)
shap_for_pred = sv_test[ci][0] if isinstance(sv_test, list) else sv_test[0, :, ci] if sv_test.ndim == 3 else sv_test[0]
top_idx = np.argsort(np.abs(shap_for_pred))[::-1][:3]
print(f"\n🧪 Test explanation for '{predicted}':")
for i in top_idx:
    feat = feature_cols[i]; val = shap_for_pred[i]
    human = FEATURE_NAMES_HUMAN.get(feat, feat.replace("_", " "))
    print(f"  Your {human} was {'noticeably' if abs(val) >= 0.05 else 'slightly'} {'higher' if val > 0 else 'lower'} than usual")

print("\n✅ SHAP analysis complete!")
print("   Saved: gbm_for_shap.pkl, shap_explainer.pkl, feature_names_human.pkl")