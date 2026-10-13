"""Handwritten Digit Recognition + Label Quality Review (Python only).
Dataset: scikit-learn digits (1,797 8x8 grayscale images, 10 classes), bundled with scikit-learn.

Part 1: train and compare classifiers.
Part 2: use out-of-fold model predictions to FLAG SUSPICIOUS LABELS (a data-quality / annotation-review workflow).
        The original labels are clean, so we deliberately corrupt 5% of them and measure how many the review catches.
Part 3: measure agreement between two models and the labels with Cohen's kappa (like inter-annotator agreement).
"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report, cohen_kappa_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

RS = 42
rng = np.random.default_rng(RS)
digits = load_digits()
X, y = digits.data, digits.target
images = digits.images
print("Images:", X.shape, "| Classes:", np.unique(y))

# ---------- EDA ----------
fig, ax = plt.subplots(2, 10, figsize=(12, 3))
for d in range(10):
    for r in range(2):
        idx = np.where(y == d)[0][r]
        ax[r, d].imshow(images[idx], cmap="gray_r"); ax[r, d].axis("off")
        if r == 0: ax[r, d].set_title(str(d))
plt.suptitle("Sample images per digit"); plt.tight_layout(); plt.savefig("sample_digits.png", dpi=150); plt.close()
print("Class counts:", dict(zip(*np.unique(y, return_counts=True))))

# ---------- Part 1: model comparison ----------
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RS)
cv = StratifiedKFold(5, shuffle=True, random_state=RS)
models = {
    "KNN": Pipeline([("s", StandardScaler()), ("m", KNeighborsClassifier())]),
    "Logistic Regression": Pipeline([("s", StandardScaler()), ("m", LogisticRegression(max_iter=3000))]),
    "Random Forest": RandomForestClassifier(n_estimators=300, random_state=RS),
    "SVM (RBF)": Pipeline([("s", StandardScaler()), ("m", SVC(probability=True, random_state=RS))]),
}
rows = []
for n, m in models.items():
    s = cross_val_score(m, X_tr, y_tr, cv=cv, scoring="accuracy")
    rows.append({"Model": n, "CV accuracy": round(s.mean(), 4), "Std": round(s.std(), 4)})
cmp_df = pd.DataFrame(rows).sort_values("CV accuracy", ascending=False)
print("\n", cmp_df.to_string(index=False)); cmp_df.to_csv("model_comparison.csv", index=False)

best_name = cmp_df.iloc[0]["Model"]; best = models[best_name].fit(X_tr, y_tr)
pred = best.predict(X_te)
test_acc = accuracy_score(y_te, pred)
print(f"\nBest model: {best_name} | test accuracy = {test_acc:.4f}")
print(classification_report(y_te, pred))
ConfusionMatrixDisplay.from_predictions(y_te, pred, cmap="Blues", colorbar=False)
plt.title(f"{best_name} - test set"); plt.tight_layout(); plt.savefig("confusion_matrix.png", dpi=150); plt.close()
cm = pd.crosstab(pd.Series(y_te, name="true"), pd.Series(pred, name="pred"))
errs = [(int(t), int(p), int(cm.loc[t, p])) for t in cm.index for p in cm.columns if t != p and cm.loc[t, p] > 0]
errs.sort(key=lambda e: -e[2]); print("Most common confusions (true, predicted, count):", errs[:3])

# ---------- Part 2: label-quality review with injected noise ----------
n_noisy = int(0.05 * len(y))
noisy_idx = rng.choice(len(y), n_noisy, replace=False)
y_noisy = y.copy()
for i in noisy_idx:
    y_noisy[i] = rng.choice([c for c in range(10) if c != y[i]])      # wrong label on purpose
is_noisy = np.zeros(len(y), bool); is_noisy[noisy_idx] = True

# out-of-fold probabilities: every image is scored by a model that never saw it in training
oof_proba = cross_val_predict(models["SVM (RBF)"], X, y_noisy, cv=cv, method="predict_proba")
conf_in_label = oof_proba[np.arange(len(y)), y_noisy]          # model's confidence in the GIVEN label
suspicion = 1 - conf_in_label                                     # higher = more suspicious
order = np.argsort(-suspicion)
res = {}
for k in (n_noisy, 2 * n_noisy):
    flagged = order[:k]
    caught = int(is_noisy[flagged].sum())
    res[f"top_{k}"] = {"flagged": k, "noisy_caught": caught,
                       "precision": round(caught / k, 3), "recall": round(caught / n_noisy, 3)}
print(f"\nInjected {n_noisy} wrong labels (5%). Review results:", json.dumps(res, indent=1))

# Review the 12 most suspicious images
fig, ax = plt.subplots(2, 6, figsize=(11, 5.8))
for a, i in zip(ax.ravel(), order[:12]):
    a.imshow(images[i], cmap="gray_r"); a.axis("off")
    guess = oof_proba[i].argmax()
    a.set_title(f"label {y_noisy[i]} | model {guess}\n{'WRONG LABEL' if is_noisy[i] else 'label ok'}", fontsize=8,
                color="crimson" if is_noisy[i] else "black")
plt.suptitle("12 most suspicious labels flagged for human review"); plt.tight_layout(h_pad=2.5)
plt.savefig("flagged_labels.png", dpi=150); plt.close()

# ---------- Part 3: agreement (Cohen's kappa) ----------
p_svm = cross_val_predict(models["SVM (RBF)"], X, y, cv=cv)
p_rf = cross_val_predict(models["Random Forest"], X, y, cv=cv)
kappa = {"svm_vs_labels": round(cohen_kappa_score(y, p_svm), 4),
         "rf_vs_labels": round(cohen_kappa_score(y, p_rf), 4),
         "svm_vs_rf": round(cohen_kappa_score(p_svm, p_rf), 4)}
print("\nCohen's kappa:", kappa)

# Cases where the two models agree with each other but disagree with the given label
both_disagree = np.where((p_svm == p_rf) & (p_svm != y))[0]
print("Images where both models disagree with the label:", len(both_disagree))

json.dump({"best_model": best_name, "test_accuracy": round(test_acc, 4), "cv_table": cmp_df.to_dict("records"),
           "top_confusions": errs[:3], "noise_injected": n_noisy, "label_review": res, "kappa": kappa,
           "both_models_disagree_with_label": int(len(both_disagree))}, open("results.json", "w"), indent=2)
