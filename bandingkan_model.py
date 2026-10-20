"""Machine Learning: bandingkan 4 model klasifikasi pada dataset breast cancer (bawaan sklearn)."""
import numpy as np, pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

X, y = load_breast_cancer(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.25, stratify=y, random_state=0)
models = {
    "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
    "KNN (k=7)": make_pipeline(StandardScaler(), KNeighborsClassifier(7)),
    "Random Forest": RandomForestClassifier(300, random_state=0),
    "Gradient Boosting": GradientBoostingClassifier(random_state=0),
}
rows, fig = [], plt.figure(figsize=(12, 4.5))
a1, a2 = fig.add_subplot(121), fig.add_subplot(122)
for n, m in models.items():
    cv = cross_val_score(m, Xtr, ytr, cv=5, scoring="accuracy")
    m.fit(Xtr, ytr); p = m.predict_proba(Xte)[:, 1]
    rows.append(dict(model=n, cv_acc=cv.mean(), cv_std=cv.std(), test_acc=(m.predict(Xte) == yte).mean(), test_auc=roc_auc_score(yte, p)))
    f, t, _ = roc_curve(yte, p); a2.plot(f, t, label=f"{n} (AUC {rows[-1]['test_auc']:.3f})")
df = pd.DataFrame(rows).sort_values("cv_acc", ascending=False); print(df.round(4).to_string(index=False)); df.to_csv("hasil_model.csv", index=False)
a1.barh(df.model, df.cv_acc, xerr=df.cv_std, color="#0072B2"); a1.set(xlim=(.9, 1), title="Akurasi CV 5-fold (±std)")
a2.plot([0, 1], [0, 1], "--", color="gray"); a2.legend(fontsize=8); a2.set(title="Kurva ROC (data uji)", xlabel="FPR", ylabel="TPR")
plt.tight_layout(); plt.savefig("ml_hasil.png", dpi=120); print("Grafik: ml_hasil.png")
