# Machine Learning Perbandingan Model

Membandingkan Logistic Regression, KNN, Random Forest, dan Gradient Boosting pada dataset *breast cancer* (bawaan scikit-learn, tanpa unduhan)
dengan **cross-validation 5-fold**, akurasi uji, dan **ROC-AUC**.

```bash
pip install numpy pandas scikit-learn matplotlib
python bandingkan_model.py
```
Keluaran: tabel peringkat model, `hasil_model.csv`, `ml_hasil.png`.

**Lanjutan:** `GridSearchCV` untuk tuning, SHAP untuk interpretasi, ganti dataset sendiri (CSV).
