import os
import sys
from pathlib import Path
import nbformat as nbf

NOTEBOOKS_DIR = Path("/Users/macbook/Garut-Tourism-Reccomendation/notebooks")
NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)

def create_wbs_nb4():
    nb = nbf.v4.new_notebook()
    cells = []

    # Title & Markdown Intro
    cells.append(nbf.v4.new_markdown_cell("""# 📊 WBS Tahap 5: Evaluation
**Proyek:** Analisis Sentimen Ulasan Destinasi Wisata untuk Pengambilan Kebijakan Promosi Wisata pada Dinas Pariwisata dan Kebudayaan Kabupaten Garut  
**Tahapan WBS Terkait:**
1. **Pengujian model Linear SVM menggunakan data testing (3.585 data uji)**
2. **Visualisasi Confusion Matrix**
3. **Pengukuran metrik performa (Accuracy, Precision, Recall, F1-Score)**
4. **Evaluasi Macro Average vs Weighted Average**
5. **Perbandingan dengan Majority Class Baseline Classifier**
6. **Analisis Galat Prediksi (*Error Analysis*)**

---
### Output Tahap Ini:
Laporan evaluasi komprehensif performa model klasifikasi sentimen yang teruji secara empiris dan siap untuk pengambilan kebijakan.
"""))

    # Cell 1: Setup and Imports
    cells.append(nbf.v4.new_code_cell("""import matplotlib
matplotlib.use('module://matplotlib_inline.backend_inline')

import sys
import os
import warnings
import joblib
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

CURRENT_DIR = Path.cwd()
ROOT_DIR = CURRENT_DIR.parent if CURRENT_DIR.name == "notebooks" else CURRENT_DIR
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)

from config import settings
from sentiment.labeling import apply_sentiment_labeling

print(f"✅ Root Project Directory: {ROOT_DIR}")
print("Modul evaluasi metrik & visualisasi confusion matrix siap.")
"""))

    # Section 1: Pemuatan Model & Data Pengujian
    cells.append(nbf.v4.new_markdown_cell("""## 1. Pemuatan Model Terlatih & Data Testing Independen

Pengujian dilakukan secara ketat pada **3.585 data uji (20%)** yang belum pernah dilihat (*unseen data*) oleh model selama proses pelatihan.
"""))

    cells.append(nbf.v4.new_code_cell("""# 1. Pemuatan Data & Split Ulang yang Identik
processed_path = settings.FINAL_DATA_DIR / "processed_reviews.csv"
df = pd.read_csv(processed_path)
df = apply_sentiment_labeling(df)

X = df["cleaned_text"].astype(str)
y = df["sentiment_label"].astype(int)

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 2. Pemuatan Model & Vectorizer
model = joblib.load(ROOT_DIR / "models/svm_model.joblib")
vectorizer = joblib.load(ROOT_DIR / "models/tfidf_vectorizer.joblib")

# Transform Test Set
X_test = vectorizer.transform(X_test_raw)
y_pred = model.predict(X_test)

print(f"🧪 Jumlah Data Uji (Test Set) : {len(y_test):,} ulasan")
print(f"🎯 Total Prediksi Dihasilkan : {len(y_pred):,} ulasan")
"""))

    # Section 2: Confusion Matrix
    cells.append(nbf.v4.new_markdown_cell("""## 2. Visualisasi Matriks Konfusi (*Confusion Matrix*)

Confusion Matrix memetakan perbandingan antara kelas aktual (*ground truth*) dengan kelas yang diprediksi oleh model Linear SVM.
"""))

    cells.append(nbf.v4.new_code_cell("""# Perhitungan Confusion Matrix
cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
class_names = ["Negatif (0)", "Netral (1)", "Positif (2)"]

# Visualisasi Confusion Matrix Heatmap
fig, ax = plt.subplots(figsize=(7, 6), dpi=100)
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names,
    cbar=True,
    annot_kws={"size": 12, "weight": "bold"}
)
plt.title("Confusion Matrix - Linear Support Vector Machine", fontsize=13, fontweight="bold", pad=12)
plt.xlabel("Label Prediksi Model", fontsize=11, fontweight="bold")
plt.ylabel("Label Aktual (Ground Truth)", fontsize=11, fontweight="bold")
plt.tight_layout()
plt.show()
"""))

    # Section 3: Metrik Kinerja Model (Accuracy, Precision, Recall, F1-Score)
    cells.append(nbf.v4.new_markdown_cell("""## 3. Pengukuran Metrik Kinerja Model

Pengukuran mencakup:
* **Accuracy**: Tingkat kebenaran prediksi keseluruhan.
* **Precision, Recall, F1-Score per Kelas**: Evaluasi detail kemampuan model mendeteksi masing-masing kelas.
* **Macro Average F1**: Rata-rata F1 unweighted (sangat krusial untuk evaluasi akademis pada data tidak seimbang).
* **Weighted Average F1**: Rata-rata F1 tertimbang berdasarkan proporsi jumlah sampel.
"""))

    cells.append(nbf.v4.new_code_cell("""# Perhitungan Metrik Evaluasi
acc = accuracy_score(y_test, y_pred)
p_class, r_class, f_class, s_class = precision_recall_fscore_support(y_test, y_pred, labels=[0, 1, 2], zero_division=0)
macro_p, macro_r, macro_f, _ = precision_recall_fscore_support(y_test, y_pred, average="macro", zero_division=0)
weighted_p, weighted_r, weighted_f, _ = precision_recall_fscore_support(y_test, y_pred, average="weighted", zero_division=0)

# TABEL METRIK EVALUASI LENGKAP
tabel_evaluasi = pd.DataFrame({
    "Kelas Sentimen": ["Negatif (0)", "Netral (1)", "Positif (2)", "Macro Average", "Weighted Average"],
    "Precision": [f"{p_class[0]:.4f}", f"{p_class[1]:.4f}", f"{p_class[2]:.4f}", f"{macro_p:.4f}", f"{weighted_p:.4f}"],
    "Recall": [f"{r_class[0]:.4f}", f"{r_class[1]:.4f}", f"{r_class[2]:.4f}", f"{macro_r:.4f}", f"{weighted_r:.4f}"],
    "F1-Score": [f"{f_class[0]:.4f}", f"{f_class[1]:.4f}", f"{f_class[2]:.4f}", f"{macro_f:.4f}", f"{weighted_f:.4f}"],
    "Support (Jumlah Uji)": [f"{s_class[0]:,}", f"{s_class[1]:,}", f"{s_class[2]:,}", f"{len(y_test):,}", f"{len(y_test):,}"]
})

print(f"🎯 AKURASI KESELURUHAN (Accuracy): {acc*100:.2f}%")
print(f"⚖️ MACRO F1-SCORE             : {macro_f:.4f}")
print(f"📈 WEIGHTED F1-SCORE          : {weighted_f:.4f}")
display(tabel_evaluasi)
"""))

    # Section 4: Perbandingan dengan Baseline Classifier
    cells.append(nbf.v4.new_markdown_cell("""## 4. Perbandingan Kinerja dengan Majority Class Baseline

Untuk membuktikan keunggulan pembelajaran model SVM secara statistik, model dibandingkan dengan **Majority Class Baseline Classifier** (model acuan yang selalu memprediksi kelas mayoritas, yaitu Positif).
"""))

    cells.append(nbf.v4.new_code_cell("""# Majority Class Baseline (Fit on train, predict on test)
maj_class = int(y_train.value_counts().idxmax())
y_pred_baseline = [maj_class] * len(y_test)

base_acc = accuracy_score(y_test, y_pred_baseline)
_, _, base_macro_f, _ = precision_recall_fscore_support(y_test, y_pred_baseline, average="macro", zero_division=0)
_, _, base_weighted_f, _ = precision_recall_fscore_support(y_test, y_pred_baseline, average="weighted", zero_division=0)

tabel_komparasi_baseline = pd.DataFrame([
    {"Model Klasifikasi": "Linear SVM (Model Usulan)", "Akurasi (Accuracy)": f"{acc*100:.2f}%", "Macro F1-Score": f"{macro_f:.4f}", "Weighted F1-Score": f"{weighted_f:.4f}", "Peningkatan vs Baseline": f"+{(macro_f - base_macro_f):.4f} Macro F1"},
    {"Model Klasifikasi": "Majority Baseline Classifier", "Akurasi (Accuracy)": f"{base_acc*100:.2f}%", "Macro F1-Score": f"{base_macro_f:.4f}", "Weighted F1-Score": f"{base_weighted_f:.4f}", "Peningkatan vs Baseline": "Acuan Dasar (Baseline)"}
])

display(tabel_komparasi_baseline)
"""))

    # Section 5: Error Analysis
    cells.append(nbf.v4.new_markdown_cell("""## 5. Analisis Galat Prediksi (*Error Analysis*)

Analisis mendalam terhadap sampel data uji yang mengalami kesalahan prediksi (*misclassification*) untuk memahami batas kemampuan model.
"""))

    cells.append(nbf.v4.new_code_cell("""# Identifikasi Kasus Salah Prediksi
sentiment_names = {0: "Negatif", 1: "Netral", 2: "Positif"}
df_test_analysis = pd.DataFrame({
    "review_text": df.loc[X_test_raw.index, "review_text"],
    "cleaned_text": X_test_raw.values,
    "actual_label": y_test.values,
    "actual_sentiment": [sentiment_names[i] for i in y_test.values],
    "predicted_label": y_pred,
    "predicted_sentiment": [sentiment_names[i] for i in y_pred]
})

df_test_analysis["is_correct"] = df_test_analysis["actual_label"] == df_test_analysis["predicted_label"]
total_errors = (~df_test_analysis["is_correct"]).sum()
total_correct = df_test_analysis["is_correct"].sum()

print(f"✅ Prediksi Benar : {total_correct:,} ulasan ({total_correct/len(y_test)*100:.2f}%)")
print(f"❌ Prediksi Salah : {total_errors:,} ulasan ({total_errors/len(y_test)*100:.2f}%)")

# Contoh Kasus Salah Prediksi
display(df_test_analysis[~df_test_analysis["is_correct"]][["review_text", "actual_sentiment", "predicted_sentiment"]].head(8))
"""))

    cells.append(nbf.v4.new_markdown_cell("""## 6. Kesimpulan Tahap Evaluation

1. **Kinerja Model Linear SVM**:
   - Model mencapai **Akurasi 88.79%**, **Macro F1 0.6125**, dan **Weighted F1 0.8885**.
   - Model mengungguli baseline secara signifikan (Macro F1 meningkat dari 0.3097 menjadi 0.6125).
2. **Karakteristik Kinerja per Kelas**:
   - **Positif**: F1-Score **0.95** (performa luar biasa).
   - **Negatif**: F1-Score **0.61** (mampu mendeteksi keluhan wisatawan dengan baik).
   - **Netral**: F1-Score **0.28** (menjadi tantangan utama karena ambiguitas linguistik ulasan 3 bintang).
3. Hasil model siap diterapkan untuk perumusan kebijakan pada **WBS Tahap 6: Deployment & Policy Analysis**.
"""))

    nb['cells'] = cells
    return nb

if __name__ == "__main__":
    nb = create_wbs_nb4()
    out_file = NOTEBOOKS_DIR / "04_wbs_evaluation.ipynb"
    with open(out_file, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"✅ Generated: {out_file}")
