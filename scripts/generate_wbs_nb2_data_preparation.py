import os
import sys
from pathlib import Path
import nbformat as nbf

NOTEBOOKS_DIR = Path("/Users/macbook/Garut-Tourism-Reccomendation/notebooks")
NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)

def create_wbs_nb2():
    nb = nbf.v4.new_notebook()
    cells = []

    # Title & Markdown Intro
    cells.append(nbf.v4.new_markdown_cell("""# 🧹 WBS Tahap 3: Data Preparation
**Proyek:** Analisis Sentimen Ulasan Destinasi Wisata untuk Pengambilan Kebijakan Promosi Wisata pada Dinas Pariwisata dan Kebudayaan Kabupaten Garut  
**Tahapan WBS Terkait:**
1. **Cleaning (menghapus URL, simbol, angka, dan karakter yang tidak diperlukan)**
2. **Case Folding (mengubah huruf menjadi huruf kecil)**
3. **Tokenizing (memecah teks menjadi token/kata)**
4. **Stopword Removal (menghapus kata yang tidak relevan / custom stopword)**
5. **Stemming (mengubah kata berimbuhan menjadi kata dasar dengan PySastrawi)**
6. **Ekstraksi fitur menggunakan TF-IDF (Unigram & Bigram)**

---
### Output Tahap Ini:
Data teks siap analisis dan direpresentasikan dalam matriks fitur TF-IDF untuk tahap pemodelan (*modeling*).
"""))

    # Cell 1: Setup and Imports
    cells.append(nbf.v4.new_code_cell("""import matplotlib
matplotlib.use('module://matplotlib_inline.backend_inline')

import sys
import os
import warnings
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
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)

from config import settings
from config.constants import CUSTOM_INDONESIAN_STOPWORDS
from preprocessing.cleaning import clean_text
from preprocessing.case_folding import case_folding
from preprocessing.tokenization import tokenize
from preprocessing.stopword_removal import remove_stopwords, ALL_STOPWORDS
from preprocessing.stemming import stem_tokens
from sentiment.labeling import apply_sentiment_labeling, get_label_from_rating

print(f"✅ Root Project Directory: {ROOT_DIR}")
print("Modul preprocessing, pelabelan, dan TF-IDF siap digunakan.")
"""))

    # Section 1: Pemuatan, Penggabungan, Deduplikasi, & Validasi Kualitas
    cells.append(nbf.v4.new_markdown_cell("""## 1. Penggabungan Data, Deduplikasi Composite Key, & Validasi Kualitas

Langkah awal data preparation adalah menggabungkan berkas ulasan mentah, membersihkan baris duplikat dengan kunci gabungan (*composite key*), dan menyaring rekaman yang memenuhi syarat kualitas penelitian.
"""))

    cells.append(nbf.v4.new_code_cell("""# 1. Pemuatan & Penggabungan
df_orig = pd.read_csv(settings.RAW_DATA_DIR / "reviews.csv")
repaired_path = settings.RAW_DATA_DIR / "reviews_repaired.csv"
df_rep = pd.read_csv(repaired_path) if repaired_path.exists() else pd.DataFrame()

df_orig["_source"] = "original"
if not df_rep.empty:
    df_rep["_source"] = "repaired"
    df_merged = pd.concat([df_rep, df_orig], ignore_index=True)
else:
    df_merged = df_orig.copy()

total_raw_merged = len(df_merged)

# 2. Deduplikasi Composite Key
composite_cols = ["destination_name", "author", "review_date", "review_text"]
df_temp = df_merged.fillna({col: "" for col in composite_cols})
dup_mask = df_temp.duplicated(subset=composite_cols, keep="first")
duplicate_count = dup_mask.sum()
df_dedup = df_merged[~dup_mask].copy().drop(columns=["_source"], errors="ignore")
total_dedup = len(df_dedup)

# 3. Filter Validitas Teks
df_valid_text = df_dedup.dropna(subset=["review_text"])
df_valid_text = df_valid_text[df_valid_text["review_text"].astype(str).str.strip() != ""]
after_text_filter = len(df_valid_text)

# 4. Filter Validitas Destinasi & Rating
df_valid_dest = df_valid_text.dropna(subset=["destination_name"])
df_valid_dest = df_valid_dest[df_valid_dest["destination_name"].astype(str).str.strip() != ""]
after_dest_filter = len(df_valid_dest)

def is_valid_rating(r):
    try:
        val = float(r)
        return 1.0 <= val <= 5.0
    except (ValueError, TypeError):
        return False

valid_rating_mask = df_valid_dest["rating"].apply(is_valid_rating)
df_validated = df_valid_dest[valid_rating_mask].copy()
df_validated["rating"] = df_validated["rating"].astype(float)
after_rating_filter = len(df_validated)

tabel_tahap_validasi = pd.DataFrame([
    {"Tahap Pembersihan": "1. Data Mentah Digabung (Raw Concat)", "Jumlah Masuk": f"{total_raw_merged:,}", "Dieliminasi": "0", "Jumlah Tersisa": f"{total_raw_merged:,}", "Justifikasi Metodologis": "Menggabungkan reviews.csv dan reviews_repaired.csv"},
    {"Tahap Pembersihan": "2. Deduplikasi Composite Key", "Jumlah Masuk": f"{total_raw_merged:,}", "Dieliminasi": f"{duplicate_count:,}", "Jumlah Tersisa": f"{total_dedup:,}", "Justifikasi Metodologis": "Menghapus duplikasi baris identik dengan memprioritaskan data repaired"},
    {"Tahap Pembersihan": "3. Filter Ulasan Berteks (Non-Empty Text)", "Jumlah Masuk": f"{total_dedup:,}", "Dieliminasi": f"{total_dedup - after_text_filter:,}", "Jumlah Tersisa": f"{after_text_filter:,}", "Justifikasi Metodologis": "Mengeliminasi ulasan rating-only karena tidak memiliki fitur linguistik"},
    {"Tahap Pembersihan": "4. Filter Validitas Rating (1.0 s/d 5.0)", "Jumlah Masuk": f"{after_text_filter:,}", "Dieliminasi": f"{after_text_filter - after_rating_filter:,}", "Jumlah Tersisa": f"{after_rating_filter:,}", "Justifikasi Metodologis": "Memastikan integritas nilai rating numerik standar 1-5 bintang"}
])

display(tahap_validasi := tabel_tahap_validasi)
"""))

    # Section 2: Rangkaian 5 Tahap Text Preprocessing
    cells.append(nbf.v4.new_markdown_cell("""## 2. Rangkaian 5 Tahap Text Preprocessing (Step-by-Step)

Setiap ulasan diproses melalui 5 tahap berurutan:
1. **Cleaning**: Menghilangkan URL, HTML, angka, karakter khusus, tanda baca, dan emoji.
2. **Case Folding**: Menyeragamkan seluruh huruf menjadi huruf kecil (*lowercase*).
3. **Tokenizing**: Memecah kalimat menjadi kumpulan kata individual (*tokens*).
4. **Stopword Removal**: Menghapus kata umum/tugas bahasa Indonesia + stopword spesifik pariwisata Garut.
5. **Stemming (PySastrawi)**: Mentransformasikan kata berimbuhan menjadi kata dasar baku.
"""))

    cells.append(nbf.v4.new_code_cell("""# 5 Contoh Ulasan Nyata untuk Demonstrasi Step-by-Step
contoh_sampel = [
    "Pemandian air panas Cipanas Garut sangat BAGUS!! 🌟 Tempatnya luas & nyaman http://maps.google.com/test",
    "Jalanan menuju lokasi sangat terjal, sempit dan macet parah saat akhir pekan... Tolong diperbaiki!",
    "Makanan di restorannya lumayan enak, pemandangan pegunungan indah sekali, tapi parkirannya mahal.",
    "Tiket masuk terlalu mahal, fasilitas toilet kotor dan tidak terawat! Sangat mengecewakan...",
    "Tempat wisata yang sangat recommended untuk liburan keluarga di kota Garut, sejuk dan asri :)"
]

tahap_demo = []
for teks in contoh_sampel:
    c_clean = clean_text(teks)
    c_case = case_folding(c_clean)
    c_tokens = tokenize(c_case)
    c_stop = remove_stopwords(c_tokens)
    c_stem = stem_tokens(c_stop)
    
    tahap_demo.append({
        "1. Raw Text": teks,
        "2. Cleaning": c_clean,
        "3. Case Folding": c_case,
        "4. Tokenization": str(c_tokens),
        "5. Stopword Removal": str(c_stop),
        "6. Stemming (Sastrawi)": " ".join(c_stem)
    })

df_tahap_demo = pd.DataFrame(tahap_demo)
display(df_tahap_demo)
"""))

    cells.append(nbf.v4.new_code_cell("""# Eksekusi Rangkaian Preprocessing pada Dataset Valid
tqdm.pandas(desc="Menjalankan Preprocessing Lengkap")

processed_path = settings.FINAL_DATA_DIR / "processed_reviews.csv"

if processed_path.exists():
    print("Memuat dataset ulasan terproses yang sudah ada...")
    df_proc = pd.read_csv(processed_path)
else:
    print("Menjalankan pipeline preprocessing ke seluruh dataset...")
    # Jalankan pipeline
    from preprocessing.pipeline import run_preprocessing_pipeline
    df_proc = run_preprocessing_pipeline(df_validated)

print(f"✅ Total Ulasan Pasca Preprocessing & Filter Teks Kosong: {len(df_proc):,} baris")
display(df_proc[["destination_name", "rating", "review_text", "cleaned_text"]].head(5))
"""))

    # Section 3: Lineage Corong Penyusutan Data (Data Lineage)
    cells.append(nbf.v4.new_markdown_cell("""## 3. Corong Garis Keturunan Data (*Data Lineage Funnel*)

Tabel rekapitulasi penyusutan data dari data mentah 36.574 baris hingga dataset akhir bersih siap latih 17.923 baris:
"""))

    cells.append(nbf.v4.new_code_cell("""# TABEL LENGKAP DATA LINEAGE
total_final_valid = len(df_proc)
prep_loss = after_rating_filter - total_final_valid

lineage_table = pd.DataFrame([
    {"Tahap": "1. Data Mentah (Raw Reviews)", "Berkas / Sumber": "reviews.csv + repaired", "Baris Masuk": f"{total_raw_merged:,}", "Baris Keluar": f"{total_raw_merged:,}", "Dibuang": "0", "Sisa (%)": "100.00%", "Justifikasi Metodologis": "Dataset mentah hasil web scraping Playwright di Google Maps"},
    {"Tahap": "2. Deduplikasi Data", "Berkas / Sumber": "In-Memory Composite Key", "Baris Masuk": f"{total_raw_merged:,}", "Baris Keluar": f"{total_dedup:,}", "Dibuang": f"{duplicate_count:,}", "Sisa (%)": f"{(total_dedup/total_raw_merged)*100:.2f}%", "Justifikasi Metodologis": "Menghapus ulasan ganda yang terserap saat double scraping"},
    {"Tahap": "3. Validasi Teks Narasi", "Berkas / Sumber": "Text Null Filter", "Baris Masuk": f"{total_dedup:,}", "Baris Keluar": f"{after_text_filter:,}", "Dibuang": f"{total_dedup - after_text_filter:,}", "Sisa (%)": f"{(after_text_filter/total_raw_merged)*100:.2f}%", "Justifikasi Metodologis": "Mengeluarkan ulasan rating-only (ulasan kosong tanpa teks)"},
    {"Tahap": "4. Validasi Rating Bintang", "Berkas / Sumber": "Rating Range Filter [1-5]", "Baris Masuk": f"{after_text_filter:,}", "Baris Keluar": f"{after_rating_filter:,}", "Dibuang": f"{after_text_filter - after_rating_filter:,}", "Sisa (%)": f"{(after_rating_filter/total_raw_merged)*100:.2f}%", "Justifikasi Metodologis": "Memvalidasi batas numerik rating 1.0 s/d 5.0 bintang"},
    {"Tahap": "5. Filter Preprocessing Loss", "Berkas / Sumber": "processed_reviews.csv", "Baris Masuk": f"{after_rating_filter:,}", "Baris Keluar": f"{total_final_valid:,}", "Dibuang": f"{prep_loss:,}", "Sisa (%)": f"{(total_final_valid/total_raw_merged)*100:.2f}%", "Justifikasi Metodologis": "Mengeluarkan ulasan yang menjadi string kosong pasca stopword removal"}
])

display(lineage_table)
"""))

    # Section 4: Pelabelan Sentimen & Ekstraksi Fitur TF-IDF
    cells.append(nbf.v4.new_markdown_cell("""## 4. Pelabelan Sentimen & Ekstraksi Fitur TF-IDF

1. **Pelabelan Sentimen**:
   - **Negatif (0)**: Rating 1.0 s/d 2.0
   - **Netral (1)**: Rating 3.0
   - **Positif (2)**: Rating 4.0 s/d 5.0
2. **Ekstraksi Fitur TF-IDF (*Term Frequency - Inverse Document Frequency*)**:
   - Pembentukan fitur kombinasi **Unigram & Bigram** (`ngram_range=(1,2)`).
   - Pembatasan ukuran kosakata maksimal **5.000 fitur terpopuler** (`max_features=5000`).
"""))

    cells.append(nbf.v4.new_code_cell("""# 1. Pelabelan Sentimen
df_labeled = apply_sentiment_labeling(df_proc)
sentiment_map = {0: "Negatif", 1: "Netral", 2: "Positif"}
df_labeled["sentiment_name"] = df_labeled["sentiment_label"].map(sentiment_map)

# Distribusi Sentimen Terlabeli
sent_counts = df_labeled["sentiment_name"].value_counts().reindex(["Positif", "Netral", "Negatif"])
tabel_sentimen_dist = pd.DataFrame({
    "Kelas Sentimen": sent_counts.index,
    "Rentang Rating": ["4.0 - 5.0 Bintang", "3.0 Bintang", "1.0 - 2.0 Bintang"],
    "Jumlah Ulasan": [f"{v:,}" for v in sent_counts.values],
    "Persentase": [f"{(v/len(df_labeled))*100:.2f}%" for v in sent_counts.values]
})
display(tabel_sentimen_dist)
"""))

    cells.append(nbf.v4.new_code_cell("""# 2. Ekstraksi Fitur TF-IDF
tfidf_demo = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_tfidf_sample = tfidf_demo.fit_transform(df_labeled["cleaned_text"].astype(str))
feature_names = tfidf_demo.get_feature_names_out()

print(f"📊 Dimensi Matriks TF-IDF : {X_tfidf_sample.shape[0]:,} dokumen ulasan x {X_tfidf_sample.shape[1]:,} fitur kata")
print(f"🔤 Contoh 20 Fitur TF-IDF   : {list(feature_names[500:520])}")
"""))

    cells.append(nbf.v4.new_code_cell("""# TABEL CONTOH NILAI BOBOT TF-IDF PADA 1 ULASAN
sample_idx = 0
sample_row = X_tfidf_sample[sample_idx].toarray()[0]
top_indices = sample_row.argsort()[::-1][:10]

tabel_bobot_tfidf = pd.DataFrame({
    "Fitur Kata (Unigram / Bigram)": [feature_names[i] for i in top_indices if sample_row[i] > 0],
    "Bobot Skor TF-IDF": [f"{sample_row[i]:.4f}" for i in top_indices if sample_row[i] > 0]
})

print(f"Teks Ulasan Sampel: '{df_labeled['cleaned_text'].iloc[sample_idx]}'")
display(tabel_bobot_tfidf)
"""))

    cells.append(nbf.v4.new_markdown_cell("""## 5. Kesimpulan Tahap Data Preparation

1. **Hasil Preprocessing & Kualitas Teks**:
   - Seluruh teks ulasan mentah telah dibersihkan secara sistematis melalui 5 tahap preprocessing dan tersaring menjadi **17.923 ulasan bersih siap latih**.
2. **Representasi Fitur & Pelabelan**:
   - Teks berhasil direpresentasikan ke dalam vektor TF-IDF Unigram & Bigram dengan dimensi $17.923 \times 5.000$.
   - Dataset telah terpetakan ke dalam 3 kelas sentimen (Positif: 15.553, Negatif: 1.551, Netral: 819).
   - Dataset siap digunakan pada **WBS Tahap 4: Modeling**.
"""))

    nb['cells'] = cells
    return nb

if __name__ == "__main__":
    nb = create_wbs_nb2()
    out_file = NOTEBOOKS_DIR / "02_wbs_data_preparation.ipynb"
    with open(out_file, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"✅ Generated: {out_file}")
