import os
import sys
from pathlib import Path
import nbformat as nbf

NOTEBOOKS_DIR = Path("/Users/macbook/Garut-Tourism-Reccomendation/notebooks")
NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)

def create_wbs_nb1():
    nb = nbf.v4.new_notebook()
    cells = []

    # Title & Markdown Intro
    cells.append(nbf.v4.new_markdown_cell("""# 🌐 WBS Tahap 2: Data Understanding
**Proyek:** Analisis Sentimen Ulasan Destinasi Wisata untuk Pengambilan Kebijakan Promosi Wisata pada Dinas Pariwisata dan Kebudayaan Kabupaten Garut  
**Tahapan WBS Terkait:**
1. **Menentukan sumber data (Google Maps)**
2. **Menentukan dan mengumpulkan destinasi wisata (382/377 destinasi awal)**
3. **Web scraping ulasan menggunakan Playwright**
4. **Mengidentifikasi jumlah dan atribut data**
5. **Eksplorasi karakteristik data ulasan**
6. **Pemeriksaan kualitas dan kelengkapan data (Missing values & Data integrity)**

---
### Output Tahap Ini:
Dataset ulasan wisatawan yang terkumpul dan dipahami karakteristiknya secara empiris sebelum memasuki tahapan data preparation.
"""))

    # Cell 1: Setup and Imports
    cells.append(nbf.v4.new_code_cell("""import matplotlib
matplotlib.use('module://matplotlib_inline.backend_inline')

import sys
import os
import json
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

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica', 'sans-serif']
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)

from config import settings

print(f"✅ Root Project Directory: {ROOT_DIR}")
print("Pustaka eksplorasi data & visualisasi berhasil dimuat.")
"""))

    # Section 1: Sumber Data & Metadata Destinasi Wisata
    cells.append(nbf.v4.new_markdown_cell("""## 1. Sumber Data & Metadata Destinasi Wisata Kabupaten Garut

Data destinasi wisata dikumpulkan melalui penelusuran lokasi pariwisata resmi di wilayah Kabupaten Garut pada platform **Google Maps** menggunakan *automated scraping* berbasis framework **Playwright**.
"""))

    cells.append(nbf.v4.new_code_cell("""# 1. Pemuatan Metadata Destinasi Wisata
dest_path = settings.RAW_DATA_DIR / "destinations.csv"
urls_path = settings.RAW_DATA_DIR / "destination_urls.json"

df_dest = pd.read_csv(dest_path)
total_dest_rows = len(df_dest)
unique_dest_names = df_dest["name"].nunique()

print(f"📊 Total Rekaman Metadata Destinasi : {total_dest_rows} baris")
print(f"📍 Jumlah Nama Destinasi Unik       : {unique_dest_names} destinasi")

# Contoh tampilan metadata destinasi
display(df_dest.head(5))
"""))

    cells.append(nbf.v4.new_code_cell("""# TABEL RINGKASAN ATRIBUT METADATA DESTINASI
dest_meta_summary = []
for col in df_dest.columns:
    dest_meta_summary.append({
        "Nama Kolom": col,
        "Tipe Data": str(df_dest[col].dtype),
        "Jumlah Terisi": f"{df_dest[col].notnull().sum():,} baris",
        "Jumlah Missing (NaN)": f"{df_dest[col].isnull().sum():,} baris",
        "Persentase Missing": f"{(df_dest[col].isnull().sum() / len(df_dest))*100:.2f}%",
        "Contoh Nilai": str(df_dest[col].dropna().iloc[0]) if df_dest[col].notnull().sum() > 0 else "-"
    })

df_dest_meta_table = pd.DataFrame(dest_meta_summary)
display(df_dest_meta_table)
"""))

    # Section 2: Pemuatan Dataset Ulasan Mentah
    cells.append(nbf.v4.new_markdown_cell("""## 2. Pemuatan & Identifikasi Atribut Dataset Ulasan Mentah (*Raw Reviews*)

Dataset ulasan mentah terdiri dari dua berkas:
1. `reviews.csv` (Dataset ulasan primer hasil scraping Google Maps).
2. `reviews_repaired.csv` (Dataset ulasan hasil scraping perbaikan untuk memastikan ulasan terakomodasi).
"""))

    cells.append(nbf.v4.new_code_cell("""# Pemuatan file ulasan mentah
reviews_path = settings.RAW_DATA_DIR / "reviews.csv"
repaired_path = settings.RAW_DATA_DIR / "reviews_repaired.csv"

df_orig = pd.read_csv(reviews_path)
df_rep = pd.read_csv(repaired_path) if repaired_path.exists() else pd.DataFrame()

print(f"📄 Ulasan File Original (reviews.csv)         : {len(df_orig):,} baris")
print(f"📄 Ulasan File Perbaikan (reviews_repaired.csv): {len(df_rep):,} baris")
print(f"📦 Total Baris Mentah Sebelum Penggabungan    : {len(df_orig) + len(df_rep):,} baris")

# Tampilan contoh 5 data ulasan teratas
display(df_orig.head(5))
"""))

    cells.append(nbf.v4.new_code_cell("""# TABEL IDENTIFIKASI ATRIBUT DATASET ULASAN MENTAH
reviews_attr_summary = []
for col in df_orig.columns:
    reviews_attr_summary.append({
        "Nama Atribut": col,
        "Tipe Data": str(df_orig[col].dtype),
        "Jumlah Terisi (Non-Null)": f"{df_orig[col].notnull().sum():,} baris",
        "Jumlah Missing Values": f"{df_orig[col].isnull().sum():,} baris",
        "Persentase Missing": f"{(df_orig[col].isnull().sum() / len(df_orig))*100:.2f}%",
        "Keterangan Fungsional": {
            "destination_name": "Nama destinasi objek wisata Garut yang diulas",
            "author": "Nama / identitas akun pengguna Google Maps",
            "rating": "Nilai rating bintang ulasan dari pengguna (1.0 s/d 5.0)",
            "review_date": "Keterangan waktu / tanggal publikasi ulasan",
            "review_text": "Teks narasi opini ulasan pengunjung",
            "scraped_at": "Stempel waktu proses web scraping Playwright",
            "review_id": "Identifikasi unik ulasan dari Google Maps",
            "has_text": "Penanda boolean keberadaan teks narasi ulasan"
        }.get(col, "Atribut tambahan scraping")
    })

df_reviews_attr_table = pd.DataFrame(reviews_attr_summary)
display(df_reviews_attr_table)
"""))

    # Section 3: Eksplorasi Karakteristik Data & Pemeriksaan Kualitas
    cells.append(nbf.v4.new_markdown_cell("""## 3. Eksplorasi Karakteristik Data & Pemeriksaan Kualitas

Pemeriksaan kelengkapan dan karakteristik data:
1. **Analisis Missing Values pada Teks (*Rating-Only Reviews*)**: Mengidentifikasi proporsi ulasan yang hanya memberikan bintang tanpa komentar narasi tertulis.
2. **Distribusi Bintang Rating Mentah (1.0 - 5.0)**: Memeriksa sebaran rating awal wisatawan.
3. **Analisis Panjang Karakter Ulasan**: Menghitung panjang karakter dan jumlah kata ulasan mentah.
"""))

    cells.append(nbf.v4.new_code_cell("""# 1. Analisis Ulasan Berteks vs Rating-Only (Teks Kosong)
total_raw_orig = len(df_orig)
null_text_count = df_orig["review_text"].isnull().sum()
empty_str_count = (df_orig["review_text"].astype(str).str.strip() == "").sum() - null_text_count
valid_text_count = total_raw_orig - null_text_count - empty_str_count

tabel_kualitas_teks = pd.DataFrame([
    {
        "Kategori Ulasan": "Ulasan Memiliki Teks Narasi (Valid Text Reviews)",
        "Jumlah Baris": f"{valid_text_count:,}",
        "Persentase": f"{(valid_text_count / total_raw_orig)*100:.2f}%",
        "Dampak Metodologis": "Dapat diekstrak fiturnya dengan TF-IDF untuk pembelajaran SVM"
    },
    {
        "Kategori Ulasan": "Ulasan Tanpa Teks (Rating-Only Reviews / NaN)",
        "Jumlah Baris": f"{null_text_count + empty_str_count:,}",
        "Persentase": f"{((null_text_count + empty_str_count) / total_raw_orig)*100:.2f}%",
        "Dampak Metodologis": "Secara metodologis dieliminasi karena tidak memiliki fitur linguistik"
    }
])

display(tabel_kualitas_teks)
"""))

    cells.append(nbf.v4.new_code_cell("""# 2. TABEL DISTRIBUSI BINTANG RATING MENTAH (1 - 5 Bintang)
raw_ratings = df_orig["rating"].dropna()
rating_counts = raw_ratings.value_counts().sort_index()

tabel_dist_rating_mentah = pd.DataFrame({
    "Bintang Rating": [f"★ {int(r)}" if float(r).is_integer() else f"★ {r}" for r in rating_counts.index],
    "Jumlah Ulasan": [f"{v:,}" for v in rating_counts.values],
    "Persentase (%)": [f"{(v / len(raw_ratings))*100:.2f}%" for v in rating_counts.values]
})

display(tabel_dist_rating_mentah)
"""))

    cells.append(nbf.v4.new_code_cell("""# Visualisasi Karakteristik Data Mentah
fig, axes = plt.subplots(1, 2, figsize=(14, 4.5), dpi=100)

# Grafik 1: Distribusi Rating Mentah
r_idx = [int(r) if float(r).is_integer() else r for r in rating_counts.index]
sns.barplot(x=r_idx, y=rating_counts.values, hue=r_idx, legend=False, ax=axes[0], palette="crest")
axes[0].set_title("Distribusi Rating Mentah (reviews.csv)", fontsize=12, fontweight="bold")
axes[0].set_xlabel("Nilai Rating Bintang", fontsize=10)
axes[0].set_ylabel("Jumlah Ulasan", fontsize=10)
for i, v in enumerate(rating_counts.values):
    axes[0].text(i, v + 250, f"{v:,}", ha="center", fontsize=9, fontweight="bold")

# Grafik 2: Proporsi Ulasan Berteks vs Rating-Only
pie_data = [valid_text_count, null_text_count + empty_str_count]
pie_labels = ["Ulasan Berteks (Valid)", "Rating-Only (Tanpa Teks)"]
axes[1].pie(
    pie_data,
    labels=pie_labels,
    autopct="%1.1f%%",
    startangle=140,
    colors=["#2b8a3e", "#adb5bd"],
    wedgeprops=dict(width=0.6, edgecolor='w', linewidth=2),
    textprops=dict(fontsize=10, fontweight="bold")
)
axes[1].set_title("Proporsi Ulasan Berteks vs Rating-Only", fontsize=12, fontweight="bold")

plt.tight_layout()
plt.show()
"""))

    cells.append(nbf.v4.new_code_cell("""# 3. Analisis Panjang Teks Ulasan Mentah
df_text_only = df_orig.dropna(subset=["review_text"]).copy()
df_text_only["char_length"] = df_text_only["review_text"].astype(str).apply(len)
df_text_only["word_count"] = df_text_only["review_text"].astype(str).apply(lambda x: len(x.split()))

tabel_statistik_panjang = pd.DataFrame({
    "Ukuran Statistik": ["Rata-rata (Mean)", "Median", "Nilai Minimum", "Nilai Maksimum", "Standar Deviasi"],
    "Panjang Karakter": [
        f"{df_text_only['char_length'].mean():.2f} karakter",
        f"{df_text_only['char_length'].median():.0f} karakter",
        f"{df_text_only['char_length'].min():,} karakter",
        f"{df_text_only['char_length'].max():,} karakter",
        f"{df_text_only['char_length'].std():.2f}"
    ],
    "Jumlah Kata (Words)": [
        f"{df_text_only['word_count'].mean():.2f} kata",
        f"{df_text_only['word_count'].median():.0f} kata",
        f"{df_text_only['word_count'].min():,} kata",
        f"{df_text_only['word_count'].max():,} kata",
        f"{df_text_only['word_count'].std():.2f}"
    ]
})

display(tabel_statistik_panjang)
"""))

    cells.append(nbf.v4.new_markdown_cell("""## 4. Kesimpulan Tahap Data Understanding

1. **Sumber Data & Cakupan Destinasi**:
   - Total 382/377 destinasi wisata resmi teridentifikasi di Kabupaten Garut dengan koordinat geografis di Google Maps.
   - Total 36.574 rekaman ulasan mentah terkumpul melalui scraping Playwright.
2. **Karakteristik & Integritas Kualitas Data**:
   - Terdapat **14.538 ulasan bertipe *rating-only*** (tanpa narasi teks) yang secara metodologis dikeluarkan sebelum ekstraksi fitur TF-IDF.
   - Terdapat dominasi rating bintang tinggi (Rating 5.0 mencapai >70% ulasan), yang menjadi dasar pertimbangan penanganan *imbalanced class* pada tahap modeling.
   - Dataset siap dilanjutkan ke **WBS Tahap 3: Data Preparation**.
"""))

    nb['cells'] = cells
    return nb

if __name__ == "__main__":
    nb = create_wbs_nb1()
    out_file = NOTEBOOKS_DIR / "01_wbs_data_understanding.ipynb"
    with open(out_file, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"✅ Generated: {out_file}")
