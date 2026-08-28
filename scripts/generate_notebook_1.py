import json
import os
import sys
from pathlib import Path
import nbformat as nbf

NOTEBOOKS_DIR = Path("/Users/macbook/Garut-Tourism-Reccomendation/notebooks")
NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)

def create_notebook_1():
    nb = nbf.v4.new_notebook()
    cells = []
    
    # Markdown Header
    cells.append(nbf.v4.new_markdown_cell("""# 📊 Tahap 1: Eksplorasi Data Awal, Penggabungan, dan Pembersihan Duplikat
**Proyek:** Sistem Rekomendasi & Analisis Sentimen Pariwisata Kabupaten Garut  
**Tujuan:**
1. Memuat dataset ulasan mentah (*raw reviews*, *repaired reviews*, dan data destinasi).
2. Memeriksa nilai kosong (*missing values*) dan tipe data dalam bentuk tabel terstruktur.
3. Melakukan penggabungan dataset dengan prioritas data perbaikan (*repaired*).
4. Melakukan pembersihan data duplikat (*deduplication*) berdasarkan *composite key*.
5. Menjalankan filter kualitas data (filter teks kosong, nama destinasi kosong, dan validasi rentang rating 1-5).
6. Menyajikan **Tabel Rekapitulasi Lengkap (Sebelum vs Sesudah)** pada setiap langkah pembersihan.
"""))

    # Cell 1: Setup and Imports
    cells.append(nbf.v4.new_code_cell("""import matplotlib
matplotlib.use('module://matplotlib_inline.backend_inline')

import sys
import os
import warnings
from pathlib import Path

# Matikan UserWarning non-interactive canvas agar output cell bersih
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Pastikan Root Project ditambahkan ke sys.path
CURRENT_DIR = Path.cwd()
ROOT_DIR = CURRENT_DIR.parent if CURRENT_DIR.name == "notebooks" else CURRENT_DIR
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Pengaturan visualisasi & tabel Pandas
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica', 'sans-serif']
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)

from config import settings
print(f"✅ Root Project Directory : {ROOT_DIR}")
print(f"📁 Raw Data Directory     : {settings.RAW_DATA_DIR}")
"""))

    # Markdown Section 1
    cells.append(nbf.v4.new_markdown_cell("""## 1. Pemuatan Dataset Mentah (Raw Datasets) & Audit Awal"""))

    cells.append(nbf.v4.new_code_cell("""reviews_path = settings.RAW_DATA_DIR / "reviews.csv"
repaired_path = settings.RAW_DATA_DIR / "reviews_repaired.csv"
destinations_path = settings.RAW_DATA_DIR / "destinations.csv"

df_orig = pd.read_csv(reviews_path)
df_rep = pd.read_csv(repaired_path) if repaired_path.exists() else pd.DataFrame()
df_dest = pd.read_csv(destinations_path) if destinations_path.exists() else pd.DataFrame()

# Tabel Ringkasan Dataset Mentah
raw_summary_table = pd.DataFrame([
    {"Nama File": "reviews.csv", "Deskripsi": "Ulasan Asli Scraping Google Maps", "Jumlah Baris": f"{len(df_orig):,}", "Jumlah Kolom": df_orig.shape[1]},
    {"Nama File": "reviews_repaired.csv", "Deskripsi": "Ulasan Targeted Repair (Pelengkap)", "Jumlah Baris": f"{len(df_rep):,}", "Jumlah Kolom": df_rep.shape[1] if not df_rep.empty else 0},
    {"Nama File": "destinations.csv", "Deskripsi": "Metadata Destinasi Wisata Garut", "Jumlah Baris": f"{len(df_dest):,}", "Jumlah Kolom": df_dest.shape[1] if not df_dest.empty else 0}
])

display(raw_summary_table)
"""))

    cells.append(nbf.v4.new_code_cell("""# Tabel Pengecekan Missing Values pada Data Mentah Awal
missing_orig = pd.DataFrame({
    "Kolom": df_orig.columns,
    "Tipe Data": [str(t) for t in df_orig.dtypes],
    "Jumlah Baris Terisi (Non-Null)": [f"{v:,}" for v in df_orig.notnull().sum()],
    "Jumlah Nilai Kosong (Null/NaN)": [f"{v:,}" for v in df_orig.isnull().sum()],
    "Persentase Kosong (%)": [f"{(v/len(df_orig))*100:.2f}%" for v in df_orig.isnull().sum()]
})

display(missing_orig)
"""))

    # Markdown Section 2
    cells.append(nbf.v4.new_markdown_cell("""## 2. Penggabungan Data & Pembersihan Data Duplikat (Deduplication)

### Strategi Penggabungan & Deduplikasi:
1. Menandai sumber data: `repaired` dan `original`.
2. Menggabungkan dengan meletakkan data `repaired` di atas `original`.
3. Menetapkan **Composite Key** deduplikasi:
   - `destination_name`
   - `author`
   - `review_date`
   - `review_text`
4. Menjalankan `drop_duplicates(keep='first')` sehingga ulasan hasil perbaikan diutamakan.
"""))

    cells.append(nbf.v4.new_code_cell("""# 1. Tandai sumber dan gabungkan
df_orig["_source"] = "original"
if not df_rep.empty:
    df_rep["_source"] = "repaired"
    df_merged = pd.concat([df_rep, df_orig], ignore_index=True)
else:
    df_merged = df_orig.copy()

total_before_dedup = len(df_merged)

# 2. Identifikasi composite key
composite_cols = ["destination_name", "author", "review_date", "review_text"]
df_temp = df_merged.fillna({col: "" for col in composite_cols})

dup_mask = df_temp.duplicated(subset=composite_cols, keep="first")
duplicate_count = dup_mask.sum()

# 3. Hapus duplikat
df_dedup = df_merged[~dup_mask].copy().drop(columns=["_source"], errors="ignore")
total_after_dedup = len(df_dedup)

# Tabel Hasil Deduplikasi
dedup_table = pd.DataFrame([
    {"Kategori": "Total Baris Original", "Jumlah Baris": f"{len(df_orig):,}", "Keterangan": "Data ulasan mentah asli"},
    {"Kategori": "Total Baris Repaired", "Jumlah Baris": f"{len(df_rep):,}", "Keterangan": "Data ulasan hasil perbaikan targeted"},
    {"Kategori": "Total Gabungan Sebelum Deduplikasi", "Jumlah Baris": f"{total_before_dedup:,}", "Keterangan": "Concat repaired + original"},
    {"Kategori": "Duplikat Terdeteksi & Dihapus", "Jumlah Baris": f"{duplicate_count:,}", "Keterangan": f"Dihapus ({duplicate_count/total_before_dedup*100:.2f}% dari total gabungan)"},
    {"Kategori": "Total Baris Setelah Deduplikasi", "Jumlah Baris": f"{total_after_dedup:,}", "Keterangan": "Data unik pasca deduplikasi"}
])

display(dedup_table)
"""))

    cells.append(nbf.v4.new_code_cell("""# Tabel Contoh Sampel Ulasan yang Teridentifikasi Sebagai Duplikat
sample_dup_rows = df_temp[dup_mask][["destination_name", "author", "rating", "review_text"]].head(5)
sample_dup_rows.columns = ["Nama Destinasi", "Penulis Ulasan (Author)", "Rating", "Teks Ulasan Duplikat"]
display(sample_dup_rows)
"""))

    # Markdown Section 3
    cells.append(nbf.v4.new_markdown_cell("""## 3. Validasi Integritas & Filter Kualitas Data

Tahapan Filter:
1. **Filter Teks Kosong**: Menghapus baris jika `review_text` null atau hanya whitespace.
2. **Filter Destinasi Kosong**: Menghapus baris jika `destination_name` null / kosong.
3. **Filter Rating Valid**: Memastikan kolom `rating` berupa angka dan dalam rentang 1.0 s/d 5.0.
"""))

    cells.append(nbf.v4.new_code_cell("""# 1. Filter Teks Kosong
count_before_text = len(df_dedup)
df_valid = df_dedup.dropna(subset=["review_text"])
df_valid = df_valid[df_valid["review_text"].astype(str).str.strip() != ""]
count_after_text = len(df_valid)
removed_empty_text = count_before_text - count_after_text

# 2. Filter Destinasi Kosong
count_before_dest = count_after_text
df_valid = df_valid.dropna(subset=["destination_name"])
df_valid = df_valid[df_valid["destination_name"].astype(str).str.strip() != ""]
count_after_dest = len(df_valid)
removed_empty_dest = count_before_dest - count_after_dest

# 3. Filter Validitas Rating (1.0 - 5.0)
def is_valid_rating(r):
    try:
        val = float(r)
        return 1.0 <= val <= 5.0
    except (ValueError, TypeError):
        return False

count_before_rating = count_after_dest
valid_rating_mask = df_valid["rating"].apply(is_valid_rating)
df_valid = df_valid[valid_rating_mask].copy()
df_valid["rating"] = df_valid["rating"].astype(float)
count_after_rating = len(df_valid)
removed_invalid_rating = count_before_rating - count_after_rating

# TABEL REKAPITULASI SEBELUM DAN SESUDAH PEMBERSIHAN
rekap_lengkap = pd.DataFrame([
    {"Tahap Pembersihan": "1. Data Mentah Gabungan", "Jumlah Baris": f"{total_before_dedup:,}", "Baris Berkurang": "0", "Persentase Sisa": "100.0%"},
    {"Tahap Pembersihan": "2. Pembersihan Duplikat (Composite Key)", "Jumlah Baris": f"{total_after_dedup:,}", "Baris Berkurang": f"-{duplicate_count:,}", "Persentase Sisa": f"{(total_after_dedup/total_before_dedup)*100:.2f}%"},
    {"Tahap Pembersihan": "3. Filter Ulasan Teks Kosong / Spasi", "Jumlah Baris": f"{count_after_text:,}", "Baris Berkurang": f"-{removed_empty_text:,}", "Persentase Sisa": f"{(count_after_text/total_before_dedup)*100:.2f}%"},
    {"Tahap Pembersihan": "4. Filter Nama Destinasi Kosong", "Jumlah Baris": f"{count_after_dest:,}", "Baris Berkurang": f"-{removed_empty_dest:,}", "Persentase Sisa": f"{(count_after_dest/total_before_dedup)*100:.2f}%"},
    {"Tahap Pembersihan": "5. Filter Validitas Rating (1.0 s/d 5.0)", "Jumlah Baris": f"{count_after_rating:,}", "Baris Berkurang": f"-{removed_invalid_rating:,}", "Persentase Sisa": f"{(count_after_rating/total_before_dedup)*100:.2f}%"}
])

display(rekap_lengkap)
"""))

    # Markdown Section 4
    cells.append(nbf.v4.new_markdown_cell("""## 4. Eksplorasi Awal: Top Destinasi & Distribusi Rating Pasca Pembersihan"""))

    cells.append(nbf.v4.new_code_cell("""# Tabel Top 15 Destinasi dengan Ulasan Terbanyak dan Rata-rata Rating
top15_summary = df_valid.groupby("destination_name").agg(
    Jumlah_Ulasan=("rating", "count"),
    Rating_Rata_Rata=("rating", "mean")
).sort_values("Jumlah_Ulasan", ascending=False).head(15).reset_index()

top15_summary.columns = ["Nama Destinasi", "Jumlah Ulasan Valid", "Rating Rata-rata (1-5)"]
top15_summary["Rating Rata-rata (1-5)"] = top15_summary["Rating Rata-rata (1-5)"].round(2)
display(top15_summary)
"""))

    cells.append(nbf.v4.new_code_cell("""# Tabel Distribusi Rating Mentah (1 - 5 Bintang)
rating_dist = df_valid["rating"].value_counts().sort_index().reset_index()
rating_dist.columns = ["Bintang Rating", "Jumlah Ulasan"]
rating_dist["Persentase"] = [(v / len(df_valid)) * 100 for v in rating_dist["Jumlah Ulasan"]]
rating_dist["Persentase"] = rating_dist["Persentase"].map("{:.2f}%".format)
display(rating_dist)
"""))

    cells.append(nbf.v4.new_code_cell("""# Simpan dataset hasil validasi ke folder interim
interim_output_path = settings.INTERIM_DATA_DIR / "reviews_validated.csv"
df_valid.to_csv(interim_output_path, index=False)
print(f"✅ Data hasil validasi berhasil disimpan ke: {interim_output_path}")
print(f"Total baris valid yang siap dipreprocessing: {len(df_valid):,} baris")
"""))

    nb.cells = cells
    return nb

if __name__ == "__main__":
    print("Notebook 1 generator ready.")
