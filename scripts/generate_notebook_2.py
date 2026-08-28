import json
import os
import sys
from pathlib import Path
import nbformat as nbf

NOTEBOOKS_DIR = Path("/Users/macbook/Garut-Tourism-Reccomendation/notebooks")
NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)

def create_notebook_2():
    nb = nbf.v4.new_notebook()
    cells = []

    # Title & Markdown Intro
    cells.append(nbf.v4.new_markdown_cell("""# 🔍 Tahap 2: Rangkaian Preprocessing Teks Langkah demi Langkah
**Proyek:** Sistem Rekomendasi & Analisis Sentimen Pariwisata Kabupaten Garut  
**Tujuan:**
Menampilkan secara transparan hasil dari setiap tahapan pemrosesan teks (*Text Preprocessing Pipeline*) dalam bentuk **Tabel Perbandingan Sebelum vs Sesudah (Before & After)**:
1. **Tahap 1: Pembersihan Teks Mentah (*Text Cleaning*)**: Menghapus URL, tag HTML, emoji, simbol, dan tanda baca.
2. **Tahap 2: Penyeragaman Huruf (*Case Folding*)**: Mengubah teks menjadi huruf kecil (*lowercase*).
3. **Tahap 3: Pemotongan Kata (*Tokenization*)**: Memecah kalimat menjadi kumpulan token kata.
4. **Tahap 4: Pembersihan Kata Tidak Bermakna (*Stopword Removal*)**: Menghapus stopword bahasa Indonesia (NLTK + custom domain Garut).
5. **Tahap 5: Pengembalian ke Kata Dasar (*Stemming*)**: Menggunakan PySastrawi untuk mereduksi kata berimbuhan.
6. **Tahap 6: Filter Ulasan Kosong (*Empty Review Filter*)**: Menyaring ulasan yang menjadi kosong setelah stopwords dibersihkan.
"""))

    # Imports & Setup
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
from collections import Counter

pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)

from config import settings
from config.constants import CUSTOM_INDONESIAN_STOPWORDS
from preprocessing.cleaning import clean_text
from preprocessing.case_folding import case_folding
from preprocessing.tokenization import tokenize
from preprocessing.stopword_removal import remove_stopwords, ALL_STOPWORDS
from preprocessing.stemming import stem_tokens

print(f"✅ Root Directory: {ROOT_DIR}")
print("Module preprocessing berhasil dimuat.")
"""))

    # Load Validated Data
    cells.append(nbf.v4.new_markdown_cell("""## 1. Pemuatan Data Hasil Validasi Tahap 1"""))

    cells.append(nbf.v4.new_code_cell("""interim_path = settings.INTERIM_DATA_DIR / "reviews_validated.csv"
if not interim_path.exists():
    raise FileNotFoundError("reviews_validated.csv belum ada. Jalankan notebook 01 terlebih dahulu.")

df = pd.read_csv(interim_path)

# Tabel Ringkasan Input Preprocessing
input_summary = pd.DataFrame([
    {"Deskripsi": "Total Baris Ulasan Siap Preprocessing", "Nilai": f"{len(df):,} baris"},
    {"Deskripsi": "Jumlah Destinasi Wisata", "Nilai": f"{df['destination_name'].nunique():,} destinasi"},
    {"Deskripsi": "Rata-rata Rating Awal", "Nilai": f"{df['rating'].mean():.2f} / 5.0"}
])
display(input_summary)
"""))

    # Section: Step 1 Text Cleaning
    cells.append(nbf.v4.new_markdown_cell("""## 2. Tahap 1: Text Cleaning (Pembersihan Teks Mentah)
Fungsi `clean_text` membersihkan:
- URL (`http...`, `www...`)
- Tag HTML (`<...>` )
- Emoji dan karakter Non-ASCII
- Karakter khusus, simbol, dan tanda baca (diubah menjadi spasi)
- Spasi ganda, newline, dan tab (dinormalkan menjadi spasi tunggal)
"""))

    cells.append(nbf.v4.new_code_cell("""df["step1_cleaned"] = df["review_text"].apply(clean_text)

# Hitung statistik panjang karakter
df["len_raw_char"] = df["review_text"].str.len()
df["len_clean_char"] = df["step1_cleaned"].str.len()

# TABEL STATISTIK TEXT CLEANING
cleaning_stat_table = pd.DataFrame([
    {"Metrik Karakter": "Rata-rata Panjang Karakter Sebelum Cleaning (Raw)", "Nilai": f"{df['len_raw_char'].mean():.1f} karakter"},
    {"Metrik Karakter": "Rata-rata Panjang Karakter Sesudah Cleaning", "Nilai": f"{df['len_clean_char'].mean():.1f} karakter"},
    {"Metrik Karakter": "Rata-rata Karakter Simbol/Emoji/Punctuation yang Terhapus", "Nilai": f"{df['len_raw_char'].mean() - df['len_clean_char'].mean():.1f} karakter"},
    {"Metrik Karakter": "Maksimum Panjang Karakter", "Nilai": f"{df['len_raw_char'].max():,} $\\rightarrow$ {df['len_clean_char'].max():,} karakter"}
])
display(cleaning_stat_table)
"""))

    cells.append(nbf.v4.new_code_cell(r"""# TABEL CONTOH PERBANDINGAN SEBELUM VS SESUDAH TEXT CLEANING
# Pilih 5 sampel ulasan riil yang mengandung tanda baca, emoji, link, atau karakter khusus
samples_symbols = df[df["review_text"].str.contains(r"[^\w\s]|http", regex=True, na=False)].sample(5, random_state=42)

tabel_step1 = pd.DataFrame({
    "No": range(1, 6),
    "Teks Mentah Awal (Sebelum Cleaning)": samples_symbols["review_text"].values,
    "Hasil Teks (Sesudah Cleaning)": samples_symbols["step1_cleaned"].values
})
display(tabel_step1)
"""))

    # Section: Step 2 Case Folding
    cells.append(nbf.v4.new_markdown_cell("""## 3. Tahap 2: Case Folding (Penyeragaman Huruf Kecil)
Mengubah semua huruf kapital menjadi huruf kecil (*lowercase*) agar representasi kata seragam (*Garut*, *GARUT*, *garut* $\rightarrow$ *garut*).
"""))

    cells.append(nbf.v4.new_code_cell("""df["step2_folded"] = df["step1_cleaned"].apply(case_folding)

# TABEL CONTOH SEBELUM VS SESUDAH CASE FOLDING
samples_cf = df[df["step1_cleaned"].str.contains(r"[A-Z]", regex=True, na=False)].sample(5, random_state=42)

tabel_step2 = pd.DataFrame({
    "No": range(1, 6),
    "Sebelum Case Folding (Ada Huruf Kapital)": samples_cf["step1_cleaned"].values,
    "Sesudah Case Folding (Seluruhnya Huruf Kecil)": samples_cf["step2_folded"].values
})
display(tabel_step2)
"""))

    # Section: Step 3 Tokenization
    cells.append(nbf.v4.new_markdown_cell("""## 4. Tahap 3: Tokenization (Pemotongan Kata)
Memecah kalimat teks menjadi daftar token kata (*list of tokens*) berdasarkan spasi.
"""))

    cells.append(nbf.v4.new_code_cell("""df["step3_tokens"] = df["step2_folded"].apply(tokenize)
df["num_tokens_raw"] = df["step3_tokens"].apply(len)

# TABEL STATISTIK TOKENISASI
token_stat_table = pd.DataFrame([
    {"Metrik": "Total Keseluruhan Kata (Tokens) di Dataset", "Nilai": f"{df['num_tokens_raw'].sum():,} kata"},
    {"Metrik": "Rata-rata Jumlah Kata per Ulasan", "Nilai": f"{df['num_tokens_raw'].mean():.2f} kata"},
    {"Metrik": "Jumlah Kata Terbanyak dalam 1 Ulasan", "Nilai": f"{df['num_tokens_raw'].max():,} kata"},
    {"Metrik": "Jumlah Kata Tersedikit dalam 1 Ulasan", "Nilai": f"{df['num_tokens_raw'].min():,} kata"}
])
display(token_stat_table)
"""))

    cells.append(nbf.v4.new_code_cell("""# TABEL CONTOH SEBELUM VS SESUDAH TOKENIZATION
samples_tok = df.sample(5, random_state=42)

tabel_step3 = pd.DataFrame({
    "No": range(1, 6),
    "Kalimat Teks Input (String)": samples_tok["step2_folded"].values,
    "Hasil Tokenisasi (List of Words)": [str(toks) for toks in samples_tok["step3_tokens"].values],
    "Jumlah Token": samples_tok["num_tokens_raw"].values
})
display(tabel_step3)
"""))

    # Section: Step 4 Stopword Removal
    cells.append(nbf.v4.new_markdown_cell("""## 5. Tahap 4: Stopword Removal (Pembersihan Kata Tidak Bermakna)

Stopwords yang dieliminasi:
1. **NLTK Indonesian Stopwords**: Kata hubung (*dan, yang, di, ke, dari, adalah*).
2. **Custom Indonesian Stopwords**: Kata slang & pengisi ulasan (*yg, gak, tempat, wisata, garut, lokasi, nya, banget, deh*).
"""))

    cells.append(nbf.v4.new_code_cell("""df["step4_no_stopwords"] = df["step3_tokens"].apply(remove_stopwords)
df["num_tokens_clean"] = df["step4_no_stopwords"].apply(len)

total_tokens_before = df["num_tokens_raw"].sum()
total_tokens_after = df["num_tokens_clean"].sum()
removed_tokens = total_tokens_before - total_tokens_after

# TABEL REKAPITULASI STOPWORD REMOVAL
sw_summary_table = pd.DataFrame([
    {"Kategori": "Total Kata Sebelum Stopword Removal", "Jumlah": f"{total_tokens_before:,} kata", "Persentase": "100.0%"},
    {"Kategori": "Total Kata Stopword yang Dihapus", "Jumlah": f"{removed_tokens:,} kata", "Persentase": f"{(removed_tokens/total_tokens_before)*100:.2f}%"},
    {"Kategori": "Total Kata Bermakna yang Dipertahankan", "Jumlah": f"{total_tokens_after:,} kata", "Persentase": f"{(total_tokens_after/total_tokens_before)*100:.2f}%"}
])
display(sw_summary_table)
"""))

    cells.append(nbf.v4.new_code_cell("""# TABEL TOP 20 STOPWORDS YANG PALING BANYAK DIHAPUS DARI ULASAN
all_raw_tokens = [tok for sublist in df["step3_tokens"] for tok in sublist]
removed_stopword_list = [tok for tok in all_raw_tokens if tok in ALL_STOPWORDS]
stopword_counter = Counter(removed_stopword_list).most_common(20)

df_top_sw = pd.DataFrame(stopword_counter, columns=["Kata Stopword", "Frekuensi Terhapus"])
df_top_sw["Persentase dari Seluruh Stopword"] = [(v / len(removed_stopword_list)) * 100 for v in df_top_sw["Frekuensi Terhapus"]]
df_top_sw["Persentase dari Seluruh Stopword"] = df_top_sw["Persentase dari Seluruh Stopword"].map("{:.2f}%".format)
df_top_sw.index = range(1, 21)
display(df_top_sw)
"""))

    cells.append(nbf.v4.new_code_cell("""# TABEL CONTOH SEBELUM VS SESUDAH STOPWORD REMOVAL
samples_sw = df.sample(5, random_state=42)

tabel_step4 = pd.DataFrame({
    "No": range(1, 6),
    "Token Sebelum Stopwords": [str(t) for t in samples_sw["step3_tokens"].values],
    "Token Sesudah Stopwords Dihapus": [str(t) for t in samples_sw["step4_no_stopwords"].values],
    "Jumlah Kata Dihapus": samples_sw["num_tokens_raw"].values - samples_sw["num_tokens_clean"].values
})
display(tabel_step4)
"""))

    # Section: Step 5 Stemming
    cells.append(nbf.v4.new_markdown_cell("""## 6. Tahap 5: Stemming (Pengubahan ke Kata Dasar)
Menggunakan **PySastrawi** untuk mereduksi kata berimbuhan bahasa Indonesia menjadi bentuk dasarnya (*root words*).
"""))

    cells.append(nbf.v4.new_code_cell("""# Eksekusi Stemming dengan progress bar
tqdm.pandas(desc="Menjalankan Stemming")
df["step5_stemmed"] = df["step4_no_stopwords"].progress_apply(stem_tokens)

# Gabungkan token menjadi kalimat string bersih akhir
df["cleaned_text"] = df["step5_stemmed"].apply(lambda tokens: " ".join(tokens))
"""))

    cells.append(nbf.v4.new_code_cell("""# TABEL CONTOH PEMETAAN KATA BERIMBUHAN -> KATA DASAR (STEMMED)
sample_pairs = []
seen_pairs = set()
for orig_toks, stem_toks in zip(df["step4_no_stopwords"], df["step5_stemmed"]):
    for o, s in zip(orig_toks, stem_toks):
        if o != s and (o, s) not in seen_pairs:
            seen_pairs.add((o, s))
            sample_pairs.append({"Kata Berimbuhan (Sebelum)": o, "Kata Dasar / Root Word (Sesudah)": s})
            if len(sample_pairs) >= 15:
                break
    if len(sample_pairs) >= 15:
        break

df_stem_samples = pd.DataFrame(sample_pairs)
df_stem_samples.index = range(1, len(df_stem_samples) + 1)
display(df_stem_samples)
"""))

    # Section: Step 6 Filter Empty Reviews
    cells.append(nbf.v4.new_markdown_cell("""## 7. Tahap 6: Evaluasi Akhir & Filter Ulasan Kosong
Setelah pembersihan stopwords dan stemming, beberapa ulasan yang aslinya hanya berisi stopwords menghasilkan string kosong (`""`). Baris kosong ini disaring agar model ML menerima data yang valid.
"""))

    cells.append(nbf.v4.new_code_cell("""count_before_empty_filter = len(df)
df_final = df[df["cleaned_text"].str.strip() != ""].copy()
count_after_empty_filter = len(df_final)
empty_filtered_count = count_before_empty_filter - count_after_empty_filter

# TABEL CORONG DATA PREPROCESSING (DATA FUNNEL REKAP)
rekap_preprocessing = pd.DataFrame([
    {"Tahap Preprocessing": "1. Ulasan Masuk Tahap Preprocessing", "Jumlah Baris": f"{count_before_empty_filter:,}", "Ulasan Terfilter": "0", "Persentase": "100.0%"},
    {"Tahap Preprocessing": "2. Ulasan Menjadi Kosong Pasca Stopword Removal", "Jumlah Baris": f"-{empty_filtered_count:,}", "Ulasan Terfilter": f"-{empty_filtered_count:,}", "Persentase": f"{(empty_filtered_count/count_before_empty_filter)*100:.2f}%"},
    {"Tahap Preprocessing": "3. Ulasan BERSIH AKHIR (Siap Latih Model ML)", "Jumlah Baris": f"{count_after_empty_filter:,}", "Ulasan Terfilter": "0", "Persentase": f"{(count_after_empty_filter/count_before_empty_filter)*100:.2f}%"}
])
display(rekap_preprocessing)
"""))

    cells.append(nbf.v4.new_code_cell("""# TABEL PERBANDINGAN KOMPREHENSIF DARI RAW SAMPAI FINAL
sample_all_steps = df_final.sample(5, random_state=42)

tabel_komprehensif = pd.DataFrame({
    "No": range(1, 6),
    "1. Raw Review (Mentah)": sample_all_steps["review_text"].values,
    "2. Cleaned (Bebas Simbol/URL/Emoji)": sample_all_steps["step1_cleaned"].values,
    "3. Case Folded (Huruf Kecil)": sample_all_steps["step2_folded"].values,
    "4. Cleaned & Stemmed (Final Siap ML)": sample_all_steps["cleaned_text"].values
})
display(tabel_komprehensif)
"""))

    cells.append(nbf.v4.new_code_cell("""# Simpan dataset hasil preprocessing final
output_final_path = settings.FINAL_DATA_DIR / "processed_reviews.csv"
output_cols = ["destination_name", "author", "rating", "review_text", "cleaned_text"]
df_final[output_cols].to_csv(output_final_path, index=False)
print(f"✅ Data ulasan bersih final berhasil disimpan ke: {output_final_path}")
print(f"Total baris ulasan bersih final: {len(df_final):,} baris")
"""))

    nb.cells = cells
    return nb

if __name__ == "__main__":
    print("Notebook 2 generator ready.")
