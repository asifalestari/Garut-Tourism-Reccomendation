import os
import sys
import time
from pathlib import Path
import nbformat as nbf
from nbclient import NotebookClient

# Ensure root directory in python path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from scripts.generate_notebook_1 import create_notebook_1
from scripts.generate_notebook_2 import create_notebook_2
from scripts.generate_notebook_3 import create_notebook_3
from scripts.generate_notebook_master import create_notebook_master

NOTEBOOKS_DIR = ROOT_DIR / "notebooks"
NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)

def execute_and_save(nb, output_filename):
    out_path = NOTEBOOKS_DIR / output_filename
    print(f"\n==========================================")
    print(f"🚀 Memproses notebook: {output_filename}...")
    print(f"==========================================")
    
    start_time = time.time()
    # Save initial notebook
    with open(out_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
        
    print(f"Mengeksekusi semua cell di {output_filename} dengan ipykernel...")
    client = NotebookClient(nb, timeout=600, kernel_name="python3", resources={"metadata": {"path": str(ROOT_DIR)}})
    client.execute()
    
    # Save executed notebook with outputs
    with open(out_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
        
    elapsed = time.time() - start_time
    print(f"✅ Selesai: {out_path} ({elapsed:.1f} detik)")

def main():
    print("Memulai pembuatan dan eksekusi seluruh Jupyter Notebooks...")
    
    # 1. Notebook 1: Data Ingestion & Deduplication
    nb1 = create_notebook_1()
    execute_and_save(nb1, "01_eksplorasi_dan_pembersihan_data.ipynb")
    
    # 2. Notebook 2: Text Preprocessing Step-by-Step
    nb2 = create_notebook_2()
    execute_and_save(nb2, "02_tahapan_text_preprocessing.ipynb")
    
    # 3. Notebook 3: Advanced Analysis & Visualizations
    nb3 = create_notebook_3()
    execute_and_save(nb3, "03_analisis_lanjutan_dan_visualisasi.ipynb")
    
    # 4. Notebook Master
    nb_master = create_notebook_master()
    execute_and_save(nb_master, "00_master_eda_dan_preprocessing.ipynb")
    
    print("\n🎉 SELURUH NOTEBOOKS BERHASIL DIBUAT DAN DIEKSEKUSI!")
    print(f"Lokasi: {NOTEBOOKS_DIR}")

if __name__ == "__main__":
    main()
