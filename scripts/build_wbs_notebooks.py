import os
import sys
import time
from pathlib import Path
import nbformat as nbf
from nbclient import NotebookClient

# Ensure root directory in python path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from scripts.generate_wbs_nb1_data_understanding import create_wbs_nb1
from scripts.generate_wbs_nb2_data_preparation import create_wbs_nb2
from scripts.generate_wbs_nb3_modeling import create_wbs_nb3
from scripts.generate_wbs_nb4_evaluation import create_wbs_nb4
from scripts.generate_wbs_nb5_deployment_policy import create_wbs_nb5
from scripts.generate_wbs_nb0_master import create_wbs_nb0

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
    print("Memulai pembuatan dan eksekusi seluruh WBS Jupyter Notebooks...")
    
    # 1. WBS 2: Data Understanding
    nb1 = create_wbs_nb1()
    execute_and_save(nb1, "01_wbs_data_understanding.ipynb")
    
    # 2. WBS 3: Data Preparation
    nb2 = create_wbs_nb2()
    execute_and_save(nb2, "02_wbs_data_preparation.ipynb")
    
    # 3. WBS 4: Modeling
    nb3 = create_wbs_nb3()
    execute_and_save(nb3, "03_wbs_modeling.ipynb")
    
    # 4. WBS 5: Evaluation
    nb4 = create_wbs_nb4()
    execute_and_save(nb4, "04_wbs_evaluation.ipynb")
    
    # 5. WBS 6: Deployment & Policy
    nb5 = create_wbs_nb5()
    execute_and_save(nb5, "05_wbs_deployment_dan_kebijakan.ipynb")
    
    # 6. WBS Master Pipeline
    nb0 = create_wbs_nb0()
    execute_and_save(nb0, "00_wbs_master_pipeline.ipynb")
    
    print("\n🎉 SELURUH WBS NOTEBOOKS BERHASIL DIBUAT DAN DIEKSEKUSI!")
    print(f"Lokasi: {NOTEBOOKS_DIR}")

if __name__ == "__main__":
    main()
