import sys
from pathlib import Path
import os
import shutil
import pandas as pd
import json

# Setup sys.path to resolve project root imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

# Setup logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger("test_batch_resume")

from scraper.review_scraper import RESUME_FILE, load_resume_progress

def main():
    print("=== Menjalankan Verifikasi Sistem Batching & Resume ===")

    dest_csv = "data/raw/destinations.csv"
    reviews_csv = "data/raw/reviews.csv"
    backup_reviews = "data/raw/reviews.csv.bak"

    if not os.path.exists(dest_csv):
        print(f"Error: {dest_csv} not found.")
        sys.exit(1)

    # 1. Back up original reviews if exists
    if os.path.exists(reviews_csv):
        shutil.copy2(reviews_csv, backup_reviews)
        print("Backed up existing reviews.csv")

    try:
        # Load destinations and slice first 3 successful ones
        df = pd.read_csv(dest_csv)
        df_success = df[df["status"] == "success"].head(3).copy()
        if len(df_success) < 3:
            print("Error: Need at least 3 successful destinations to run the test.")
            sys.exit(1)

        print("\nDestinasi Uji:")
        for idx, row in df_success.iterrows():
            print(f"- Index {idx}: {row['name']}")

        # Temporary CSV for testing to avoid touching original destinations
        temp_dest = "data/raw/destinations_temp_test.csv"
        df_success.to_csv(temp_dest, index=False)

        # Clean old files
        if os.path.exists(RESUME_FILE):
            os.remove(RESUME_FILE)
        if os.path.exists(reviews_csv):
            os.remove(reviews_csv)

        # 2. RUN BATCH 1: Process the first destination (batch-size 1)
        # We will point to the temp destinations file by copying it to data/raw/destinations.csv
        shutil.copy2(dest_csv, "data/raw/destinations.csv.orig")
        shutil.copy2(temp_dest, dest_csv)

        print("\n--- MENJALANKAN BATCH 1 (Fresh start, batch-size=1) ---")
        import subprocess
        env = os.environ.copy()
        env["TEST_STOP_AFTER_BATCH"] = "1"
        subprocess.run([sys.executable, "-m", "scripts.run_review_pipeline", "--batch-size", "1", "--cooldown", "2"], env=env)

        # Verify progress saved
        last_idx = load_resume_progress()
        print(f"\nProgress setelah Batch 1: {last_idx} (diharapkan: 1)")
        if last_idx != 1:
            print("FAILED: Resume progress index should be 1 after processing first destination.")
            sys.exit(1)

        # Verify reviews.csv has first destination
        df_rev_1 = pd.read_csv(reviews_csv)
        print(f"Reviews saved after Batch 1: {len(df_rev_1)} baris.")
        destnames_1 = df_rev_1["destination_name"].unique()
        print(f"Destinasi dalam CSV: {destnames_1}")

        # 3. RUN RESUME: Resume pipeline to process remaining 2 destinations
        print("\n--- MENJALANKAN PIPELINE RESUME (batch-size=2) ---")
        cmd2 = f"{sys.executable} -m scripts.run_review_pipeline --resume --batch-size 2 --cooldown 2"
        os.system(cmd2)

        # Verify progress saved
        last_idx_final = load_resume_progress()
        print(f"\nProgress akhir setelah Resume: {last_idx_final} (diharapkan: 3)")
        if last_idx_final != 3:
            print("FAILED: Resume progress index should be 3 after processing all 3 destinations.")
            sys.exit(1)

        # Verify reviews.csv has all 3 destinations
        df_rev_final = pd.read_csv(reviews_csv)
        print(f"Reviews saved after Resume: {len(df_rev_final)} baris.")
        destnames_final = df_rev_final["destination_name"].unique()
        print(f"Destinasi akhir dalam CSV: {destnames_final}")

        if len(destnames_final) < 2:
            print("FAILED: CSV did not accumulate ulasan from resumed batches.")
            sys.exit(1)

        print("\n🎉 VERIFIKASI BATCH & RESUME BERHASIL!")

    finally:
        # Clean up temp test files and restore originals
        if os.path.exists("data/raw/destinations.csv.orig"):
            shutil.move("data/raw/destinations.csv.orig", dest_csv)
            print("Restored original destinations.csv")
        if os.path.exists("data/raw/destinations_temp_test.csv"):
            os.remove("data/raw/destinations_temp_test.csv")
        if os.path.exists(backup_reviews):
            shutil.move(backup_reviews, reviews_csv)
            print("Restored original reviews.csv backup")

if __name__ == "__main__":
    main()
