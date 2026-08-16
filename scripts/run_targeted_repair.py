import os
import sys
import logging
import argparse
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger("targeted_repair")

from scraper.review_scraper import scrape_reviews, BlockingDialogError

def main():
    parser = argparse.ArgumentParser(description="Google Maps Review Targeted Repair Scraper")
    parser.add_argument("--smoke-test", action="store_true", help="Run only for Tirtagangga Hot Spring Resort Cipanas")
    parser.add_argument("--priority", type=str, default="HIGH,MEDIUM", help="Comma-separated priorities to run (e.g. HIGH or HIGH,MEDIUM)")
    parser.add_argument("--batch-size", type=int, default=10, help="Batch size of destinations to process per browser session (default 10)")
    parser.add_argument("--cooldown", type=int, default=10, help="Cooldown period in seconds between batches (default 10)")
    parser.add_argument("--force", action="store_true", help="Force re-scraping even if destination already exists in reviews_repaired.csv")
    args = parser.parse_args()

    targets_csv = "data/analysis/review_repair_targets.csv"
    dest_csv = "data/raw/destinations.csv"
    output_file = "data/raw/reviews_repaired.csv"

    if not os.path.exists(targets_csv):
        logger.error(f"Repair targets file not found: {targets_csv}. Please run audit first.")
        sys.exit(1)
    if not os.path.exists(dest_csv):
        logger.error(f"Destinations file not found: {dest_csv}")
        sys.exit(1)

    df_targets = pd.read_csv(targets_csv)
    df_dests = pd.read_csv(dest_csv)

    # 1. Filter targets based on args
    if args.smoke_test:
        target_names = ["Tirtagangga Hot Spring Resort Cipanas", "Curug Orete"]
        logger.info("SMOKE TEST MODE: Targeting 'Tirtagangga Hot Spring Resort Cipanas' and 'Curug Orete'.")
    else:
        allowed_priorities = [p.strip().upper() for p in args.priority.split(",")]
        df_filtered_targets = df_targets[df_targets["priority"].isin(allowed_priorities)]
        target_names = df_filtered_targets["destination_name"].tolist()
        logger.info(f"Targeting priorities: {allowed_priorities}. Found {len(target_names)} target names.")

    if not target_names:
        logger.warning("No destinations matched the target criteria. Exiting.")
        return

    # 2. Get destination details (url, address, name) from destinations.csv
    df_dests_success = df_dests[df_dests["status"] == "success"].copy()
    df_run = df_dests_success[df_dests_success["name"].isin(target_names)].copy()

    if df_run.empty:
        logger.error("Could not find matching successful destinations in destinations.csv for targets.")
        sys.exit(1)

    # 3. Resume logic: check what is already in reviews_repaired.csv
    if os.path.exists(output_file) and not args.force and not args.smoke_test:
        try:
            df_existing = pd.read_csv(output_file)
            completed = set(df_existing["destination_name"].dropna().unique())
            logger.info(f"Found existing reviews_repaired.csv. {len(completed)} destinations already processed.")
            df_run = df_run[~df_run["name"].isin(completed)]
        except Exception as e:
            logger.warning(f"Error reading existing reviews_repaired.csv: {e}. Starting fresh.")

    total_to_run = len(df_run)
    logger.info(f"Total destinations to scrape now: {total_to_run}")

    if total_to_run == 0:
        logger.info("All target destinations have already been scraped.")
        return

    # 4. Run in batches
    batch_size = args.batch_size
    cooldown = args.cooldown
    current_idx = 0

    while current_idx < total_to_run:
        end_idx = min(current_idx + batch_size, total_to_run)
        df_batch = df_run.iloc[current_idx:end_idx]
        
        logger.info(f"🚀 Starting Batch: Processing repair targets {current_idx + 1} to {end_idx} of {total_to_run}...")
        
        try:
            # Limit reviews per destination to 500 for repair (to be fast and safe, or default to settings)
            scrape_reviews(df_batch, max_reviews_per_dest=500, output_file=output_file)
        except BlockingDialogError as bde:
            logger.error(f"🛑 Pipeline terminated due to blocking dialog: {bde}")
            sys.exit(1)
        except KeyboardInterrupt:
            logger.info("🛑 Pipeline paused by user interrupt.")
            sys.exit(0)
        except Exception as err:
            logger.error(f"🛑 Critical error occurred during batch: {err}")
            sys.exit(1)

        current_idx = end_idx
        if current_idx < total_to_run:
            logger.info(f"☕ Batch finished. Waiting for cooldown of {cooldown} seconds before next batch...")
            import time
            time.sleep(cooldown)

    logger.info("🎉 Targeted repair scraping completed successfully.")

if __name__ == "__main__":
    main()
