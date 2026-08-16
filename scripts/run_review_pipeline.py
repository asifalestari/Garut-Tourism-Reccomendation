import os
import sys
import logging
import time
import argparse
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger("review_pipeline")

from scraper.review_scraper import scrape_reviews, BlockingDialogError, load_resume_progress, RESUME_FILE

def main():
    parser = argparse.ArgumentParser(description="Google Maps Review Scraper Pipeline")
    parser.add_argument("--resume", action="store_true", help="Resume scraping from the last processed destination")
    parser.add_argument("--batch-size", type=int, default=50, help="Batch size of destinations to process per browser session (default 50)")
    parser.add_argument("--cooldown", type=int, default=10, help="Cooldown period in seconds between batches (default 10)")
    args = parser.parse_args()

    dest_path = "data/raw/destinations.csv"
    out_path = "data/raw/reviews.csv"
    
    if not os.path.exists(dest_path):
        logger.error(f"Destinations file not found at {dest_path}")
        return
        
    df = pd.read_csv(dest_path)
    # Filter only successfully scraped destinations
    df_success = df[df["status"] == "success"].copy()
    
    if df_success.empty:
        logger.warning("No successful destinations found in the input CSV.")
        return
        
    total_dest = len(df_success)
    logger.info(f"Loaded {total_dest} successful destinations.")

    # Ensure raw output dir exists
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Handle resume index parsing
    start_idx = 0
    if args.resume:
        start_idx = load_resume_progress()
        logger.info(f"Resuming pipeline from destination index: {start_idx}")
        if start_idx >= total_dest:
            logger.info("All successful destinations have already been processed.")
            return
        
        # Ensure reviews.csv exists, write headers only if not present
        if not os.path.exists(out_path):
            headers = ["destination_name", "author", "rating", "review_date", "review_text", "scraped_at", "review_id", "has_text"]
            with open(out_path, mode="w", encoding="utf-8", newline="") as f:
                import csv
                writer = csv.writer(f)
                writer.writerow(headers)
    else:
        # If starting fresh, truncate the CSV and delete any existing resume progress file
        if os.path.exists(RESUME_FILE):
            try:
                os.remove(RESUME_FILE)
            except Exception as e:
                logger.warning(f"Could not remove old resume file: {e}")
                
        logger.info("Initializing fresh reviews.csv schema...")
        headers = ["destination_name", "author", "rating", "review_date", "review_text", "scraped_at", "review_id", "has_text"]
        with open(out_path, mode="w", encoding="utf-8", newline="") as f:
            import csv
            writer = csv.writer(f)
            writer.writerow(headers)

    # Process destinations in batches
    batch_size = args.batch_size
    cooldown = args.cooldown
    current_idx = start_idx

    while current_idx < total_dest:
        end_idx = min(current_idx + batch_size, total_dest)
        df_batch = df_success.iloc[current_idx:end_idx]
        
        logger.info(f"🚀 Starting Batch: Processing destinations {current_idx + 1} to {end_idx} of {total_dest}...")
        
        try:
            scrape_reviews(df_batch)
        except BlockingDialogError as bde:
            logger.error(f"🛑 Pipeline terminated due to blocking dialog: {bde}")
            sys.exit(1)
        except KeyboardInterrupt:
            logger.info("🛑 Pipeline paused by user interrupt (KeyboardInterrupt).")
            sys.exit(0)
        except Exception as err:
            logger.error(f"🛑 Critical error occurred during batch: {err}")
            sys.exit(1)
            
        current_idx = end_idx
        if os.environ.get("TEST_STOP_AFTER_BATCH") == "1":
            logger.info("Stopping after first batch due to TEST_STOP_AFTER_BATCH env var.")
            sys.exit(0)
            
        if current_idx < total_dest:
            logger.info(f"☕ Batch finished. Waiting for cooldown of {cooldown} seconds before starting the next batch...")
            time.sleep(cooldown)

    logger.info("🎉 Full reviews scraping pipeline completed successfully.")

if __name__ == "__main__":
    main()
