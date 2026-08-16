import os
import sys
import json
import logging
import pandas as pd
from pathlib import Path

# Setup sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(PROJECT_ROOT / "logs" / "pipeline.log", encoding="utf-8")
    ]
)
logger = logging.getLogger("pipeline")

from scraper.search_scraper import get_all_garut_destination_urls
from scraper.destination_scraper import DestinationScraper

def main():
    logger.info("=== STARTING FULL AUTOMATED MULTI-KEYWORD DESTINATION PIPELINE ===")
    
    # Ensure raw directory exists
    os.makedirs(PROJECT_ROOT / "data" / "raw", exist_ok=True)
    
    # Step 1: Run search scraper to collect URLs automatically from all keywords
    logger.info("Step 1: Collecting unique destination URLs across all tourism categories...")
    get_all_garut_destination_urls()
    
    # Step 2: Read list of URLs from destination_urls.json
    urls_file = PROJECT_ROOT / "data" / "raw" / "destination_urls.json"
    if not urls_file.exists():
        logger.error(f"Error: {urls_file} was not generated. Exiting pipeline.")
        sys.exit(1)
        
    with open(urls_file, "r", encoding="utf-8") as f:
        urls = json.load(f)
        
    logger.info(f"Step 2: Loaded {len(urls)} unique destination URLs from JSON.")
    if not urls:
        logger.error("No URLs collected. Exiting pipeline.")
        sys.exit(1)
        
    # Step 3: Run DestinationScraper V2
    logger.info(f"Step 3: Starting scraping details for all {len(urls)} collected URLs...")
    output_csv = PROJECT_ROOT / "data" / "raw" / "destinations.csv"
    
    # Use polite but slightly faster scraping delay (1.0 to 2.0s) to handle larger datasets
    scraper = DestinationScraper(
        urls=urls,
        output_path=str(output_csv),
        delay_range=(1.0, 2.0)
    )
    
    df = scraper.run()
    
    # Step 4: Show summary and head
    total = len(df)
    success = len(df[df["status"] == "success"]) if "status" in df.columns else 0
    failed = len(df[df["status"] == "failed"]) if "status" in df.columns else 0
    
    logger.info("=== PIPELINE RUN COMPLETE ===")
    logger.info(f"Total processed : {total}")
    logger.info(f"Success         : {success}")
    logger.info(f"Failed          : {failed}")
    logger.info(f"Results saved to: {output_csv}")
    
    if not df.empty:
        print("\n=== FIRST 5 ROWS OF RAW DESTINATIONS ===")
        print(df.head(5)[["name", "category", "rating", "status"]].to_string())
    else:
        logger.warning("Scraping output DataFrame is empty.")

if __name__ == "__main__":
    main()
