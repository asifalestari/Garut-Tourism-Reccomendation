import time
import random
import logging
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Tuple

from config import settings
from scraper.browser import BrowserManager
from scraper.parser import parse_destination_detail

logger = logging.getLogger("pipeline")

class DestinationScraper:
    """
    Orchestrates the crawling of a list of Google Maps destination URLs.
    Extracts structured data and saves them to a CSV file.
    """
    def __init__(
        self,
        urls: List[str],
        output_path: str = "data/destinations.csv",
        delay_range: Tuple[float, float] = (1.0, 3.0)
    ) -> None:
        self.urls = urls
        self.output_path = Path(output_path)
        self.delay_range = delay_range
        self.results: List[Dict[str, Any]] = []

    def run(self) -> pd.DataFrame:
        """
        Executes the scraping loop over all target URLs.
        """
        logger.info(f"Starting destination scraping loop for {len(self.urls)} URLs.")
        
        # Ensure parent directory exists
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with BrowserManager() as manager:
            page = manager.get_page()
            
            for index, url in enumerate(self.urls):
                logger.info(f"[{index+1}/{len(self.urls)}] Navigating to: {url}")
                parsed_data = {
                    "name": "N/A",
                    "category": "N/A",
                    "rating": None,
                    "address": "N/A",
                    "scraped_at": pd.Timestamp.now().isoformat()
                }
                
                try:
                    # Navigate and load
                    page.goto(url, wait_until="domcontentloaded", timeout=settings.DEFAULT_TIMEOUT)

                    print("=" * 80)
                    print("CSV URL")
                    print(url)

                    print()

                    print("FINAL URL")
                    print(page.url)

                    print()

                    print("TITLE")
                    print(page.title())

                    print()

                    print("H1")
                    print(page.locator("h1").first.inner_text())
                    print("=" * 80)
                    
                    # Wait explicitly for h1 element
                    page.wait_for_selector("h1", timeout=5000)
                    
                    # Parse the page
                    parsed_data = parse_destination_detail(page)
                    parsed_data["url"] = url
                    parsed_data["status"] = "success"
                    self.results.append(parsed_data)
                    logger.info(f"Successfully scraped: {parsed_data['name']}")
                    
                except Exception as e:
                    logger.error(f"Error scraping URL {url}: {e}")
                    parsed_data["url"] = url
                    parsed_data["status"] = "failed"
                    self.results.append(parsed_data)
                    
                    # Apply random delay before continuing (except for last item)
                    if index < len(self.urls) - 1:
                        sleep_time = random.uniform(*self.delay_range)
                        logger.info(f"Sleeping for {sleep_time:.2f} seconds...")
                        time.sleep(sleep_time)
                    continue
                
                # Apply random delay for polite scraping (except for last item)
                if index < len(self.urls) - 1:
                    sleep_time = random.uniform(*self.delay_range)
                    logger.info(f"Sleeping for {sleep_time:.2f} seconds...")
                    time.sleep(sleep_time)
        
        # Convert results to DataFrame and save to CSV
        df = pd.DataFrame(self.results)
        df.to_csv(self.output_path, index=False)
        logger.info(f"Scraping complete. Saved {len(df)} results to {self.output_path}")
        return df

if __name__ == "__main__":
    # Small test loop if run directly
    logging.basicConfig(level=logging.INFO)
    sample_urls = [
        "https://www.google.com/maps/place/Kawah+Kamojang/@-7.1466667,107.7966667,15z"
    ]
    scraper = DestinationScraper(sample_urls, output_path="data/destinations_test_direct.csv")
    scraper.run()
