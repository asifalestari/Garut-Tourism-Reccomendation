import time
import logging
import json
from pathlib import Path
from typing import List

from config import settings
from scraper.browser import BrowserManager
from scraper import selectors

logger = logging.getLogger("pipeline")

# def get_garut_destination_urls(keyword="Wisata Kabupaten Garut", max_results=60) -> List[str]:
#     """
#     Search for a keyword on Google Maps and auto-scroll the results feed
#     to gather destination place URLs. Saves URLs to data/raw/destination_urls.json.
#     """
#     logger.info(f"Searching for target keyword: '{keyword}' on Google Maps...")
    
#     # Ensure data/raw directory exists
#     out_dir = Path(settings.RAW_DATA_DIR)
#     out_dir.mkdir(parents=True, exist_ok=True)
#     out_file = out_dir / "destination_urls.json"
    
#     collected_urls = set()
    
#     with BrowserManager() as manager:
#         page = manager.get_page()
        
#         # Navigate to Google Maps
#         page.goto(settings.BASE_MAPS_URL, wait_until="domcontentloaded", timeout=settings.DEFAULT_TIMEOUT)
        
#         # Wait for search input
#         page.wait_for_selector(selectors.SEARCH_INPUT, timeout=settings.DEFAULT_TIMEOUT)
#         page.fill(selectors.SEARCH_INPUT, keyword)
#         page.press(selectors.SEARCH_INPUT, "Enter")
        
#         # Wait for results panel or search to finish
#         try:
#             page.wait_for_selector(selectors.RESULT_FEED, timeout=10000)
#             logger.info("Google Maps search results feed container detected.")
#         except Exception as e:
#             logger.warning(f"Could not find RESULT_FEED: {e}. Proceeding with fallback delay.")
#             time.sleep(5)
            
#         no_new_results_count = 0
#         prev_count = 0
        
#         # Start scrolling loop
#         while len(collected_urls) < max_results and no_new_results_count < 10:
#             feed = page.locator(selectors.RESULT_FEED)
#             if feed.count() > 0:
#                 feed.first.evaluate("el => el.scrollTo(0, el.scrollHeight)")
#             else:
#                 page.evaluate("window.scrollBy(0, window.innerHeight)")
                
#             # Allow items to load
#             time.sleep(2.0)
            
#             # Find place card link elements
#             links = page.locator(f'{selectors.RESULT_ITEM_LINK}, a.hfA2B').all()
#             for link in links:
#                 try:
#                     href = link.get_attribute("href")
#                     if href and "/maps/place/" in href:
#                         collected_urls.add(href)
#                 except Exception:
#                     pass
            
#             current_count = len(collected_urls)
#             logger.info(f"Collected {current_count} destination URLs so far...")
            
#             if current_count == prev_count:
#                 no_new_results_count += 1
#             else:
#                 no_new_results_count = 0
                
#             prev_count = current_count
            
#             # Check end of list
#             content = page.content()
#             if "ujung daftar" in content or "reached the end" in content:
#                 logger.info("Reached the end of Google Maps results feed.")
#                 break
                
#     url_list = list(collected_urls)[:max_results]
#     logger.info(f"Total collected and selected URLs: {len(url_list)}")
    
#     # Save to file
#     with open(out_file, "w", encoding="utf-8") as f:
#         json.dump(url_list, f, indent=4)
#     logger.info(f"Saved {len(url_list)} URLs to {out_file}")
    
#     return url_list

def get_all_garut_destination_urls() -> List[str]:
    """
    Search for multiple categories of tourist destinations in Garut on Google Maps,
    auto-scroll to gather all links, and perform automatic deduplication.
    Saves the full list to data/raw/destination_urls.json.
    """
    SEARCH_KEYWORDS = [
        "Wisata Kabupaten Garut",
        "Curug di Garut",
        "Pantai di Garut",
        "Pemandian Air Panas Garut",
        "Wisata Alam Garut",
        "Situ Danau Garut",
        "Taman Wisata Garut",
        "Desa Wisata Garut",
        "Wisata Kuliner Garut"
    ]
    
    out_dir = Path(settings.RAW_DATA_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "destination_urls.json"
    
    collected_urls = set()
    
    with BrowserManager() as manager:
        page = manager.get_page()
        
        # Navigate to Google Maps
        page.goto(settings.BASE_MAPS_URL, wait_until="domcontentloaded", timeout=settings.DEFAULT_TIMEOUT)
        
        for idx, keyword in enumerate(SEARCH_KEYWORDS):
            logger.info(f"[{idx+1}/{len(SEARCH_KEYWORDS)}] Searching for keyword: '{keyword}' on Google Maps...")
            try:
                # Wait for search input
                page.wait_for_selector(selectors.SEARCH_INPUT, timeout=settings.DEFAULT_TIMEOUT)
                page.fill(selectors.SEARCH_INPUT, keyword)
                page.press(selectors.SEARCH_INPUT, "Enter")

                page.wait_for_timeout(5000)

                print("="*80)
                print("CURRENT URL")
                print(page.url)

                print("="*80)
                print("TITLE")
                print(page.title())

                print("="*80)
                print("HAS FEED :", page.locator("div[role='feed']").count())

                print("="*80)
                print("HAS PLACE LINK :", page.locator("a[href*='/maps/place/']").count())

                print("="*80)
                print(page.content()[:1000])
                
                # Wait for feed to load or update
                try:
                    page.wait_for_selector(selectors.RESULT_FEED, timeout=10000)
                    logger.info("Google Maps search results feed container detected.")
                except Exception:
                    logger.warning(f"RESULT_FEED not found for keyword '{keyword}'. Sleeping 5s fallback.")
                    time.sleep(5)
                
                # Scroll results feed
                no_new_results_count = 0
                prev_count = len(collected_urls)
                scroll_count = 0
                max_scrolls = 20 # Limit scrolls per keyword
                
                while no_new_results_count < 5 and scroll_count < max_scrolls:
                    feed = page.locator(selectors.RESULT_FEED)
                    if feed.count() > 0:
                        feed.first.evaluate("el => el.scrollTo(0, el.scrollHeight)")
                    else:
                        page.evaluate("window.scrollBy(0, window.innerHeight)")
                    
                    time.sleep(1.5)
                    scroll_count += 1
                    
                    # Extract URLs
                    links = page.locator(f'{selectors.RESULT_ITEM_LINK}, a.hfA2B').all()
                    for link in links:
                        try:
                            href = link.get_attribute("href")
                            if href and "/maps/place/" in href:
                                collected_urls.add(href)
                        except Exception:
                            pass
                    
                    current_count = len(collected_urls)
                    if current_count == prev_count:
                        no_new_results_count += 1
                    else:
                        no_new_results_count = 0
                        
                    prev_count = current_count
                    
                    # Check end of list
                    content = page.content()
                    if "ujung daftar" in content or "reached the end" in content:
                        logger.info("Reached the end of Google Maps feed for this query.")
                        break
                
                logger.info(f"Finished search for '{keyword}'. Cumulative unique URLs: {len(collected_urls)}")
                
            except Exception as kw_err:
                logger.error(f"Error searching keyword '{keyword}': {kw_err}")
                
    url_list = list(collected_urls)
    logger.info(f"Scraping complete. Total collected unique URLs: {len(url_list)}")
    
    # Save to file
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(url_list, f, indent=4)
    logger.info(f"Saved {len(url_list)} unique URLs to {out_file}")
    
    return url_list

if __name__ == "__main__":
    # Test execution
    logging.basicConfig(level=logging.INFO)
    urls = get_all_garut_destination_urls()
    print("Total unique URLs:", len(urls))
