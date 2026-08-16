import time
import re
import logging
from typing import Optional
from playwright.sync_api import Page

logger = logging.getLogger("pipeline")

def parse_rating(rating_str: Optional[str]) -> float:
    """Parse average rating string (e.g., '4,5' or '4.5') to float."""
    if not rating_str:
        return 0.0
    try:
        # Standardize comma to dot
        clean_str = rating_str.replace(",", ".").strip()
        match = re.search(r"(\d+\.\d+|\d+)", clean_str)
        if match:
            return float(match.group(1))
    except Exception as e:
        logger.warning(f"Failed to parse rating string '{rating_str}': {e}")
    return 0.0

def parse_reviews_count(reviews_str: Optional[str]) -> int:
    """Parse total reviews count string (e.g., '1.243 ulasan' or '567 reviews') to int."""
    if not reviews_str:
        return 0
    try:
        # Remove dots (Indonesian thousand separator) and commas
        clean_str = reviews_str.replace(".", "").replace(",", "").strip()
        match = re.search(r"(\d+)", clean_str)
        if match:
            return int(match.group(1))
    except Exception as e:
        logger.warning(f"Failed to parse reviews count string '{reviews_str}': {e}")
    return 0

def scroll_element_by_selector(page: Page, selector: str, max_scrolls: int = 50, pause_time: float = 2.0) -> None:
    """
    Scroll a specific element with the given selector to trigger lazy loading.
    Useful for scrolling the destination feed or the reviews list.
    """
    try:
        page.wait_for_selector(selector, timeout=10000)
        element = page.locator(selector).first
        
        last_height = page.evaluate("(el) => el.scrollHeight", element.element_handle())
        
        for i in range(max_scrolls):
            # Scroll down to bottom of the element
            page.evaluate("(el) => el.scrollTo(0, el.scrollHeight)", element.element_handle())
            time.sleep(pause_time)
            
            new_height = page.evaluate("(el) => el.scrollHeight", element.element_handle())
            if new_height == last_height:
                # Try scrolling a bit up and down again to force loading
                page.evaluate("(el) => el.scrollTo(0, el.scrollHeight - 500)", element.element_handle())
                time.sleep(0.5)
                page.evaluate("(el) => el.scrollTo(0, el.scrollHeight)", element.element_handle())
                time.sleep(pause_time)
                new_height = page.evaluate("(el) => el.scrollHeight", element.element_handle())
                if new_height == last_height:
                    logger.info("Reached the end of scrollable area.")
                    break
            last_height = new_height
            logger.info(f"Scrolling progress: step {i+1}/{max_scrolls}")
    except Exception as e:
        logger.error(f"Error while scrolling selector '{selector}': {e}")

def clean_text(raw_text: str) -> str:
    """
    Cleans raw scraped text:
    - Removes non-ASCII characters and special icons (e.g., Maps pin icon )
    - Removes duplicate spaces and excessive newlines
    - Trims starting and ending whitespace
    """
    if not raw_text:
        return ""
        
    # Remove non-ASCII characters
    cleaned = re.sub(r"[^\x00-\x7F]+", " ", raw_text)
    
    # Remove duplicate spaces and newlines
    cleaned = re.sub(r"\s+", " ", cleaned)
    
    return cleaned.strip()

