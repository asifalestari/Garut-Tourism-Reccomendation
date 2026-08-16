"""
review_scraper.py - Controller module handling Google Maps review scraping process (Open, Scroll, Parse, Save).
"""
import csv
import logging
import os
import random
import time
from datetime import datetime, timezone
from playwright.sync_api import Page, TimeoutError
import re

from scraper.browser import BrowserManager
from scraper.parser import parse_review_cards, _count_unique_cards
from scraper import selectors

logger = logging.getLogger("pipeline")

import json

RESUME_FILE = "data/raw/.resume_info.json"


class BlockingDialogError(Exception):
    """Exception raised when a Google Login popup or other blocking dialog is detected."""
    pass


def save_resume_progress(index: int) -> None:
    os.makedirs(os.path.dirname(RESUME_FILE), exist_ok=True)
    try:
        with open(RESUME_FILE, "w") as f:
            json.dump({"last_processed_index": index}, f)
    except Exception as e:
        logger.warning(f"Failed to save resume progress: {e}")


def load_resume_progress() -> int:
    if os.path.exists(RESUME_FILE):
        try:
            with open(RESUME_FILE, "r") as f:
                data = json.load(f)
                return data.get("last_processed_index", 0)
        except Exception:
            pass
    return 0


def check_for_blocking_dialogs(page: Page) -> None:
    """Checks for blocking Google sign-in dialogs or other overlays."""
    try:
        login_iframe = page.locator('iframe[src*="accounts.google.com"]')
        if login_iframe.count() > 0:
            logger.error("Google Login popup/prompt detected! Raising BlockingDialogError.")
            raise BlockingDialogError("Google login iframe detected")
        
        signin_promo = page.locator('div[role="dialog"]:has-text("Sign in"), div[role="dialog"]:has-text("Masuk")')
        if signin_promo.count() > 0:
            logger.error("Blocking Sign-In promo dialog detected! Raising BlockingDialogError.")
            raise BlockingDialogError("Google sign-in promo dialog detected")
    except BlockingDialogError:
        raise
    except Exception:
        pass


def dismiss_popups(page: Page) -> None:
    """Dismisses Google Maps consent or sign-in popups if visible."""
    for selector in selectors.DISMISS_BUTTONS:
        try:
            page.wait_for_selector(selector, timeout=500)
            btn = page.locator(selector).first
            logger.info(f"Dismissing overlay/cookie popup using selector: {selector}")
            btn.click()
            page.wait_for_timeout(500)
        except Exception:
            pass


def clean_text_for_match(text: str) -> str:
    text = text.lower().strip()
    # Normalize common abbreviations
    replacements = {
        r"\bgn\b": "gunung",
        r"\bjl\b": "jalan",
        r"\brm\b": "rumah makan",
        r"\bkp\b": "kampung",
        r"\bds\b": "desa",
        r"\bkec\b": "kecamatan",
        r"\bkab\b": "kabupaten",
    }
    for pattern, repl in replacements.items():
        text = re.sub(pattern, repl, text)
    return text


def click_best_search_result(page: Page, destination_name: str) -> bool:
    try:
        results = page.locator('a[href*="/maps/place/"]').all()
        if len(results) > 0:
            logger.info(f"Search results list detected. Found {len(results)} items.")
            best_match_idx = 0
            best_score = -1
            
            for idx, res in enumerate(results):
                aria_label = res.get_attribute("aria-label") or ""
                if not aria_label:
                    continue
                clean_aria = clean_text_for_match(aria_label)
                clean_dest = clean_text_for_match(destination_name)
                
                # Direct equality check gets highest score
                if clean_dest == clean_aria:
                    best_match_idx = idx
                    break
                
                # Word intersection score
                dest_words = set(re.findall(r"\w+", clean_dest))
                aria_words = set(re.findall(r"\w+", clean_aria))
                intersection = dest_words.intersection(aria_words)
                
                score = len(intersection) / max(len(dest_words), len(aria_words)) if dest_words else 0
                if score > best_score:
                    best_score = score
                    best_match_idx = idx
                    
            aria_label = results[best_match_idx].get_attribute("aria-label") or ""
            logger.info(f"Clicking search result #{best_match_idx} ('{aria_label.strip()}')")
            results[best_match_idx].click()
            page.wait_for_timeout(4000)
            return True
    except Exception as list_err:
        logger.warning(f"Error selecting best search result: {list_err}")
    return False


def check_and_recover_from_redirect(page: Page, destination_name: str, address: str = None) -> bool:
    """Detects if redirected to generic coordinates, and recovers using localized map searches."""
    if "/place/" not in page.url and "/search/" not in page.url:
        logger.info(f"Redirect detected (URL: {page.url}). Recovering via localized searches for '{destination_name}'...")
        
        # Construct prioritize recovery search queries
        queries = []
        if address:
            # Clean address parts to filter plus-codes/postal-codes
            parts = [p.strip() for p in address.split(",")]
            clean_parts = []
            for part in parts:
                if not part:
                    continue
                if "+" in part and len(part) < 15:
                    continue
                if part.isdigit():
                    continue
                clean_parts.append(part)
            if len(clean_parts) >= 2:
                location_context = ", ".join(clean_parts[-3:]) if len(clean_parts) >= 3 else ", ".join(clean_parts[-2:])
                queries.append(f"{destination_name}, {location_context}")
        
        # Regency and provincial specificity queries
        queries.append(f"{destination_name}, Garut")
        queries.append(f"{destination_name}, Garut, Jawa Barat")
        queries.append(destination_name)
        
        for query in queries:
            logger.info(f"Attempting redirect recovery search using query: '{query}'")
            try:
                page.goto("https://www.google.com/maps", timeout=60000)
                page.wait_for_selector('input[name="q"]', timeout=5000)
                page.fill('input[name="q"]', query)
                page.press('input[name="q"]', 'Enter')
                page.wait_for_timeout(5000)
                
                # Try clicking the best search result
                click_best_search_result(page, destination_name)
 
                # Verify place details page loaded
                page.wait_for_selector("h1", timeout=15000)
                logger.info(f"Successfully recovered from redirect using query: '{query}'")
                return True
            except Exception as recovery_err:
                logger.warning(f"Recovery attempt failed for query '{query}': {recovery_err}")
        
        logger.error("All redirect recovery query attempts failed.")
        return False
    return True


def is_entity_match(expected_name: str, actual_name: str, expected_address: str, actual_address: str) -> bool:
    if not actual_name:
        return False
    
    exp_name_clean = clean_text_for_match(expected_name)
    act_name_clean = clean_text_for_match(actual_name)
    
    # 1. Exact or substring match with coverage checks
    if exp_name_clean in act_name_clean or act_name_clean in exp_name_clean:
        exp_words = set(re.findall(r"\w+", exp_name_clean))
        act_words = set(re.findall(r"\w+", act_name_clean))
        intersection = exp_words.intersection(act_words)
        
        # Word coverage of expected name
        coverage_exp = len(intersection) / len(exp_words) if exp_words else 0
        if coverage_exp < 0.85:
            return False
        name_match = True
    else:
        # Word coverage check fallback
        exp_words = set(re.findall(r"\w+", exp_name_clean))
        act_words = set(re.findall(r"\w+", act_name_clean))
        intersection = exp_words.intersection(act_words)
        
        coverage_exp = len(intersection) / len(exp_words) if exp_words else 0
        if coverage_exp >= 0.85:
            name_match = True
        else:
            name_match = False
            
    # 2. Address check: if actual address exists, check if expected address keywords are in it
    address_match = True
    if expected_address and actual_address:
        exp_addr_clean = expected_address.lower().strip()
        act_addr_clean = actual_address.lower().strip()
        if exp_addr_clean[:15] in act_addr_clean or act_addr_clean[:15] in exp_addr_clean:
            address_match = True
        else:
            # Check overlap of key words (Garut, Cipanas, Tarogong, Pakenjeng, etc. > 4 chars)
            exp_keywords = [w for w in re.findall(r"\w+", exp_addr_clean) if len(w) > 4]
            act_keywords = [w for w in re.findall(r"\w+", act_addr_clean) if len(w) > 4]
            overlap = set(exp_keywords).intersection(set(act_keywords))
            if len(overlap) >= 1:
                address_match = True
            else:
                address_match = False
                
    return name_match and address_match


def get_actual_place_details(page: Page) -> tuple[str, str]:
    actual_name = ""
    actual_address = ""
    
    # Try h1
    try:
        h1s = page.locator("h1").all()
        for h1 in h1s:
            t = h1.inner_text().strip()
            if t and t.lower() not in ["hasil", "results"]:
                actual_name = t
                break
    except Exception:
        pass
        
    # Fallback to page title (split by " - Google Maps" or " – Google Maps")
    if not actual_name:
        try:
            title = page.title()
            if title:
                for separator in [" - Google Maps", " – Google Maps"]:
                    if separator in title:
                        actual_name = title.split(separator)[0].strip()
                        break
        except Exception:
            pass
            
    # Try address selector
    try:
        addr_loc = page.locator("button[data-item-id='address']").first
        if addr_loc.count() > 0:
            actual_address = addr_loc.inner_text().strip()
    except Exception:
        pass
        
    return actual_name, actual_address


def check_if_zero_reviews_or_closed(page: Page) -> bool:
    try:
        zero_indicators = [
            "belum ada ulasan",
            "tulis ulasan pertama",
            "no reviews",
            "be the first to write a review"
        ]
        page_text = page.locator("body").inner_text().lower()
        for ind in zero_indicators:
            if ind in page_text:
                return True
    except Exception:
        pass
    return False


def check_if_permanently_closed(page: Page) -> bool:
    try:
        closed_indicators = [
            "tutup permanen",
            "permanently closed"
        ]
        page_text = page.locator("body").inner_text().lower()
        for ind in closed_indicators:
            if ind in page_text:
                return True
    except Exception:
        pass
    return False


def open_review_page(page: Page, destination_url: str, destination_name: str, address: str = None) -> bool:
    """
    Opens the Google Maps destination URL, dismisses overlays,
    and clicks open the reviews tab/modal.
    """
    # Helper function to check if reviews are open or to try clicking to open them
    def try_open_reviews_flow() -> bool:
        # Check if already active
        for card_sel in selectors.REVIEW_CARD:
            try:
                if page.locator(card_sel).count() > 0:
                    logger.info("Reviews panel is already active/visible.")
                    return True
            except Exception:
                pass
                
        # Attempt to find standard tab button
        for selector in selectors.REVIEW_TAB_BUTTON:
            try:
                elem = page.locator(selector).first
                if elem.count() > 0 and elem.is_visible():
                    logger.info(f"Clicking review tab button using: {selector}")
                    elem.click()
                    page.wait_for_timeout(3000)
                    # Verify
                    for card_sel in selectors.REVIEW_CARD:
                        if page.locator(card_sel).count() > 0:
                            logger.info("Reviews panel successfully opened via tab click.")
                            return True
            except Exception:
                pass
                
        # Fallback button scan
        try:
            buttons = page.locator("button").all()
            for btn in buttons:
                try:
                    if btn.is_visible():
                        aria = (btn.get_attribute("aria-label") or "").lower()
                        text = (btn.inner_text() or "").lower()
                        if "ulasan" in aria or "reviews" in aria or "ulasan" in text or "reviews" in text:
                            # Exclude buttons for writing reviews (tulis, write, add, buat)
                            if any(x in aria or x in text for x in ["tulis", "write", "add", "buat"]):
                                continue
                            # Exclude tag/keyword filter buttons (e.g. "disebutkan dalam ... ulasan")
                            if any(x in aria or x in text for x in ["disebutkan dalam", "mentioned in"]):
                                continue
                            logger.info(f"Clicking review button candidate: '{text.strip()}' (aria: '{aria}')")
                            btn.click()
                            page.wait_for_timeout(3000)
                            for card_sel in selectors.REVIEW_CARD:
                                if page.locator(card_sel).count() > 0:
                                    logger.info("Reviews panel successfully opened via button scan.")
                                    return True
                except Exception:
                    pass
        except Exception:
            pass
            
        return False

    # A helper to do post-navigation setup (dismiss popups, handle search results click, verify redirect)
    def prepare_page_details() -> bool:
        dismiss_popups(page)
        if not check_and_recover_from_redirect(page, destination_name, address):
            logger.warning("Verification of destination name/address failed during redirect check.")
            return False
            
        # If search list is visible, click the best matching result item
        try:
            if click_best_search_result(page, destination_name):
                dismiss_popups(page)
                if not check_and_recover_from_redirect(page, destination_name, address):
                    return False
        except Exception as list_err:
            logger.warning(f"Error clicking search result: {list_err}")
            
        return True

    # Helper to check and recover entity match
    def verify_and_recover_entity() -> bool:
        act_name, act_address = get_actual_place_details(page)
        logger.info(f"Verifying entity: actual_name='{act_name}', actual_address='{act_address}'")
        
        if is_entity_match(destination_name, act_name, address, act_address):
            return True
            
        logger.warning(f"Entity mismatch. Expected: '{destination_name}' | Actual: '{act_name}'. Running recovery...")
        
        # Recovery Loop (1-2 attempts)
        for recovery_idx in range(1, 3):
            logger.info(f"Entity Recovery Attempt {recovery_idx}/2...")
            # Re-run redirect recovery
            check_and_recover_from_redirect(page, destination_name, address)
            page.wait_for_timeout(3000)
            
            # Dismiss overlays
            dismiss_popups(page)
            
            # Click search result if visible
            try:
                click_best_search_result(page, destination_name)
            except Exception:
                pass
                
            act_name, act_address = get_actual_place_details(page)
            logger.info(f"Verifying entity after recovery {recovery_idx}: actual_name='{act_name}'")
            if is_entity_match(destination_name, act_name, address, act_address):
                logger.info("Entity successfully verified after recovery!")
                return True
                
        return False

    # --- Initial Navigation ---
    try:
        page.goto(destination_url, timeout=60000)
        page.wait_for_timeout(5000)
        try:
            page.wait_for_selector("h1", timeout=10000)
            logger.info("Destination detail page loaded.")
        except Exception:
            logger.info("H1 header not loaded. Proceeding with checks...")
    except Exception as e:
        logger.error(f"Initial navigation failed: {e}")
        return False

    # --- STAGE 1: Initial Detection ---
    logger.info("[RECOVERY STAGE 1] Attempting initial reviews detection and opening...")
    if prepare_page_details():
        if check_if_permanently_closed(page):
            logger.warning(f"Destination '{destination_name}' is permanently closed. Skipping reviews opening.")
            raise ValueError("PERMANENTLY_CLOSED")
        if verify_and_recover_entity():
            if try_open_reviews_flow():
                logger.info("Reviews page successfully opened and verified in Stage 1 (Initial Detection).")
                return True
        else:
            raise ValueError("ENTITY_MISMATCH")
            
    # --- STAGE 2: Wait & Retry ---
    if check_if_zero_reviews_or_closed(page):
        logger.warning(f"Destination '{destination_name}' has 0 reviews or is closed. Skipping recovery.")
        return False

    logger.info("[RECOVERY STAGE 2] Initial detection failed or reviews panel missing. Starting Wait & Retry...")
    logger.info("   Wait & Retry: Waiting 3s...")
    page.wait_for_timeout(3000)
    if prepare_page_details():
        if verify_and_recover_entity():
            if try_open_reviews_flow():
                logger.info("Reviews page successfully opened and verified in Stage 2 (Wait & Retry).")
                return True
        else:
            raise ValueError("ENTITY_MISMATCH")

    logger.warning("❌ All recovery stages failed. Reviews panel could not be opened.")
    return False


def sort_reviews_by_newest(page: Page) -> bool:
    """
    Attempts to change review sorting to 'Newest' (Terbaru).
    Supports English and Indonesian interfaces.
    """
    import re
    logger.info("Opening review sort menu...")
    
    # Capture the state of the first review card before sorting to check for refresh
    first_review_before = ""
    try:
        for card_sel in selectors.REVIEW_CARD:
            loc = page.locator(card_sel).first
            if loc.count() > 0:
                first_review_before = loc.inner_text()
                break
    except Exception:
        pass

    try:
        # 1. Locate and click the Sort Button
        sort_btn = None
        
        # Try locator fallback strategies from most specific to most generic:
        for selector in [
            "button[aria-label*='relevan' i]",
            "button[aria-label*='relevant' i]",
            "button:has-text('Paling relevan')",
            "button:has-text('Most relevant')",
            "button[aria-label*='urutkan' i]",
            "button:has-text('Urutkan')",
            "button[aria-label*='sort' i]",
            "button:has-text('Sort')"
        ]:
            try:
                btn = page.locator(selector).first
                if btn.count() > 0 and btn.is_visible():
                    sort_btn = btn
                    logger.info(f"Found sort button using: {selector}")
                    break
            except Exception:
                pass

        sort_btn.click(timeout=3000)
        try:
            page.wait_for_selector(".mLuXec", timeout=3000)
        except Exception:
            page.wait_for_timeout(1000)
        
        # 2. Locate and click the "Newest" / "Terbaru" option
        logger.info("Selecting 'Newest' reviews...")
        newest_opt = None
        
        # a) .mLuXec (specific class for Google Maps reviews dropdown sort menu items)
        try:
            items = page.locator(".mLuXec").all()
            for item in items:
                text = (item.inner_text() or "").lower()
                if "terbaru" in text or "newest" in text:
                    newest_opt = item
                    logger.info(f"Found newest option using .mLuXec loop: {text.strip()}")
                    break
        except Exception:
            pass

        # b) role menuitem / menuitemradio with name regex
        if not newest_opt:
            try:
                for role_name in ["menuitem", "menuitemradio"]:
                    opt = page.get_by_role(role_name, name=re.compile(r"terbaru|newest", re.I)).first
                    if opt.count() > 0 and opt.is_visible():
                        newest_opt = opt
                        break
            except Exception:
                pass
            
        # c) Scan all role elements for text matching
        if not newest_opt:
            try:
                for role_name in ["menuitem", "menuitemradio"]:
                    items = page.locator(f"[role='{role_name}']").all()
                    for item in items:
                        text = (item.inner_text() or "").lower()
                        aria = (item.get_attribute("aria-label") or "").lower()
                        if "terbaru" in text or "newest" in text or "terbaru" in aria or "newest" in aria:
                            newest_opt = item
                            break
                    if newest_opt:
                        break
            except Exception:
                pass
                
        # d) Pure text selection wildcard
        if not newest_opt:
            try:
                opt = page.get_by_text(re.compile(r"terbaru|newest", re.I)).first
                if opt.count() > 0:
                    newest_opt = opt
            except Exception:
                pass

        if not newest_opt:
            logger.warning("Option 'Newest' / 'Terbaru' not found. Continue using default sorting.")
            return False

        newest_opt.click(force=True, timeout=3000)
        
        # 3. Wait for refresh
        logger.info("Waiting for reviews list to refresh...")
        page.wait_for_timeout(1000)
        
        # Check if first card content changed (up to 4 seconds wait)
        refreshed = False
        start_time = time.time()
        while time.time() - start_time < 4.0:
            try:
                first_card = None
                for card_sel in selectors.REVIEW_CARD:
                    loc = page.locator(card_sel).first
                    if loc.count() > 0:
                        first_card = loc
                        break
                if first_card is not None:
                    current_text = first_card.inner_text()
                    if current_text != first_review_before:
                        refreshed = True
                        break
            except Exception:
                pass
            page.wait_for_timeout(500)
            
        if refreshed:
            logger.info("Newest sorting successfully applied (refresh confirmed).")
        else:
            logger.info("Newest sorting clicked. Proceeding with scroll...")
            
        return True

    except Exception as e:
        logger.warning(f"Failed to change sorting to Newest: {e}. Continue using default sorting.")
        return False


def scroll_reviews(page: Page, max_reviews: int = 500, start_time: float = None, timeout: int = 180) -> None:
    """Men-scroll modal ulasan hingga ulasan paling bawah berumur lebih dari 12 bulan."""
    from config import settings
    from scraper.parser import parse_relative_date_to_months, _find_locator
    
    max_review_months = getattr(settings, "MAX_REVIEW_MONTHS", 12)
    logger.info(f"🔄 Memulai Smart Scroll (Batas Umur: {max_review_months} bulan, Hard Safety Limit: {max_reviews} ulasan)...")
    
    # Find active scroll container
    scroll_container = None
    all_selectors = ["div.m6QErb.dS8AEf:not(.ecceSd)"] + selectors.REVIEW_SCROLL_CONTAINER
    
    for selector in all_selectors:
        try:
            locators = page.locator(selector).all()
            for loc in locators:
                sh = loc.evaluate("el => el.scrollHeight")
                ch = loc.evaluate("el => el.clientHeight")
                if sh > ch:
                    scroll_container = loc
                    logger.info(f"Scroll container detected using: {selector} (scrollHeight={sh}, clientHeight={ch})")
                    break
            if scroll_container:
                break
        except Exception:
            pass

    if not scroll_container:
        # Fallback to old .first logic if dynamic check yielded nothing
        for selector in all_selectors:
            try:
                loc = page.locator(selector).first
                if loc.count() > 0:
                    scroll_container = loc
                    logger.info(f"Scroll container fallback (first) using: {selector}")
                    break
            except Exception:
                pass

    if not scroll_container:
        logger.warning("No scrollable container found. Smart Scroll skipped.")
        return

    last_count = 0
    same_count_counter = 0
    consecutive_older_scrolls = 0
    max_attempts = 50  # Hard safety limit on scroll attempts

    for attempt in range(1, max_attempts + 1):
        # Bounded processing time check
        if start_time is not None and (time.time() - start_time) > timeout:
            logger.warning("Timeout reached during scrolling. Stopping scroll.")
            break

        # 1. Count current visible review cards
        current_count = 0
        cards_locator = None
        for card_sel in selectors.REVIEW_CARD:
            try:
                locator = page.locator(card_sel)
                cnt = _count_unique_cards(locator)
                if cnt > current_count:
                    current_count = cnt
                    cards_locator = locator
            except Exception:
                pass

        logger.info(f"   [Scroll #{attempt}] {current_count} cards loaded so far...")

        # 2. Extract the age of the bottom-most review card
        bottom_age = None
        if cards_locator and current_count > 0:
            # Look at the last few elements in reverse order to find a valid date element
            # This is robust because nested elements might match the selector but not have the date.
            for idx in range(cards_locator.count() - 1, -1, -1):
                card = cards_locator.nth(idx)
                date_loc = _find_locator(card, selectors.REVIEW_DATE)
                if date_loc and date_loc.first.count() > 0:
                    try:
                        date_str = date_loc.first.inner_text().strip()
                        if date_str:
                            bottom_age = parse_relative_date_to_months(date_str)
                            break
                    except Exception:
                        pass

        # 3. Handle stop conditions
        # Condition A: Hard safety limit reached
        if current_count >= max_reviews:
            logger.info(f"🛑 Hard safety limit of {max_reviews} reviews reached. Stopping scroll.")
            break

        # Condition B: Bottom review is older than limit
        if bottom_age is not None:
            if bottom_age > max_review_months:
                consecutive_older_scrolls += 1
                logger.info(f"   ℹ️ Ulasan terbawah berumur {bottom_age:.2f} bulan (> {max_review_months} bulan). (Scroll tambahan ke-{consecutive_older_scrolls})")
                if consecutive_older_scrolls >= 3:
                    logger.info("✅ Ulasan terbawah tetap > 12 bulan setelah scroll tambahan. Hentikan scroll.")
                    break
            else:
                consecutive_older_scrolls = 0
        else:
            logger.debug("   ⚠️ Tidak dapat menemukan umur ulasan terbawah pada scroll ini.")

        # Condition C: No new reviews loaded (Google Maps end reached)
        if current_count == last_count:
            same_count_counter += 1
            if same_count_counter >= 3:  # 3 consecutive times with same count means end of list
                logger.info("ℹ️ Google Maps sudah tidak memiliki ulasan baru (habis). Hentikan scroll.")
                break
        else:
            same_count_counter = 0
            last_count = current_count

        # Execute scroll action
        try:
            scroll_container.evaluate("el => el.scrollTop = el.scrollHeight")
            page.keyboard.press("PageDown")
        except Exception as scroll_err:
            logger.warning(f"Scroll action failed: {scroll_err}")
            break

        page.wait_for_timeout(2000)


def log_failure(destination_name: str, url: str, error_stage: str, error_message: str, retry_count: int) -> None:
    """Logs a destination scraping failure to data/analysis/review_repair_failures.csv."""
    failure_file = "data/analysis/review_repair_failures.csv"
    os.makedirs(os.path.dirname(failure_file), exist_ok=True)
    fieldnames = ["destination_name", "url", "error_stage", "error_message", "retry_count", "timestamp"]
    file_exists = os.path.exists(failure_file)
    
    with open(failure_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "destination_name": destination_name,
            "url": url,
            "error_stage": error_stage,
            "error_message": error_message,
            "retry_count": retry_count,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })


def scrape_reviews(df, max_reviews_per_dest: int = None, output_file: str = "data/raw/reviews.csv") -> None:
    """Main pipeline execution loop for processing destination list and saving reviews."""
    from config import settings
    from scraper.parser import parse_relative_date_to_months

    # Resolve settings parameters
    max_reviews = max_reviews_per_dest if max_reviews_per_dest is not None else getattr(settings, "MAX_REVIEWS_PER_DESTINATION", 500)
    max_review_months = getattr(settings, "MAX_REVIEW_MONTHS", getattr(settings, "MAX_REVIEW_AGE_MONTHS", 12))

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    fieldnames = [
        "destination_name",
        "author",
        "rating",
        "review_date",
        "review_text",
        "scraped_at",
        "review_id",
        "has_text",
    ]

    # Initialize CSV header if not exists
    if not os.path.exists(output_file):
        with open(output_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

    with BrowserManager() as browser:
        page = browser.get_page()

        for index, row in df.iterrows():
            destination = row["name"]
            url = row["url"]
            address = row.get("address") if "address" in row else None

            # Retry attempt logic (1 try + 1 retry = max 2 attempts)
            success = False
            for try_idx in range(1, 3):
                start_time = time.time()
                error_stage = "Initialization"
                error_msg = ""
                
                logger.info(f"[{index+1}/{len(df)}] Processing: '{destination}' (Try {try_idx}/2)")
                
                try:
                    dest_timeout = 600 if max_reviews > 100 else 180
                    # 1. Open and load the reviews page
                    error_stage = "OpenReviewPage"
                    open_success = open_review_page(page, url, destination, address)
                    if not open_success:
                        raise RuntimeError("Failed to open review page/tab")

                    # Check for timeout
                    if time.time() - start_time > dest_timeout:
                        raise TimeoutError("Timeout reached during open_review_page")

                    # Check for blocking login popup
                    check_for_blocking_dialogs(page)

                    # 2. Sort reviews by Newest (Terbaru)
                    error_stage = "SortReviews"
                    sort_success = sort_reviews_by_newest(page)
                    if not sort_success:
                        # Log specific skip format required by user instruction #4
                        logger.warning(f"[SKIP] {destination} - failed to set review sorting")
                        raise RuntimeError("failed to set review sorting")

                    if time.time() - start_time > dest_timeout:
                        raise TimeoutError("Timeout reached during sorting")

                    check_for_blocking_dialogs(page)
 
                    # 2b. Check if newest review is already older than 2 years (24 months)
                    try:
                        from scraper.parser import _find_locator, parse_relative_date_to_months
                        cards_locator = _find_locator(page, selectors.REVIEW_CARD)
                        if cards_locator and cards_locator.count() > 0:
                            first_card = cards_locator.first
                            date_locator = _find_locator(first_card, selectors.REVIEW_DATE)
                            if date_locator:
                                review_date = date_locator.first.inner_text().strip()
                                age_months = parse_relative_date_to_months(review_date)
                                logger.info(f"Newest review age check: '{review_date}' ({age_months:.2f} months)")
                                if age_months >= 24:
                                    logger.warning(f"❌ '{destination}' has only reviews older than 24 months. Skipping targeted repair.")
                                    raise ValueError("REVIEWS_OLDER_THAN_2_YEARS")
                    except ValueError as ve:
                        raise ve
                    except Exception as date_check_err:
                        logger.warning(f"Could not perform newest review age check: {date_check_err}")

                    # 3. Scroll reviews panel to load target amount
                    error_stage = "ScrollReviews"
                    scroll_reviews(page, max_reviews=max_reviews, start_time=start_time, timeout=dest_timeout)

                    check_for_blocking_dialogs(page)

                    # 4. Parse reviews from the loaded cards
                    error_stage = "ParseReviews"
                    raw_reviews = parse_review_cards(page)
                    logger.info(f"Parser berhasil mengambil {len(raw_reviews)} raw ulasan dari DOM.")

                    # Filter and normalize parsed values
                    scraped_time = datetime.now(timezone.utc).isoformat()
                    final_reviews = []
                    failed_reviews_count = 0
                    filtered_reviews_count = 0

                    for r in raw_reviews:
                        try:
                            # Check age in months for chronological stop filtering
                            age_months = parse_relative_date_to_months(r["review_date"])
                            if max_review_months is not None and age_months > max_review_months:
                                filtered_reviews_count += 1
                                continue

                            r["destination_name"] = destination
                            r["scraped_at"] = scraped_time
                            final_reviews.append(r)
                        except Exception as parse_err:
                            failed_reviews_count += 1
                            logger.warning(f"Error normalisasi/filter ulasan: {parse_err}")

                    logger.info(
                        f"Hasil pemrosesan ulasan '{destination}': "
                        f"{len(final_reviews)} disimpan, "
                        f"{filtered_reviews_count} disaring (> {max_review_months} bulan), "
                        f"{failed_reviews_count} gagal diproses."
                    )

                    if not final_reviews:
                        logger.warning(f"Tidak ada ulasan ulasan baru/valid untuk disimpan untuk '{destination}'.")
                        success = True  # Marked as success (nothing new to save)
                        break

                    # 5. Save incrementally to CSV
                    with open(output_file, "a", newline="", encoding="utf-8") as f:
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        writer.writerows(final_reviews)

                    logger.info(f"💾 Simpan inkremental berhasil. {len(final_reviews)} ulasan disimpan untuk '{destination}'.")
                    success = True
                    break  # Break retry loop on success

                except Exception as e:
                    error_msg = str(e)
                    logger.error(f"Error on '{destination}' during '{error_stage}' (Attempt {try_idx}/2): {e}")
                    
                    if "ENTITY_MISMATCH" in error_msg or "PERMANENTLY_CLOSED" in error_msg or "REVIEWS_OLDER_THAN_2_YEARS" in error_msg:
                        logger.error(f"Aborting targeted repair for '{destination}' immediately: {error_msg}")
                        log_failure(destination, url, "VerificationGate", error_msg, try_idx - 1)
                        success = False
                        break
                    
                    if try_idx == 2:
                        # Log failure on last try
                        log_failure(destination, url, error_stage, error_msg, try_idx - 1)
                    else:
                        time.sleep(2)  # Brief wait before retry

            if not success:
                logger.warning(f"❌ '{destination}' failed all retry attempts. Skipping to next destination.")

            # Save resume progress to next index regardless of success/failure to keep pipeline moving
            save_resume_progress(index + 1)
            time.sleep(random.uniform(2.0, 4.0))