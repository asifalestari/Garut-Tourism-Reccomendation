import logging
import re
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
from playwright.sync_api import Page

from scraper import selectors
from scraper.utils import clean_text

logger = logging.getLogger("pipeline")

def parse_destination_detail(page: Page) -> Dict[str, Any]:
    """
    Parses destination details from a Google Maps detail Page.
    Returns a dictionary of cleaned attributes.
    """
    logger.info("Parsing destination detail page...")
    
    # Selectors dict
    sel = selectors.DESTINATION_DETAIL_SELECTORS
    
    # 1. Parse Name
    name = "N/A"
    try:
        name_el = page.locator(sel["name"]).first
        if name_el.count() > 0:
            name = clean_text(name_el.inner_text())
    except Exception as e:
        logger.warning(f"Failed to parse name: {e}")

    # 2. Parse Category
    category = "N/A"
    try:
        category_el = page.locator(sel["category"]).first
        if category_el.count() > 0:
            category = clean_text(category_el.inner_text())
    except Exception as e:
        logger.warning(f"Failed to parse category: {e}")

    # 3. Parse Rating (float or None)
    rating: Optional[float] = None
    try:
        rating_el = page.locator(sel["rating"]).first
        rating_str = ""
        if rating_el.count() > 0:
            rating_str = clean_text(rating_el.inner_text())
        else:
            # Fallback rating container
            fallback_rating = page.locator("div.F7nice").first
            if fallback_rating.count() > 0:
                inner = clean_text(fallback_rating.inner_text())
                rating_str = inner.split(" ")[0] if " " in inner else inner[:3]
                
        if rating_str:
            # Try to convert to float
            try:
                # Replace comma with dot for Indonesian rating locale
                cleaned_rating = rating_str.replace(",", ".").strip()
                rating = float(cleaned_rating)
            except ValueError:
                rating = None
    except Exception as e:
        logger.warning(f"Failed to parse rating: {e}")

    # 4. Parse Address
    address = "N/A"
    try:
        address_el = page.locator(sel["address"]).first
        if address_el.count() > 0:
            address = clean_text(address_el.inner_text())
    except Exception as e:
        logger.warning(f"Failed to parse address: {e}")

    # 5. Timestamp
    scraped_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    result = {
        "name": name,
        "category": category,
        "rating": rating,
        "address": address,
        "scraped_at": scraped_at
    }
    
    logger.info(f"Parsed attributes: {result}")
    return result


def _count_unique_cards(locator) -> int:
    """
    Counts the number of unique review cards in a locator based on data-review-id
    to prevent child elements sharing the same ID from inflating the count.
    """
    try:
        count = locator.count()
        seen = set()
        for idx in range(count):
            rid = locator.nth(idx).get_attribute("data-review-id")
            if rid:
                seen.add(rid)
        return len(seen) if seen else count
    except Exception:
        return 0


def _find_locator(root, selector_list: List[str]):
    """
    Finds and returns the locator that matches the maximum number of active elements in the DOM.
    If resolving review cards, validates them by checking for the presence of card signature child elements.
    """
    best_locator = None
    max_count = 0
    best_selector = None
    
    # Check if this search is for the review cards
    is_review_card_search = (selector_list == selectors.REVIEW_CARD)
    counts_log = []
    
    for selector in selector_list:
        locator = root.locator(selector)
        try:
            count = locator.count()
            if is_review_card_search:
                unique_count = _count_unique_cards(locator)
                counts_log.append(f"'{selector}': {count} (unik: {unique_count})")
                count = unique_count
                
            if count > 0:
                # Validation for review card elements
                if is_review_card_search:
                    first_el = locator.first
                    has_author = False
                    has_rating = False
                    # Check author/rating inside the first card
                    for auth_sel in selectors.REVIEW_AUTHOR:
                        if first_el.locator(auth_sel).count() > 0:
                            has_author = True
                            break
                    for rating_sel in selectors.REVIEW_RATING:
                        if first_el.locator(rating_sel).count() > 0:
                            has_rating = True
                            break
                    if not (has_author or has_rating):
                        continue # Skip this selector as it does not look like a review card

                if count > max_count:
                    max_count = count
                    best_locator = locator
                    best_selector = selector
        except Exception:
            pass
            
    if is_review_card_search:
        logger.info(f"Review card selector counts: {', '.join(counts_log)}")
        if best_selector:
            logger.info(f"Selected review card selector: '{best_selector}' (matched {max_count} cards)")
            
    if best_locator is not None:
        return best_locator
        
    # Standard fallback selector search
    for selector in selector_list:
        locator = root.locator(selector)
        try:
            if locator.count() > 0:
                if is_review_card_search:
                    logger.info(f"Fallback to first active selector: '{selector}' (matched {locator.count()} cards)")
                return locator
        except Exception:
            pass
            
    return None


def parse_relative_date_to_months(date_str: str) -> float:
    """
    Parses a relative date string (e.g., '3 hari lalu', '5 months ago', 'setahun lalu')
    and returns the estimated age in months.
    """
    if not date_str or date_str == "-":
        return 0.0

    s = date_str.lower()
    s = s.replace("diedit", "").replace("edited", "").strip()

    # Singular word conversions
    if s in ["sebulan lalu", "sebulan", "a month ago", "a month"]:
        return 1.0
    if s in ["setahun lalu", "setahun", "a year ago", "a year"]:
        return 12.0
    if s in ["seminggu lalu", "seminggu", "a week ago", "a week"]:
        return 0.25
    if s in ["kemarin", "sehari lalu", "yesterday", "a day ago"]:
        return 0.03

    # Extract dynamic numbers
    match = re.search(r"(\d+)", s)
    if match:
        val = float(match.group(1))
        if "hari" in s or "day" in s:
            return val / 30.0
        if "minggu" in s or "week" in s:
            return val / 4.0
        if "bulan" in s or "month" in s:
            return val
        if "tahun" in s or "year" in s:
            return val * 12.0

    return 0.0


def clean_review_text(text: str) -> str:
    """
    Normalizes review text by replacing excess newlines and double spaces with a single space,
    while preserving emoji and all Unicode characters.
    """
    if not text:
        return ""
    # Replace all whitespace sequences (newlines, tabs, multiple spaces) with a single space
    cleaned = re.sub(r"\s+", " ", text)
    return cleaned.strip()


def _expand_all_reviews(page: Page):
    """
    Locates and clicks all 'Lainnya' / 'More' buttons inside the loaded review cards.
    Restricts search and clicks strictly to the container of each review card.
    Verifies that the element is a button and does not navigate away.
    """
    logger.info("Expanding long reviews...")
    cards_locator = _find_locator(page, selectors.REVIEW_CARD)
    if not cards_locator:
        return
    
    try:
        total_cards = cards_locator.count()
        for idx in range(total_cards):
            card = cards_locator.nth(idx)
            for selector in selectors.MORE_BUTTON:
                try:
                    # Target the button strictly within the current review card's sub-tree
                    btn = card.locator(selector)
                    if btn.count() > 0 and btn.is_visible():
                        # Verify that matched element is a button and not an anchor / navigation element
                        tag_name = (btn.evaluate("el => el.tagName") or "").lower()
                        role = (btn.get_attribute("role") or "").lower()
                        href = btn.get_attribute("href")
                        
                        if tag_name == "a" or href is not None:
                            logger.warning(f"Skipping click: Element is an anchor or has an href attribute (tag: {tag_name}, href: {href})")
                            continue
                            
                        if tag_name != "button" and role != "button":
                            logger.warning(f"Skipping click: Element is not a button/role=button (tag: {tag_name}, role: {role})")
                            continue

                        btn.click(timeout=1500)
                        page.wait_for_timeout(200)  # Wait for expand animation
                        
                        # Verify the page is still a Google Maps page
                        if "google.com/maps" not in page.url:
                            logger.warning(f"Navigation to non-Maps domain detected: '{page.url}'. Aborting expansion.")
                            # Attempt to navigate back and restore page context
                            try:
                                page.go_back()
                                page.wait_for_timeout(1000)
                            except Exception as nav_err:
                                logger.warning(f"Failed to navigate back: {nav_err}")
                            return  # Abort expansion process to proceed with parsing
                            
                        break  # Found and clicked, move to the next card
                except Exception:
                    pass
    except Exception as e:
        logger.warning(f"Error expanding review cards: {e}")


def parse_review_cards(page: Page) -> List[Dict[str, Any]]:
    """
    Parses and extracts review data from all currently loaded cards in the review tab.
    Does not navigate, scroll, or manage tab clicks.
    """
    logger.info("Parsing review cards...")
    reviews = []
    seen_review = set()

    # Expand any truncated long review texts
    _expand_all_reviews(page)

    # 1. Parse Destination Name
    destination_name = "N/A"
    title_locator = _find_locator(page, selectors.DESTINATION_NAME)
    if title_locator:
        try:
            destination_name = title_locator.first.inner_text().strip()
        except Exception:
            pass

    # Fallback to page title if H1 name is not found/active in DOM
    if destination_name == "N/A" or not destination_name:
        try:
            page_title = page.title()
            if " - Google Maps" in page_title:
                destination_name = page_title.replace(" - Google Maps", "").strip()
            else:
                destination_name = page_title.strip()
        except Exception:
            pass

    logger.info(f"Destination: {destination_name}")

    # 2. Locate all Review Cards
    cards_locator = _find_locator(page, selectors.REVIEW_CARD)
    if cards_locator is None:
        logger.warning("No review cards found in the DOM.")
        return []

    total_cards = cards_locator.count()
    logger.info(f"Found {total_cards} review cards in viewport.")

    unique_cards_count = 0
    text_reviews_count = 0
    star_only_reviews_count = 0

    # 3. Iterate and parse each card
    for i in range(total_cards):
        card = cards_locator.nth(i)
        try:
            # Parse Review ID
            review_id = ""
            id_locator = _find_locator(card, selectors.REVIEW_ID)
            if id_locator:
                review_id = id_locator.first.get_attribute("data-review-id") or ""

            # De-duplicate reviews using review_id
            if review_id:
                if review_id in seen_review:
                    continue
                seen_review.add(review_id)

            # Count unique review cards
            unique_cards_count += 1

            # Parse Author
            author = "Anonim"
            author_locator = _find_locator(card, selectors.REVIEW_AUTHOR)
            if author_locator:
                author = author_locator.first.inner_text().strip()
            if not author:
                author = "Anonim"

            # Parse Rating (using keyword validation to avoid logo/badge aria-labels)
            rating = None
            for selector in selectors.REVIEW_RATING:
                try:
                    rating_locator = card.locator(selector)
                    count = rating_locator.count()
                    for idx in range(count):
                        elem = rating_locator.nth(idx)
                        aria = elem.get_attribute("aria-label") or ""
                        if any(k in aria.lower() for k in ["bintang", "star", "rating", "skor", "points"]):
                            match = re.search(r"(\d+(?:[.,]\d+)?)", aria)
                            if match:
                                rating = float(match.group(1).replace(",", "."))
                                break
                        else:
                            # Fallback to inner_text if no aria-label match (e.g. TripAdvisor "/5" text rating)
                            text = (elem.inner_text() or "").strip()
                            if "/" in text:
                                parts = text.split("/")
                                try:
                                    rating = float(parts[0].strip().replace(",", "."))
                                    break
                                except ValueError:
                                    pass
                    if rating is not None:
                        break
                except Exception:
                    pass

            # Parse Review Date
            review_date = "-"
            date_locator = _find_locator(card, selectors.REVIEW_DATE)
            if date_locator:
                review_date = re.sub(r"\s+", " ", date_locator.first.inner_text().strip())

            # Parse Review Text
            review_text = ""
            text_locator = _find_locator(card, selectors.REVIEW_TEXT)
            if text_locator:
                review_text = text_locator.first.inner_text().strip()

            # Check if review has text
            if not review_text:
                star_only_reviews_count += 1
                has_text = "FALSE"
                cleaned_text = ""
            else:
                text_reviews_count += 1
                has_text = "TRUE"
                cleaned_text = clean_review_text(review_text)

            reviews.append({
                "destination_name": destination_name,
                "author": author,
                "rating": rating,
                "review_date": review_date,
                "review_text": cleaned_text,
                "scraped_at": datetime.now(timezone.utc).isoformat(),
                "review_id": review_id,
                "has_text": has_text
            })
            logger.info(f"Parsed review {len(reviews)}: {author} ({rating} stars)")

        except Exception as e:
            logger.warning(f"Failed parsing review card #{i}: {e}")

    logger.info(f"Total review cards      : {total_cards}")
    logger.info(f"Review dengan teks      : {text_reviews_count}")
    logger.info(f"Review tanpa teks       : {star_only_reviews_count}")
    logger.info(f"Total disimpan          : {len(reviews)}")
    return reviews
