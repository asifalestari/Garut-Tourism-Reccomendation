# CSS Selectors for Google Maps Web Scraping
# Since Google Maps updates its classes periodically, these are modularized here.

# ==========================================
# DESTINATION DETAIL SELECTORS (Used by destination scraper)
# ==========================================
DESTINATION_DETAIL_SELECTORS = {
    "name": "h1.DUw3O, h1",
    "category": "button.DkL0fd, button[jsaction*='category']",
    "rating": "div.F7nice span[aria-hidden='true']",
    "address": "button[data-item-id='address']"
}

# Individual destination selector lists (used by review parsing for robustness)
DESTINATION_NAME = ["h1.DUw3O", "h1"]
DESTINATION_CATEGORY = ["button.DkL0fd", "button[jsaction*='category']", "button[jsaction='pane.rating.category']"]
DESTINATION_RATING = ["div.F7nice span[aria-hidden='true']", "div.F7nice"]
DESTINATION_ADDRESS = ["button[data-item-id='address']"]

# Search page feed selectors
SEARCH_INPUT = 'input[name="q"]'
SEARCH_BUTTON = 'button[aria-label="Cari"]'
RESULT_FEED = 'div[role="feed"]'
RESULT_ITEM_LINK = 'a[href*="/maps/place/"]'

# ==========================================
# REVIEW SELECTORS
# ==========================================
# Action / Tab & Scroll selectors
REVIEW_TAB_BUTTON = [
    "button[role='tab'][aria-label*='Ulasan' i]",
    "button[role='tab'][aria-label*='Reviews' i]",
    "button[aria-label^='Ulasan' i]",
    "button[aria-label^='Reviews' i]",
    "button:text('Ulasan')",
    "button:text('Reviews')"
]
REVIEW_SCROLL_CONTAINER = [
    "div.m6QErb.dS8AEf[scrollable='true']",
    "div.m6QErb.dS8AEf",
    "div.m6QErb.DxyBCb",
    "div.m6QErb[role='region']",
    "div[role='feed']",
    "div.m6QErb.yA1Bf"
]
REVIEW_CARD = [
    "div.jJc9Ad",
    "div.jftiEf",
    "div[data-review-id]"
]

# Internal review card field elements
REVIEW_ID = ["button[data-review-id]", "div[data-review-id]"]
REVIEW_AUTHOR = [
    "div.d4r55", 
    "span.TSqmq", 
    "div.TSqmq", 
    "a[href*='/contrib/']", 
    ".al6Kxe"
]
REVIEW_RATING = [
    "span.kvZ5cf",
    "div.kvZ5cf",
    "span.kvMYJc", 
    "span.kv348e", 
    "span[aria-label*='bintang' i]", 
    "span[aria-label*='star' i]", 
    "span[role='img']",
    "span.fontBodyLarge",
    "span.fzvQIb",
    "div.DU9Pgb span"
]
REVIEW_DATE = [
    "span.rsqaWe", 
    "span.rAxA2d",
    "span.xRkPPb"
]
REVIEW_TEXT = [
    "span.wi7C3c",
    "span.wiI7pd", 
    "span.wi9w8d", 
    "div.My5oee", 
    "div.MyEned span", 
    "div.MyEned"
]
MORE_BUTTON = [
    "button.w8nwRe.kyuRq",
    "button.w8oAoe",
    "button[aria-label*='Lainnya' i]",
    "button[aria-label*='More' i]",
    "button:text('Lainnya')",
    "button:text('More')"
]

# ==========================================
# POPUP SELECTORS
# ==========================================
DISMISS_BUTTONS = [
    "button[aria-label='Tolak semua']",
    "button[aria-label='Reject all']",
    "button[aria-label='Terima semua']",
    "button[aria-label='Accept all']",
    "button[aria-label='Setuju']",
    "button[aria-label='Agree']",
    "button[aria-label='Not now']",
    "button[aria-label='Nanti saja']",
    "button[aria-label='Tutup']",
    "button[aria-label='Close']"
]
