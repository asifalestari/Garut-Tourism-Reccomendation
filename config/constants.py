# Google Maps URLs
GOOGLE_MAPS_BASE_URL = "https://www.google.com/maps"

# Tourism Categories in Garut Regency (for normalization and Content-Based Filtering)
TOURISM_CATEGORIES = [
    "Wisata Alam",
    "Pemandian Air Panas",
    "Wisata Budaya & Sejarah",
    "Wisata Religi",
    "Taman Rekreasi",
    "Wisata Kuliner",
    "Lainnya"
]

# Additional Custom Indonesian Stopwords frequently found in Google Maps reviews
CUSTOM_INDONESIAN_STOPWORDS = {
    # Slang & abbreviations
    "yg", "dgn", "dng", "aja", "saja", "ga", "gak", "gk", "ndak", "tdk", "bgt", 
    "banget", "ya", "yah", "loh", "kok", "sih", "tuh", "deh", "doang", "dong",
    "kalo", "kalau", "biar", "buat", "utk", "untuk", "dr", "dari", "ke", "di",
    "sy", "saya", "aku", "lu", "loe", "gue", "gw", "kamu", "km", "kita", "kami",
    # Common reviews words with low sentiment meaning
    "maps", "google", "garut", "wisata", "tempat", "lokasi", "sini", "situ",
    # Other filler words
    "ada", "adalah", "dan", "atau", "yang", "dengan", "oleh", "pada", "tentang"
}
