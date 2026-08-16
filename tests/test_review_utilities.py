import sys
from pathlib import Path
import unittest

# Setup sys.path to resolve project root imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from scraper.parser import parse_relative_date_to_months, clean_review_text

class TestReviewUtilities(unittest.TestCase):
    def test_clean_review_text(self):
        self.assertEqual(clean_review_text(""), "")
        self.assertEqual(clean_review_text("Halo   Bandung"), "Halo Bandung")
        self.assertEqual(clean_review_text("Halo\n\n\nBandung"), "Halo Bandung")
        self.assertEqual(clean_review_text("Halo\tBandung"), "Halo Bandung")
        # Emoji and Unicode test
        self.assertEqual(clean_review_text("Indah sekali! 😍\nGarut mantap! 👍"), "Indah sekali! 😍 Garut mantap! 👍")
        self.assertEqual(clean_review_text("  Spasi   di  ujung  "), "Spasi di ujung")

    def test_parse_relative_date_to_months(self):
        # Indonesian
        self.assertAlmostEqual(parse_relative_date_to_months("3 hari lalu"), 0.1)
        self.assertAlmostEqual(parse_relative_date_to_months("2 minggu lalu"), 0.5)
        self.assertAlmostEqual(parse_relative_date_to_months("sebulan lalu"), 1.0)
        self.assertAlmostEqual(parse_relative_date_to_months("5 bulan lalu"), 5.0)
        self.assertAlmostEqual(parse_relative_date_to_months("setahun lalu"), 12.0)
        self.assertAlmostEqual(parse_relative_date_to_months("2 tahun lalu"), 24.0)
        
        # English
        self.assertAlmostEqual(parse_relative_date_to_months("3 days ago"), 0.1)
        self.assertAlmostEqual(parse_relative_date_to_months("a month ago"), 1.0)
        self.assertAlmostEqual(parse_relative_date_to_months("5 months ago"), 5.0)
        self.assertAlmostEqual(parse_relative_date_to_months("a year ago"), 12.0)
        self.assertAlmostEqual(parse_relative_date_to_months("2 years ago"), 24.0)
        
        # Singulars and edge cases
        self.assertAlmostEqual(parse_relative_date_to_months("kemarin"), 0.03)
        self.assertAlmostEqual(parse_relative_date_to_months("yesterday"), 0.03)
        self.assertAlmostEqual(parse_relative_date_to_months("baru saja"), 0.0)
        self.assertAlmostEqual(parse_relative_date_to_months("-"), 0.0)

if __name__ == "__main__":
    unittest.main()
