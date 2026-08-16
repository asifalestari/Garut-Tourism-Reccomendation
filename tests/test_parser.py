import os
import sys
import unittest
from pathlib import Path

# Setup sys.path to resolve project root imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from scraper.browser import BrowserManager
from scraper.parser import parse_destination_detail
from config import settings

class TestParser(unittest.TestCase):
    def test_browser_and_parser(self):
        url = "https://www.google.com/maps/place/Kawah+Kamojang/@-7.1466667,107.7966667,15z"
        
        print(f"Menguji BrowserManager dan parser dengan URL: {url}")
        
        with BrowserManager() as manager:
            page = manager.get_page()
            
            # Navigate to the target page
            page.goto(url, wait_until="load", timeout=30000)
            
            # Wait for destination title to load
            page.wait_for_selector("h1", timeout=settings.DEFAULT_TIMEOUT)
            
            # Execute Parser Layer
            data = parse_destination_detail(page)
            
            # Print output format nicely
            print("\n=== HASIL EKSTRAKSI PARSER ===")
            for key, val in data.items():
                print(f"  {key:<12}: {val} (Type: {type(val).__name__})")
            print("===============================\n")
            
            # Assertions
            self.assertIsNotNone(data["name"])
            self.assertNotEqual(data["name"], "N/A")
            self.assertIn("Kamojang", data["name"])
            
            self.assertTrue(isinstance(data["rating"], float) or data["rating"] is None)
            if data["rating"] is not None:
                self.assertGreater(data["rating"], 0.0)
                
            self.assertNotEqual(data["address"], "N/A")

if __name__ == "__main__":
    unittest.main()
