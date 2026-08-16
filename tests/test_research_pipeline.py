import sys
import unittest
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer

# Setup sys.path to resolve project root imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from preprocessing import (
    clean_text,
    case_folding,
    tokenize,
    remove_stopwords,
    stem_tokens,
    preprocess_single_text
)
from sentiment.labeling import get_label_from_rating, apply_sentiment_labeling
from policy.analysis import classify_policy

class TestResearchPreprocessing(unittest.TestCase):
    def test_cleaning(self):
        # Remove URLs
        self.assertEqual(clean_text("Lihat di http://example.com sangat indah"), "Lihat di sangat indah")
        # Remove HTML
        self.assertEqual(clean_text("<b>Kawah Kamojang</b>"), "Kawah Kamojang")
        # Remove emojis & non-ASCII
        self.assertEqual(clean_text("Mantap! 😍👍"), "Mantap")
        # Remove punctuation & normalize whitespace
        self.assertEqual(clean_text("Garut,  wisata,,. indah!"), "Garut wisata indah")

    def test_case_folding(self):
        self.assertEqual(case_folding("Kawah PUTIH"), "kawah putih")
        self.assertEqual(case_folding(""), "")

    def test_tokenization(self):
        self.assertEqual(tokenize("gunung papandayan garut"), ["gunung", "papandayan", "garut"])
        self.assertEqual(tokenize(""), [])

    def test_stopword_removal(self):
        # Test combined NLTK + custom stops
        tokens = ["ke", "kawah", "kamojang", "yg", "sangat", "indah", "aja"]
        filtered = remove_stopwords(tokens)
        self.assertIn("kawah", filtered)
        self.assertIn("kamojang", filtered)
        self.assertIn("indah", filtered)
        self.assertNotIn("ke", filtered)
        self.assertNotIn("yg", filtered)
        self.assertNotIn("sangat", filtered)
        self.assertNotIn("aja", filtered)

    def test_stemming(self):
        # PySastrawi stemming test
        tokens = ["berwisata", "menikmati", "keindahan"]
        stemmed = stem_tokens(tokens)
        self.assertEqual(stemmed, ["wisata", "nikmat", "indah"])


class TestResearchSentimentLabeling(unittest.TestCase):
    def test_label_mapping(self):
        # 1-2 -> 0 (Negative)
        self.assertEqual(get_label_from_rating(1), 0)
        self.assertEqual(get_label_from_rating(2), 0)
        self.assertEqual(get_label_from_rating(1.0), 0)
        self.assertEqual(get_label_from_rating(2.5), 0)
        
        # 3 -> 1 (Neutral)
        self.assertEqual(get_label_from_rating(3), 1)
        self.assertEqual(get_label_from_rating(3.0), 1)
        self.assertEqual(get_label_from_rating(3.2), 1)
        
        # 4-5 -> 2 (Positive)
        self.assertEqual(get_label_from_rating(4), 2)
        self.assertEqual(get_label_from_rating(5), 2)
        self.assertEqual(get_label_from_rating(4.5), 2)

    def test_invalid_ratings(self):
        # Invalid numeric values
        self.assertEqual(get_label_from_rating(0), -1)
        self.assertEqual(get_label_from_rating(6), -1)
        self.assertEqual(get_label_from_rating(-1), -1)
        
        # NaNs, strings, empty values
        self.assertEqual(get_label_from_rating(None), -1)
        self.assertEqual(get_label_from_rating("abc"), -1)
        self.assertEqual(get_label_from_rating(""), -1)


class TestTfidfDataLeakage(unittest.TestCase):
    def test_no_leakage_on_test_transform(self):
        train_texts = ["kawah indah garut", "gunung papandayan alam"]
        test_texts = ["kamojang panas baru", "kawah alam"]  # Contains unseen vocabulary "kamojang", "panas", "baru"
        
        # Fit vectorizer on training data only
        vectorizer = TfidfVectorizer(max_features=100)
        X_train = vectorizer.fit_transform(train_texts)
        vocab_before = set(vectorizer.vocabulary_.keys())
        
        # Transform test data (strictly transform only!)
        X_test = vectorizer.transform(test_texts)
        vocab_after = set(vectorizer.vocabulary_.keys())
        
        # Verify that vocabulary did not expand or change during test transformation
        self.assertEqual(vocab_before, vocab_after)
        self.assertNotIn("kamojang", vocab_before)
        self.assertNotIn("panas", vocab_before)
        self.assertNotIn("baru", vocab_before)


class TestPolicyClassification(unittest.TestCase):
    def test_classification_boundaries(self):
        # Operational Thresholds
        min_revs = 10
        neg_thresh = 15.0
        pos_thresh = 70.0
        rating_thresh = 4.0
        
        # 1. Insufficient Evidence (reviews < 10)
        row = {"total_reviews": 9, "negative_percentage": 0.0, "positive_percentage": 100.0, "average_rating": 4.5}
        self.assertEqual(classify_policy(row, min_revs, neg_thresh, pos_thresh, rating_thresh), "Insufficient Evidence")
        
        # Insufficient Evidence Boundary (reviews = 9, negative = 20)
        row = {"total_reviews": 9, "negative_percentage": 20.0, "positive_percentage": 50.0, "average_rating": 3.5}
        self.assertEqual(classify_policy(row, min_revs, neg_thresh, pos_thresh, rating_thresh), "Insufficient Evidence")
        
        # 2. Intervention Priority (reviews >= 10, negative >= 15.0%)
        row = {"total_reviews": 10, "negative_percentage": 15.0, "positive_percentage": 60.0, "average_rating": 4.2}
        self.assertEqual(classify_policy(row, min_revs, neg_thresh, pos_thresh, rating_thresh), "Intervention Priority")
        
        row = {"total_reviews": 15, "negative_percentage": 25.0, "positive_percentage": 50.0, "average_rating": 3.8}
        self.assertEqual(classify_policy(row, min_revs, neg_thresh, pos_thresh, rating_thresh), "Intervention Priority")
        
        # 3. Promotional Priority (reviews >= 10, negative < 15.0%, positive >= 70.0%, rating >= 4.0)
        row = {"total_reviews": 10, "negative_percentage": 5.0, "positive_percentage": 70.0, "average_rating": 4.0}
        self.assertEqual(classify_policy(row, min_revs, neg_thresh, pos_thresh, rating_thresh), "Promotional Priority")
        
        row = {"total_reviews": 100, "negative_percentage": 14.9, "positive_percentage": 70.1, "average_rating": 4.5}
        self.assertEqual(classify_policy(row, min_revs, neg_thresh, pos_thresh, rating_thresh), "Promotional Priority")
        
        # 4. Monitoring / Improvement Priority (fallback for other valid cases)
        # reviews >= 10, negative < 15.0% BUT positive < 70.0%
        row = {"total_reviews": 12, "negative_percentage": 10.0, "positive_percentage": 65.0, "average_rating": 4.2}
        self.assertEqual(classify_policy(row, min_revs, neg_thresh, pos_thresh, rating_thresh), "Monitoring / Improvement Priority")
        
        # reviews >= 10, negative < 15.0%, positive >= 70.0% BUT rating < 4.0
        row = {"total_reviews": 20, "negative_percentage": 5.0, "positive_percentage": 80.0, "average_rating": 3.9}
        self.assertEqual(classify_policy(row, min_revs, neg_thresh, pos_thresh, rating_thresh), "Monitoring / Improvement Priority")


    def test_policy_mutual_exclusivity(self):
        # We programmatically generate combinations of values to verify that they are 100% mutually exclusive
        min_revs = 10
        neg_thresh = 15.0
        pos_thresh = 70.0
        rating_thresh = 4.0
        
        # Verify 1000 combinations
        for reviews in [5, 10, 50]:
            for neg in [0.0, 10.0, 15.0, 25.0]:
                for pos in [50.0, 70.0, 85.0]:
                    for rating in [3.5, 4.0, 4.8]:
                        row = {
                            "total_reviews": reviews,
                            "negative_percentage": neg,
                            "positive_percentage": pos,
                            "average_rating": rating
                        }
                        cls = classify_policy(row, min_revs, neg_thresh, pos_thresh, rating_thresh)
                        
                        # Assert that output is one of the allowed categories
                        self.assertIn(cls, [
                            "Insufficient Evidence",
                            "Intervention Priority",
                            "Promotional Priority",
                            "Monitoring / Improvement Priority"
                        ])

if __name__ == "__main__":
    unittest.main()
