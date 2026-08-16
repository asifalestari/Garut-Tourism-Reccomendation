import os
import pandas as pd
import numpy as np

def run_audit():
    # File paths
    dest_path = "data/raw/destinations.csv"
    reviews_path = "data/raw/reviews.csv"
    audit_path = "data/analysis/review_audit.csv"
    targets_path = "data/analysis/review_repair_targets.csv"
    
    os.makedirs("data/analysis", exist_ok=True)
    
    # Check if files exist
    if not os.path.exists(dest_path):
        print(f"Error: {dest_path} not found.")
        return
    if not os.path.exists(reviews_path):
        print(f"Error: {reviews_path} not found.")
        return
        
    df_dests = pd.read_csv(dest_path)
    df_reviews = pd.read_csv(reviews_path)
    
    # Filter destinations with status success
    df_dests_success = df_dests[df_dests["status"] == "success"].copy()
    
    # Group reviews by destination_name
    grouped = df_reviews.groupby("destination_name")
    
    audit_rows = []
    
    # Track all destination names in destinations.csv
    success_dest_names = set(df_dests_success["name"].dropna().unique())
    
    for name in success_dest_names:
        if name in grouped.groups:
            group = grouped.get_group(name)
            
            review_count = len(group)
            
            # Count rating missing: rating is NaN or None or 0
            missing_rating_count = group["rating"].isnull().sum() + (group["rating"] == 0).sum()
            missing_rating_percentage = (missing_rating_count / review_count * 100.0) if review_count > 0 else 0.0
            
            # Count review text missing: check for null or empty strings
            missing_text_count = group["review_text"].isnull().sum() + (group["review_text"].astype(str).str.strip() == "").sum()
            
            # Count review id missing: check for null or empty strings
            missing_review_id_count = group["review_id"].isnull().sum() + (group["review_id"].astype(str).str.strip() == "").sum()
            
            # Unique reviews based on review_id
            unique_review_count = group["review_id"].nunique(dropna=True)
            
            # Let's assess status and reason
            status = "SAFE"
            reasons = []
            
            if review_count == 10:
                status = "REVIEW_COUNT_SUSPICIOUS"
                reasons.append("Review count is exactly 10")
            elif review_count < 10:
                reasons.append(f"Review count is low ({review_count})")
                
            if missing_rating_count > 0:
                status = "RATING_MISSING" if status == "SAFE" else status
                reasons.append(f"Has missing ratings: {missing_rating_count} ({missing_rating_percentage:.1f}%)")
                
            if unique_review_count < review_count:
                reasons.append(f"Has duplicates: {review_count - unique_review_count} duplicate reviews")
                
            if len(reasons) > 1 and (review_count == 10 or missing_rating_percentage >= 50):
                status = "REPAIR_REQUIRED"
            elif review_count == 0:
                status = "REPAIR_REQUIRED"
                reasons.append("No reviews scraped")
                
            reason_str = "; ".join(reasons) if reasons else "No anomalies detected"
            
            audit_rows.append({
                "destination_name": name,
                "review_count": review_count,
                "unique_review_count": unique_review_count,
                "missing_rating_count": missing_rating_count,
                "missing_rating_percentage": missing_rating_percentage,
                "missing_text_count": missing_text_count,
                "missing_review_id_count": missing_review_id_count,
                "status": status,
                "reason": reason_str
            })
        else:
            # 0 reviews in reviews.csv
            audit_rows.append({
                "destination_name": name,
                "review_count": 0,
                "unique_review_count": 0,
                "missing_rating_count": 0,
                "missing_rating_percentage": 0.0,
                "missing_text_count": 0,
                "missing_review_id_count": 0,
                "status": "REPAIR_REQUIRED",
                "reason": "Missing from reviews.csv entirely"
            })
            
    df_audit = pd.DataFrame(audit_rows)
    df_audit.to_csv(audit_path, index=False)
    print(f"Saved audit report to: {audit_path}")
    
    # TAHAP 2: IDENTIFIKASI DESTINASI BERMASALAH
    df_targets = []
    
    for idx, row in df_audit.iterrows():
        name = row["destination_name"]
        rev_count = row["review_count"]
        missing_rating_cnt = row["missing_rating_count"]
        missing_rating_pct = row["missing_rating_percentage"]
        
        # Match destination details
        dest_matches = df_dests_success[df_dests_success["name"] == name]
        overall_rating = np.nan
        category = "Unknown"
        if not dest_matches.empty:
            overall_rating = dest_matches.iloc[0]["rating"]
            category = str(dest_matches.iloc[0]["category"])
            
        is_suspicious = False
        priority = None
        reasons = []
        
        # Rule A: review_count == 10 (suspicious layout)
        if rev_count == 10:
            # It's a suspicious layout, but only repair if it's a hotel/resort/popular spot
            # or if it has missing ratings.
            is_hotel = "hotel" in category.lower() or "resort" in category.lower() or "penginapan" in category.lower()
            if is_hotel or missing_rating_pct > 0.0:
                is_suspicious = True
                reasons.append("Review count is exactly 10 (potential detail panel layout limit)")
                if is_hotel or missing_rating_pct >= 50.0:
                    priority = "HIGH"
                else:
                    priority = "MEDIUM"
            elif pd.notnull(overall_rating) and overall_rating > 0.0:
                # Suspicious, but not automatically high priority unless it's a known popular spot
                is_suspicious = True
                reasons.append("Review count is exactly 10 (suspicious layout limit)")
                priority = "MEDIUM"
            else:
                # review_count == 10 but no rating issues and not marked popular
                # We can classify it as LOW priority target
                is_suspicious = True
                reasons.append("Review count is exactly 10 (unconfirmed layout limit)")
                priority = "LOW"
                
        # Rule B: missing_rating_percentage is high
        if missing_rating_cnt > 0 and rev_count != 10: # (Already handled in Rule A if count is 10)
            is_suspicious = True
            reasons.append(f"Missing ratings: {missing_rating_cnt} reviews ({missing_rating_pct:.1f}%)")
            # Combination of low review count and high missing-rating percentage
            if rev_count <= 20 and missing_rating_pct >= 80.0:
                priority = "HIGH"
            elif missing_rating_pct >= 50.0:
                priority = "HIGH"
            else:
                priority = "MEDIUM"
                
        # Rule C: review_count is very small (including 0) but should have more
        if rev_count < 10 and rev_count != 10:
            if rev_count == 0:
                is_suspicious = True
                reasons.append("No reviews scraped (0 reviews)")
                # If it has an overall rating in destinations.csv, it should have reviews on Maps
                if pd.notnull(overall_rating) and overall_rating > 0.0:
                    # Check if category is popular
                    is_popular = any(k in category.lower() for k in ["hotel", "resort", "pantai", "curug", "wisata", "restoran", "pemandian", "kolam"])
                    if is_popular:
                        priority = "HIGH"
                    else:
                        priority = "MEDIUM"
                else:
                    priority = "MEDIUM"
            else:
                # 1 to 9 reviews
                if pd.notnull(overall_rating) and overall_rating > 0.0:
                    # Only mark if there's a strong indication of missing ratings or popular category
                    is_popular = any(k in category.lower() for k in ["hotel", "resort", "pantai", "curug", "wisata", "restoran", "pemandian", "kolam"])
                    if is_popular:
                        is_suspicious = True
                        reasons.append(f"Low review count ({rev_count}) but category '{category}' is popular")
                        priority = "MEDIUM"
                        
        if is_suspicious:
            if priority is None:
                priority = "LOW"
                
            df_targets.append({
                "destination_name": name,
                "current_review_count": rev_count,
                "missing_rating_count": missing_rating_cnt,
                "missing_rating_percentage": missing_rating_pct,
                "reason": " & ".join(reasons),
                "priority": priority
            })
            
    df_targets_df = pd.DataFrame(df_targets)
    # Sort targets: HIGH first, then MEDIUM, then LOW
    if not df_targets_df.empty:
        priority_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
        df_targets_df["priority_rank"] = df_targets_df["priority"].map(priority_order)
        df_targets_df = df_targets_df.sort_values("priority_rank").drop(columns=["priority_rank"])
        
    df_targets_df.to_csv(targets_path, index=False)
    print(f"Saved repair targets to: {targets_path}")
    
    # TAHAP 3: RINGKASAN KANDIDAT REPAIR
    print("\n==================================================")
    print("RINGKASAN AUDIT DAN TARGET REPAIR")
    print("==================================================")
    print(f"Total destinasi di destinations.csv (success): {len(df_dests_success)}")
    print(f"Total destinasi di reviews.csv: {df_reviews['destination_name'].nunique()}")
    print(f"Total review: {len(df_reviews)}")
    
    count_10_reviews = (df_audit["review_count"] == 10).sum()
    print(f"Jumlah destinasi dengan 10 review: {count_10_reviews}")
    
    count_missing_ratings = (df_audit["missing_rating_count"] > 0).sum()
    print(f"Jumlah destinasi dengan rating kosong: {count_missing_ratings}")
    
    print(f"Jumlah kandidat repair: {len(df_targets_df)}")
    
    if not df_targets_df.empty:
        high_pri = df_targets_df[df_targets_df["priority"] == "HIGH"]["destination_name"].tolist()
        med_pri = df_targets_df[df_targets_df["priority"] == "MEDIUM"]["destination_name"].tolist()
        low_pri = df_targets_df[df_targets_df["priority"] == "LOW"]["destination_name"].tolist()
        
        print(f"\nDaftar HIGH priority ({len(high_pri)}):")
        for h in high_pri[:15]:
            print(f" - {h}")
        if len(high_pri) > 15:
            print(f" ... and {len(high_pri) - 15} more")
            
        print(f"\nDaftar MEDIUM priority ({len(med_pri)}):")
        for m in med_pri[:15]:
            print(f" - {m}")
        if len(med_pri) > 15:
            print(f" ... and {len(med_pri) - 15} more")
            
        print(f"\nDaftar LOW priority ({len(low_pri)}):")
        for l in low_pri[:10]:
            print(f" - {l}")
    print("==================================================\n")

if __name__ == "__main__":
    run_audit()
