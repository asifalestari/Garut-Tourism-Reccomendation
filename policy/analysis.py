import logging
import json
import numpy as np
import pandas as pd
from config import settings

logger = logging.getLogger("pipeline")

def classify_policy(row, min_reviews: int, neg_thresh: float, pos_thresh: float, rating_thresh: float) -> str:
    """
    Classifies a destination into one of four mutually exclusive policy categories
    using a strict if-elif-else hierarchy.
    """
    # 1. Insufficient Evidence
    if row["total_reviews"] < min_reviews:
        return "Insufficient Evidence"
    # 2. Intervention Priority
    elif row["negative_percentage"] >= neg_thresh:
        return "Intervention Priority"
    # 3. Promotional Priority
    elif (row["positive_percentage"] >= pos_thresh and 
          row["average_rating"] >= rating_thresh):
        return "Promotional Priority"
    # 4. Monitoring / Improvement Priority (Fallback)
    else:
        return "Monitoring / Improvement Priority"

def collapse_dests(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapses duplicate destination metadata names in destinations.csv deterministically:
    - average_rating (rating): mean
    - category: joined string of unique categories (preserves all Place ID metadata)
    - other columns: first occurrence
    """
    agg_dict = {}
    for col in df.columns:
        if col == "name":
            continue
        elif col == "rating":
            agg_dict[col] = "mean"
        elif col == "category":
            agg_dict[col] = lambda x: ", ".join(sorted(list(set(str(v) for v in x if pd.notna(v) and str(v).strip() != ""))))
        else:
            agg_dict[col] = "first"
            
    df_collapsed = df.groupby("name", as_index=False).agg(agg_dict)
    
    # Map empty category strings back to NaN
    df_collapsed["category"] = df_collapsed["category"].apply(lambda x: x if x != "" else np.nan)
    return df_collapsed

def generate_policy_recommendations(df_predicted: pd.DataFrame, eval_metrics: dict = None) -> None:
    """
    Runs destination and category aggregations, validates mutual exclusivity of classifications,
    and exports:
    - data/final/destination_sentiment_summary.csv
    - data/final/category_sentiment_summary.csv
    - data/final/policy_recommendations.md
    """
    logger.info("Starting policy recommendation analysis...")
    
    dest_path = settings.RAW_DATA_DIR / "destinations.csv"
    if not dest_path.exists():
        raise FileNotFoundError(f"Destinations metadata file not found at: {dest_path}")
        
    df_dests = pd.read_csv(dest_path)
    df_dests_collapsed = collapse_dests(df_dests)
    
    # 1. Destination-Level Aggregation
    # Group predictions by destination_name
    dest_groups = df_predicted.groupby("destination_name")
    
    agg_rows = []
    for name, group in dest_groups:
        total = len(group)
        pos = int((group["predicted_label"] == 2).sum())
        neu = int((group["predicted_label"] == 1).sum())
        neg = int((group["predicted_label"] == 0).sum())
        
        pos_pct = round((pos / total) * 100, 2) if total > 0 else 0.0
        neu_pct = round((neu / total) * 100, 2) if total > 0 else 0.0
        neg_pct = round((neg / total) * 100, 2) if total > 0 else 0.0
        
        # Verify that percentages sum to ~100% (within floating point precision)
        total_pct = pos_pct + neu_pct + neg_pct
        assert abs(total_pct - 100.0) <= 0.5, f"Percentage inconsistency for {name}: {total_pct}%"
        
        agg_rows.append({
            "destination_name": name,
            "total_reviews": total,
            "positive_count": pos,
            "neutral_count": neu,
            "negative_count": neg,
            "positive_percentage": pos_pct,
            "neutral_percentage": neu_pct,
            "negative_percentage": neg_pct
        })
        
    df_dest_agg = pd.DataFrame(agg_rows)
    
    # Join with deduplicated collapsed destinations metadata to avoid review inflation
    df_dest_agg = pd.merge(df_dest_agg, df_dests_collapsed, left_on="destination_name", right_on="name", how="left")
    
    # Clean up joined columns
    df_dest_agg["average_rating"] = df_dest_agg["rating"].fillna(0.0)
    df_dest_agg = df_dest_agg.drop(columns=["name", "rating", "url", "status", "scraped_at", "address"], errors="ignore")
    
    # 2. Run Mutually Exclusive Policy Classification
    min_reviews = getattr(settings, "MIN_REVIEWS", 10)
    neg_thresh = getattr(settings, "NEGATIVE_THRESHOLD", 15.0)
    pos_thresh = getattr(settings, "POSITIVE_THRESHOLD", 70.0)
    rating_thresh = getattr(settings, "AVERAGE_RATING_THRESHOLD", 4.0)
    
    df_dest_agg["policy_class"] = df_dest_agg.apply(
        lambda row: classify_policy(row, min_reviews, neg_thresh, pos_thresh, rating_thresh),
        axis=1
    )
    
    # 3. Validation and Programmatic Assertions
    allowed_classes = {
        "Insufficient Evidence",
        "Intervention Priority",
        "Promotional Priority",
        "Monitoring / Improvement Priority"
    }
    
    policy_classes = df_dest_agg["policy_class"].tolist()
    
    # Assertion Checks
    assert len(df_dest_agg) == len(policy_classes), "Classification row count mismatch."
    assert df_dest_agg["policy_class"].isnull().sum() == 0, "Found null values in policy_class."
    assert set(policy_classes).issubset(allowed_classes), f"Found unauthorized policy classes: {set(policy_classes)}"
    
    # Mutual Exclusivity Assertion (checking intersection of index sets is empty)
    s_ins = set(df_dest_agg[df_dest_agg["policy_class"] == "Insufficient Evidence"]["destination_name"])
    s_int = set(df_dest_agg[df_dest_agg["policy_class"] == "Intervention Priority"]["destination_name"])
    s_pro = set(df_dest_agg[df_dest_agg["policy_class"] == "Promotional Priority"]["destination_name"])
    s_mon = set(df_dest_agg[df_dest_agg["policy_class"] == "Monitoring / Improvement Priority"]["destination_name"])
    
    # Verify no intersection
    assert s_ins.isdisjoint(s_int), "Intersection found between Insufficient and Intervention."
    assert s_ins.isdisjoint(s_pro), "Intersection found between Insufficient and Promotional."
    assert s_ins.isdisjoint(s_mon), "Intersection found between Insufficient and Monitoring."
    assert s_int.isdisjoint(s_pro), "Intersection found between Intervention and Promotional."
    assert s_int.isdisjoint(s_mon), "Intersection found between Intervention and Monitoring."
    assert s_pro.isdisjoint(s_mon), "Intersection found between Promotional and Monitoring."
    
    # Total unique destinations matches sum of subsets
    total_unique_dests = len(df_dest_agg["destination_name"].unique())
    sum_subsets = len(s_ins) + len(s_int) + len(s_pro) + len(s_mon)
    assert total_unique_dests == sum_subsets, f"Total counts mismatch: {total_unique_dests} vs {sum_subsets}"
    
    # Save Destination Summary
    dest_summary_path = settings.FINAL_DATA_DIR / "destination_sentiment_summary.csv"
    df_dest_agg.to_csv(dest_summary_path, index=False)
    logger.info(f"Destination sentiment summary saved to {dest_summary_path}")
    
    # 4. Category-Level Aggregation
    # Join prediction dataset with collapsed metadata to prevent review inflation
    df_merged_reviews = pd.merge(df_predicted, df_dests_collapsed[["name", "category"]], left_on="destination_name", right_on="name", how="left")
    
    cat_groups = df_merged_reviews.groupby("category")
    cat_rows = []
    for cat_name, group in cat_groups:
        total_revs = len(group)
        unique_dests = len(group["destination_name"].unique())
        
        # Pooled distribution (pooled review percentage - Primary Metric)
        pos_count = (group["predicted_label"] == 2).sum()
        neu_count = (group["predicted_label"] == 1).sum()
        neg_count = (group["predicted_label"] == 0).sum()
        
        pooled_pos = round((pos_count / total_revs) * 100, 2) if total_revs > 0 else 0.0
        pooled_neu = round((neu_count / total_revs) * 100, 2) if total_revs > 0 else 0.0
        pooled_neg = round((neg_count / total_revs) * 100, 2) if total_revs > 0 else 0.0
        
        # Mean destination distribution (secondary diagnostic metrics)
        df_cat_dests = df_dest_agg[df_dest_agg["category"] == cat_name]
        mean_pos = round(df_cat_dests["positive_percentage"].mean(), 2) if len(df_cat_dests) > 0 else 0.0
        mean_neu = round(df_cat_dests["neutral_percentage"].mean(), 2) if len(df_cat_dests) > 0 else 0.0
        mean_neg = round(df_cat_dests["negative_percentage"].mean(), 2) if len(df_cat_dests) > 0 else 0.0
        
        # Average rating of destinations in this category
        avg_rating = round(df_cat_dests["average_rating"].mean(), 2) if len(df_cat_dests) > 0 else 0.0
        
        cat_rows.append({
            "category": cat_name,
            "total_destinations": unique_dests,
            "total_reviews": total_revs,
            "positive_percentage": pooled_pos,       # Pooled is primary positive
            "neutral_percentage": pooled_neu,         # Pooled is primary neutral
            "negative_percentage": pooled_neg,         # Pooled is primary negative
            "average_rating": avg_rating,
            "mean_dest_positive_pct": mean_pos,
            "mean_dest_neutral_pct": mean_neu,
            "mean_dest_negative_pct": mean_neg
        })
        
    df_cat_summary = pd.DataFrame(cat_rows)
    cat_summary_path = settings.FINAL_DATA_DIR / "category_sentiment_summary.csv"
    df_cat_summary.to_csv(cat_summary_path, index=False)
    logger.info(f"Category sentiment summary saved to {cat_summary_path}")
    
    # 5. Generate Laporan Kebijakan pariwisata (policy_recommendations.md)
    write_policy_report(df_predicted, df_dest_agg, df_cat_summary, s_ins, s_int, s_pro, s_mon, eval_metrics)

def write_policy_report(df_predicted, df_dest, df_cat, s_ins, s_int, s_pro, s_mon, eval_metrics: dict = None) -> None:
    """
    Writes the markdown research policy recommendations report.
    """
    report_path = settings.FINAL_DATA_DIR / "policy_recommendations.md"
    
    total_revs = len(df_predicted)
    total_dests = len(df_dest)
    
    overall_pos = (df_predicted["predicted_label"] == 2).sum()
    overall_neu = (df_predicted["predicted_label"] == 1).sum()
    overall_neg = (df_predicted["predicted_label"] == 0).sum()
    
    pos_pct = (overall_pos / total_revs) * 100
    neu_pct = (overall_neu / total_revs) * 100
    neg_pct = (overall_neg / total_revs) * 100
    
    # Load model metrics (in memory has precedence to avoid circular ordering files bug)
    accuracy = 0.0
    macro_f1 = 0.0
    if eval_metrics:
        accuracy = eval_metrics.get("accuracy", 0.0)
        macro_f1 = eval_metrics.get("macro_f1", 0.0)
    else:
        try:
            with open(settings.FINAL_DATA_DIR / "experiment_metadata.json", "r") as f:
                meta = json.load(f)
                accuracy = meta.get("overall_accuracy", 0.0)
                macro_f1 = meta.get("macro_f1", 0.0)
        except Exception:
            pass

    with open(report_path, "w") as f:
        f.write("# Analisis Sentimen Ulasan Destinasi Wisata\n")
        f.write("## Kabupaten Garut\n\n")
        
        f.write("## 1. Dataset Overview\n")
        f.write(f"- **Total Ulasan Valid Setelah Preprocessing:** {total_revs:,} ulasan\n")
        f.write(f"- **Total Destinasi Wisata yang Terdaftar:** {total_dests} destinasi\n")
        f.write("- **Sumber Data:** Google Maps Reviews (Scraped Dataset)\n\n")
        
        f.write("## 2. Distribusi Rating\n")
        f.write("Distribusi rating ulasan individu (*Individual Review Rating*) dari seluruh dataset:\n\n")
        f.write("| Rating Bintang | Jumlah Ulasan | Persentase |\n")
        f.write("| :--- | :---: | :---: |\n")
        rating_counts = df_predicted["rating"].value_counts().sort_index()
        for r, count in rating_counts.items():
            f.write(f"| {r} Bintang | {count:,} | {count/total_revs*100:.2f}% |\n")
        f.write("\n")
        
        f.write("## 3. Distribusi Sentimen\n")
        f.write("Distribusi prediksi sentimen keseluruhan ulasan pariwisata:\n")
        f.write(f"- **Positive (2):** {overall_pos:,} ulasan ({pos_pct:.2f}%)\n")
        f.write(f"- **Neutral (1):** {overall_neu:,} ulasan ({neu_pct:.2f}%)\n")
        f.write(f"- **Negative (0):** {overall_neg:,} ulasan ({neg_pct:.2f}%)\n\n")
        
        f.write("## 4. Evaluasi Model SVM\n")
        f.write("Kinerja pengklasifikasi teks Linear SVM pada Test Set (20% split):\n")
        f.write(f"- **Akurasi Model:** {accuracy*100:.2f}%\n")
        f.write(f"- **Macro F1-Score:** {macro_f1:.4f}\n")
        f.write("- *Catatan:* Perincian presisi, recall, dan confusion matrix tersimpan di berkas biner/gambar laporan.\n\n")
        
        f.write("## 5. Analisis Sentimen per Destinasi\n")
        f.write("Daftar destinasi dengan akumulasi sentimen ulasan (menampilkan destinasi pariwisata terpopuler):\n\n")
        f.write("| Nama Destinasi | Total Ulasan | Positif (%) | Netral (%) | Negatif (%) | Avg Rating |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: |\n")
        # Show top 15 destinations by review count
        df_top_dests = df_dest.sort_values(by="total_reviews", ascending=False).head(15)
        for _, row in df_top_dests.iterrows():
            f.write(f"| {row['destination_name']} | {row['total_reviews']} | {row['positive_percentage']}% | {row['neutral_percentage']}% | {row['negative_percentage']}% | {row['average_rating']} |\n")
        f.write("\n")
        
        f.write("## 6. Analisis Sentimen Berdasarkan Kategori\n")
        f.write("Agregasi distribusi sentimen berdasarkan jenis/kategori destinasi wisata di Kabupaten Garut:\n\n")
        f.write("| Kategori Wisata | Jumlah Destinasi | Total Ulasan | Positif (%) | Netral (%) | Negatif (%) | Avg Rating |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for _, row in df_cat.iterrows():
            # Using pooled review percentage as standard
            f.write(f"| {row['category']} | {row['total_destinations']} | {row['total_reviews']:,} | {row['positive_percentage']}% | {row['neutral_percentage']}% | {row['negative_percentage']}% | {row['average_rating']} |\n")
        f.write("\n")
        
        f.write("## 7. Promotional Targets\n")
        f.write("Destinasi pariwisata unggulan dengan reputasi kepuasan publik tinggi (sentimen positif dominan) yang direkomendasikan untuk promosi masif:\n\n")
        df_promo = df_dest[df_dest["policy_class"] == "Promotional Priority"].sort_values(by="positive_percentage", ascending=False)
        if len(df_promo) > 0:
            f.write("| Nama Destinasi | Total Ulasan | Positif (%) | Avg Rating |\n")
            f.write("| :--- | :---: | :---: | :---: |\n")
            for _, row in df_promo.head(10).iterrows():
                f.write(f"| {row['destination_name']} | {row['total_reviews']} | {row['positive_percentage']}% | {row['average_rating']} |\n")
        else:
            f.write("*Tidak ada destinasi yang memenuhi kriteria Promotional Target.*\n")
        f.write("\n")
        
        f.write("## 8. Monitoring / Improvement Targets\n")
        f.write("Destinasi pariwisata dengan persentase ulasan netral yang relatif tinggi atau belum menunjukkan dominasi persepsi yang kuat:\n\n")
        df_monitor = df_dest[df_dest["policy_class"] == "Monitoring / Improvement Priority"].sort_values(by="neutral_percentage", ascending=False)
        if len(df_monitor) > 0:
            f.write("| Nama Destinasi | Total Ulasan | Netral (%) | Avg Rating |\n")
            f.write("| :--- | :---: | :---: | :---: |\n")
            for _, row in df_monitor.head(10).iterrows():
                f.write(f"| {row['destination_name']} | {row['total_reviews']} | {row['neutral_percentage']}% | {row['average_rating']} |\n")
        else:
            f.write("*Tidak ada destinasi yang memenuhi kriteria Monitoring Target.*\n")
        f.write("\n")
        
        f.write("## 9. Policy Intervention Targets\n")
        f.write("Destinasi pariwisata yang menunjukkan proporsi ulasan negatif relatif tinggi, direkomendasikan untuk ditinjau langsung oleh dinas terkait:\n\n")
        df_intervene = df_dest[df_dest["policy_class"] == "Intervention Priority"].sort_values(by="negative_percentage", ascending=False)
        if len(df_intervene) > 0:
            f.write("| Nama Destinasi | Total Ulasan | Negatif (%) | Avg Rating |\n")
            f.write("| :--- | :---: | :---: | :---: |\n")
            for _, row in df_intervene.head(10).iterrows():
                f.write(f"| {row['destination_name']} | {row['total_reviews']} | {row['negative_percentage']}% | {row['average_rating']} |\n")
        else:
            f.write("*Tidak ada destinasi yang memenuhi kriteria Intervention Target.*\n")
        f.write("\n")
        
        f.write("## 10. Interpretasi dan Rekomendasi Kebijakan\n")
        f.write("Analisis interpretasi ini didasarkan pada data persepsi ulasan ulasan digital pariwisata:\n\n")
        
        # Write specific academic phrases for top 3 Intervention Priority targets if they exist
        if len(df_intervene) > 0:
            f.write("### Rekomendasi Prioritas Intervensi:\n")
            for _, row in df_intervene.head(3).iterrows():
                f.write(f"- Destinasi **{row['destination_name']}** memiliki proporsi prediksi sentimen negatif sebesar **{row['negative_percentage']}%** dari total **{row['total_reviews']}** ulasan valid yang dianalisis. Temuan ini menunjukkan adanya ketidakpuasan pengunjung yang cukup tinggi secara statistik, sehingga destinasi tersebut direkomendasikan untuk diprioritaskan dalam evaluasi lapangan lebih lanjut oleh pemangku kepentingan pariwisata Kabupaten Garut.\n")
            f.write("\n")
            
        if len(df_monitor) > 0:
            f.write("### Analisis Pemantauan (Sentimen Netral):\n")
            for _, row in df_monitor.head(3).iterrows():
                f.write(f"- Destinasi **{row['destination_name']}** menunjukkan proporsi sentimen netral sebesar **{row['neutral_percentage']}%** dari total **{row['total_reviews']}** ulasan valid. Hal ini mengindikasikan bahwa impresi atau persepsi pengunjung terhadap destinasi pariwisata tersebut belum terbentuk ke arah positif maupun negatif secara dominan, sehingga direkomendasikan untuk pemantauan berkelanjutan terkait peningkatan mutu layanan.\n")
            f.write("\n")
            
        f.write("## 11. Kesimpulan\n")
        f.write("Sistem analisis sentimen berbasis Linear SVM dan pemetaan kebijakan prioritas ini menyediakan sarana pendukung keputusan (*decision-support tool*) objektif bagi Dinas Pariwisata Kabupaten Garut untuk merencanakan alokasi promosi dan program peningkatan mutu destinasi wisata secara transparan berbasis data (*evidence-based policy*).\n")

    logger.info(f"Laporan akhir rekomendasi kebijakan pariwisata berhasil disimpan ke {report_path}")

