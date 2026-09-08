import logging
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from config import settings

logger = logging.getLogger("pipeline")


def evaluate_imbalance_experiments() -> pd.DataFrame:
    """
    Membaca hasil eksperimen 5 skenario handling imbalance dari 
    data/final/imbalance_experiment_comparison.csv, mencetak ringkasan evaluasi,
    dan menghasilkan grafik komparasi visual (imbalance_experiments_comparison.png).
    """
    csv_path = settings.FINAL_DATA_DIR / "imbalance_experiment_comparison.csv"
    
    if not csv_path.exists():
        logger.warning(
            f"File komparasi {csv_path} belum ditemukan. "
            "Pastikan fungsi run_imbalance_experiments() di sentiment/svm_model.py sudah dijalankan."
        )
        return pd.DataFrame()

    df_exp = pd.read_csv(csv_path)
    logger.info("=== REKAP KOMPARASI 5 SKENARIO PENANGANAN IMBALANCE ===")
    for idx, row in df_exp.iterrows():
        logger.info(
            f"  {row['Skenario Eksperimen']:<30} | Acc: {row['Accuracy']:<7} | "
            f"Bal Acc: {row['Balanced Acc']:<7} | Macro F1: {row['Macro F1-Score']}"
        )

    # Visualisasi Barplot Komparasi Macro F1 & Balanced Accuracy
    try:
        fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
        
        # Bersihkan string persentase untuk plotting jika berbentuk "88.50%"
        df_plot = df_exp.copy()
        df_plot["Bal_Acc_Float"] = df_plot["Balanced Acc"].str.rstrip('%').astype(float) / 100.0
        df_plot["Macro_F1_Float"] = df_plot["Macro F1-Score"].astype(float)
        
        x = np.arange(len(df_plot))
        width = 0.35

        rects1 = ax.bar(x - width/2, df_plot["Macro_F1_Float"], width, label='Macro F1-Score', color='#1f77b4')
        rects2 = ax.bar(x + width/2, df_plot["Bal_Acc_Float"], width, label='Balanced Accuracy', color='#2ca02c')

        ax.set_ylabel('Skor Evaluation', fontsize=10, fontweight='bold')
        ax.set_title('Komparasi Performa 5 Skenario Penanganan Class Imbalance', fontsize=12, fontweight='bold', pad=12)
        ax.set_xticks(x)
        ax.set_xticklabels([s.split('. ')[1] for s in df_plot["Skenario Eksperimen"]], rotation=15, ha='right', fontsize=9)
        ax.set_ylim(0, 1.0)
        ax.legend(loc='upper left')
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        # Annotate bar values
        for rect in rects1:
            h = rect.get_height()
            ax.annotate(f'{h:.3f}', xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8, fontweight='bold')
            
        for rect in rects2:
            h = rect.get_height()
            ax.annotate(f'{h:.2f}', xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)

        plt.tight_layout()
        img_path = settings.FINAL_DATA_DIR / "imbalance_experiments_comparison.png"
        plt.savefig(img_path, bbox_inches="tight", dpi=300)
        plt.close()
        logger.info(f"✅ Grafik komparasi eksperimen disimpan ke: {img_path}")
    except Exception as e:
        logger.error(f"Gagal membuat grafik komparasi eksperimen: {e}")

    return df_exp

def plot_all_scenarios_confusion_matrices(trained_models_dict, X_test_tfidf, y_test):
    """
    Menghasilkan grid visualisasi Confusion Matrix untuk seluruh 5 skenario.
    Disimpan ke: data/final/all_scenarios_confusion_matrices.png
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix

    fig, axes = plt.subplots(2, 3, figsize=(16, 10), dpi=300)
    axes = axes.flatten()
    class_names = ["Negative", "Neutral", "Positive"]

    for idx, (name, model) in enumerate(trained_models_dict.items()):
        y_pred = model.predict(X_test_tfidf)
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])

        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names,
            ax=axes[idx], cbar=False, annot_kws={"size": 11, "weight": "bold"}
        )
        axes[idx].set_title(f"{name}", fontsize=11, fontweight="bold")
        axes[idx].set_xlabel("Predicted Label", fontsize=9)
        axes[idx].set_ylabel("Actual Label", fontsize=9)

    # Sembunyikan subplot ke-6 yang kosong
    fig.delaxes(axes[5])

    plt.suptitle("Komparasi Confusion Matrix 5 Skenario Handling Imbalance", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()

    output_path = settings.FINAL_DATA_DIR / "all_scenarios_confusion_matrices.png"
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close()
    logger.info(f"✅ Grid Confusion Matrix 5 Skenario berhasil disimpan ke: {output_path}")

def evaluate_classifier(model, vectorizer, train_meta: dict) -> dict:
    """
    Evaluates the primary SVM model strictly on the test set.
    Also evaluates the Majority Class Baseline (fit on train, predict on test).
    Saves outputs:
    - classification_report.txt
    - model_metrics.csv
    - confusion_matrix.png
    """
    logger.info("Evaluating main model on test set...")
    
    # Otomatis jalankan dan catat evaluasi komparasi 5 skenario jika file tersedia
    evaluate_imbalance_experiments()
    
    X_test_tfidf = train_meta["X_test_tfidf"]
    y_test = train_meta["y_test"]
    y_train = train_meta["y_train"]
    
    # Run SVM Prediction (Calibrated + Threshold Moving or Standard)
    y_pred = model.predict(X_test_tfidf)
    
    # Calculate SVM Metrics
    acc = accuracy_score(y_test, y_pred)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    
    p_class, r_class, f_class, s_class = precision_recall_fscore_support(
        y_test, y_pred, labels=[0, 1, 2], zero_division=0
    )
    
    macro_p, macro_r, macro_f, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro", zero_division=0
    )
    
    weighted_p, weighted_r, weighted_f, _ = precision_recall_fscore_support(
        y_test, y_pred, average="weighted", zero_division=0
    )
    
    # Majority Class Baseline
    maj_class = int(pd.Series(y_train).value_counts().idxmax())
    y_pred_baseline = [maj_class] * len(y_test)
    
    baseline_acc = accuracy_score(y_test, y_pred_baseline)
    baseline_bal_acc = balanced_accuracy_score(y_test, y_pred_baseline)
    _, _, baseline_macro_f, _ = precision_recall_fscore_support(
        y_test, y_pred_baseline, average="macro", zero_division=0
    )
    _, _, baseline_weighted_f, _ = precision_recall_fscore_support(
        y_test, y_pred_baseline, average="weighted", zero_division=0
    )
    
    logger.info("=== EVALUATION COMPARISON ===")
    logger.info(f"  Proposed Model - Accuracy: {acc:.4f}, Balanced Acc: {bal_acc:.4f}, Macro F1: {macro_f:.4f}, Weighted F1: {weighted_f:.4f}")
    logger.info(f"  Baseline       - Accuracy: {baseline_acc:.4f}, Balanced Acc: {baseline_bal_acc:.4f}, Macro F1: {baseline_macro_f:.4f}, Weighted F1: {baseline_weighted_f:.4f}")
    
    # Save scikit-learn classification report
    lbl_names = ["Negative (0)", "Neutral (1)", "Positive (2)"]
    report_str = classification_report(y_test, y_pred, target_names=lbl_names, digits=4, zero_division=0)
    
    use_smote = train_meta.get("use_smote", False)
    use_thresholds = train_meta.get("use_threshold_moving", True)
    thresh_cfg = train_meta.get("class_thresholds", {})
    
    report_path = settings.FINAL_DATA_DIR / "classification_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=== LINEAR SVM CLASSIFICATION REPORT ===\n")
        f.write(f"Configuration: SMOTE={use_smote}, Threshold Moving={use_thresholds}\n")
        if use_thresholds:
            f.write(f"Decision Thresholds: {thresh_cfg}\n")
        f.write(f"Overall Accuracy   : {acc:.4f}\n")
        f.write(f"Balanced Accuracy  : {bal_acc:.4f}\n")
        f.write(f"Macro F1-Score     : {macro_f:.4f}\n")
        f.write(f"Weighted F1-Score  : {weighted_f:.4f}\n\n")
        f.write(report_str)
        f.write("\n\n=== MAJORITY CLASS BASELINE ===\n")
        f.write(f"Majority Class in Training Set: {maj_class} ({lbl_names[maj_class]})\n")
        f.write(f"Baseline Accuracy         : {baseline_acc:.4f}\n")
        f.write(f"Baseline Balanced Accuracy: {baseline_bal_acc:.4f}\n")
        f.write(f"Baseline Macro F1         : {baseline_macro_f:.4f}\n")
        f.write(f"Baseline Weighted F1      : {baseline_weighted_f:.4f}\n")
        
    logger.info(f"Classification report saved to {report_path}")
    
    # Save metrics in CSV format
    csv_path = settings.FINAL_DATA_DIR / "model_metrics.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["class_name", "precision", "recall", "f1_score", "support"])
        for idx, name in enumerate(lbl_names):
            writer.writerow([name, f"{p_class[idx]:.4f}", f"{r_class[idx]:.4f}", f"{f_class[idx]:.4f}", s_class[idx]])
        writer.writerow(["macro_avg", f"{macro_p:.4f}", f"{macro_r:.4f}", f"{macro_f:.4f}", len(y_test)])
        writer.writerow(["weighted_avg", f"{weighted_p:.4f}", f"{weighted_r:.4f}", f"{weighted_f:.4f}", len(y_test)])
        writer.writerow(["balanced_accuracy", f"{bal_acc:.4f}", f"{bal_acc:.4f}", f"{bal_acc:.4f}", len(y_test)])
        
    logger.info(f"Metrics table saved to {csv_path}")
    
    # Generate and Save Confusion Matrix Plot
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
    
    fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=300)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Negative", "Neutral", "Positive"])
    disp.plot(cmap=plt.cm.Blues, values_format="d", ax=ax, colorbar=True)
    
    plt.title("Confusion Matrix - Primary Linear SVM", fontsize=11, fontweight="bold", pad=12)
    plt.xlabel("Predicted Label", fontsize=10, fontweight="bold")
    plt.ylabel("Actual True Label", fontsize=10, fontweight="bold")
    plt.tight_layout()
    
    cm_path = settings.FINAL_DATA_DIR / "confusion_matrix.png"
    plt.savefig(cm_path, bbox_inches="tight", dpi=300)
    plt.close()
    logger.info(f"Confusion matrix plot saved to {cm_path}")
    
    return {
        "accuracy": float(acc),
        "balanced_accuracy": float(bal_acc),
        "macro_f1": float(macro_f),
        "weighted_f1": float(weighted_f),
        "baseline_accuracy": float(baseline_acc),
        "baseline_balanced_accuracy": float(baseline_bal_acc),
        "baseline_macro_f1": float(baseline_macro_f)
    }


def run_error_analysis(model, vectorizer, df_labeled: pd.DataFrame, train_meta: dict) -> None:
    """
    Identifies and logs misclassified reviews from the test split.
    Saves results to data/final/error_analysis.csv
    """
    logger.info("Performing Error Analysis on Test Set...")
    
    X_test_tfidf = train_meta["X_test_tfidf"]
    y_test = train_meta["y_test"]
    
    y_pred = model.predict(X_test_tfidf)
    label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
    
    # Ambil index data test secara aman langsung dari X_test train_meta
    test_size = getattr(settings, "TEST_SIZE", 0.2)
    random_state = getattr(settings, "RANDOM_STATE", 42)
    
    # Pastikan sentiment_label bersih dan bertipe integer
    df_clean = df_labeled.dropna(subset=["sentiment_label"]).copy()
    df_clean["sentiment_label"] = df_clean["sentiment_label"].astype(int)
    
    indices_train, indices_test = train_test_split(
        df_clean.index,
        test_size=test_size,
        stratify=df_clean["sentiment_label"],
        random_state=random_state
    )
    
    df_test = df_clean.loc[indices_test].copy()
    df_test["predicted_label"] = y_pred
    
    # Filter hanya data yang salah prediksi (Misclassifications)
    df_errors = df_test[df_test["sentiment_label"] != df_test["predicted_label"]].copy()
    
    df_errors["actual_sentiment"] = df_errors["sentiment_label"].map(label_map)
    df_errors["predicted_sentiment"] = df_errors["predicted_label"].map(label_map)
    
    error_report_columns = [
        "destination_name",
        "review_text",
        "rating",
        "sentiment_label",
        "predicted_label",
        "actual_sentiment",
        "predicted_sentiment"
    ]
    
    # Filter kolom yang benar-benar ada di DataFrame
    available_cols = [c for c in error_report_columns if c in df_errors.columns]
    
    output_path = settings.FINAL_DATA_DIR / "error_analysis.csv"
    settings.FINAL_DATA_DIR.mkdir(parents=True, exist_ok=True)
    df_errors[available_cols].to_csv(output_path, index=False)
    
    logger.info(f"Error Analysis completed. Found {len(df_errors)} errors out of {len(y_test)} test cases. Saved to {output_path}")