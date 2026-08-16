import logging
import csv
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend to avoid window GUI issues
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from config import settings

logger = logging.getLogger("pipeline")

def evaluate_classifier(model, vectorizer, train_meta: dict) -> dict:
    """
    Evaluates the SVM model strictly on the test set.
    Also evaluates the Majority Class Baseline (fit on train, predict on test).
    Saves outputs:
    - classification_report.txt
    - model_metrics.csv
    - confusion_matrix.png
    """
    logger.info("Evaluating model on test set...")
    
    X_test_tfidf = train_meta["X_test_tfidf"]
    y_test = train_meta["y_test"]
    y_train = train_meta["y_train"]
    
    # Run SVM Prediction
    y_pred = model.predict(X_test_tfidf)
    
    # Calculate SVM Metrics
    acc = accuracy_score(y_test, y_pred)
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
    _, _, baseline_macro_f, _ = precision_recall_fscore_support(
        y_test, y_pred_baseline, average="macro", zero_division=0
    )
    _, _, baseline_weighted_f, _ = precision_recall_fscore_support(
        y_test, y_pred_baseline, average="weighted", zero_division=0
    )
    
    logger.info("=== EVALUATION COMPARISON ===")
    logger.info(f"  Linear SVM  - Accuracy: {acc:.4f}, Macro F1: {macro_f:.4f}, Weighted F1: {weighted_f:.4f}")
    logger.info(f"  Baseline    - Accuracy: {baseline_acc:.4f}, Macro F1: {baseline_macro_f:.4f}, Weighted F1: {baseline_weighted_f:.4f}")
    
    # Save scikit-learn classification report
    lbl_names = ["Negative (0)", "Neutral (1)", "Positive (2)"]
    report_str = classification_report(y_test, y_pred, target_names=lbl_names, zero_division=0)
    
    report_path = settings.FINAL_DATA_DIR / "classification_report.txt"
    with open(report_path, "w") as f:
        f.write("=== LINEAR SVM CLASSIFICATION REPORT ===\n")
        f.write(report_str)
        f.write("\n\n=== MAJORITY CLASS BASELINE ===\n")
        f.write(f"Majority Class in Training Set: {maj_class} ({lbl_names[maj_class]})\n")
        f.write(f"Baseline Accuracy: {baseline_acc:.4f}\n")
        f.write(f"Baseline Macro F1: {baseline_macro_f:.4f}\n")
        f.write(f"Baseline Weighted F1: {baseline_weighted_f:.4f}\n")
        
    logger.info(f"Classification report saved to {report_path}")
    
    # Save metrics in CSV format
    csv_path = settings.FINAL_DATA_DIR / "model_metrics.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class_name", "precision", "recall", "f1_score", "support"])
        for idx, name in enumerate(lbl_names):
            writer.writerow([name, f"{p_class[idx]:.4f}", f"{r_class[idx]:.4f}", f"{f_class[idx]:.4f}", s_class[idx]])
        writer.writerow(["macro_avg", f"{macro_p:.4f}", f"{macro_r:.4f}", f"{macro_f:.4f}", len(y_test)])
        writer.writerow(["weighted_avg", f"{weighted_p:.4f}", f"{weighted_r:.4f}", f"{weighted_f:.4f}", len(y_test)])
        
    logger.info(f"Metrics table saved to {csv_path}")
    
    # Generate and Save Confusion Matrix Plot
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Negative", "Neutral", "Positive"])
    
    plt.figure(figsize=(6, 6))
    disp.plot(cmap=plt.cm.Blues, values_format="d")
    plt.title("Confusion Matrix - Linear SVM")
    
    cm_path = settings.FINAL_DATA_DIR / "confusion_matrix.png"
    plt.savefig(cm_path, bbox_inches="tight", dpi=300)
    plt.close()
    logger.info(f"Confusion matrix plot saved to {cm_path}")
    
    return {
        "accuracy": float(acc),
        "macro_f1": float(macro_f),
        "weighted_f1": float(weighted_f),
        "baseline_accuracy": float(baseline_acc),
        "baseline_macro_f1": float(baseline_macro_f)
    }

def run_error_analysis(model, vectorizer, df_labeled: pd.DataFrame, train_meta: dict) -> None:
    """
    Identifies and logs misclassified reviews from the test split.
    Saves results to data/final/error_analysis.csv
    """
    logger.info("Performing Error Analysis on Test Set...")
    
    # We reconstruct the test set dataframe split to match labels with review text
    # Since train_test_split preserves indices, we can locate X_test in df_labeled
    X_test = train_meta["X_test"]
    y_test = train_meta["y_test"]
    X_test_tfidf = train_meta["X_test_tfidf"]
    
    y_pred = model.predict(X_test_tfidf)
    
    # Map index to review details
    errors = []
    label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
    
    # Let's map X_test back to original indices in df_labeled
    # Since random_state=42 is used, we can split indices as well
    indices_train, indices_test = train_test_split(
        df_labeled.index.tolist(),
        test_size=getattr(settings, "TEST_SIZE", 0.2),
        stratify=df_labeled["sentiment_label"].astype(int).tolist(),
        random_state=getattr(settings, "RANDOM_STATE", 42)
    )
    
    df_test = df_labeled.loc[indices_test].copy()
    df_test["predicted_label"] = y_pred
    
    # Select only misclassifications
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
    
    output_path = settings.FINAL_DATA_DIR / "error_analysis.csv"
    df_errors[error_report_columns].to_csv(output_path, index=False)
    
    logger.info(f"Error Analysis completed. Found {len(df_errors)} errors out of {len(y_test)} test cases. Saved to {output_path}")
