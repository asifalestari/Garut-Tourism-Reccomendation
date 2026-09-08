import logging
from typing import Tuple, Dict, Any
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support
)

from config import settings
from feature_extraction.tfidf import fit_tfidf, transform_tfidf
from feature_extraction.vectorizer import save_vectorizer

logger = logging.getLogger("pipeline")


def run_imbalance_experiments(df: pd.DataFrame) -> pd.DataFrame:
    """
    Menjalankan 5 Skenario Eksperimen Penanganan Class Imbalance secara terisolasi (Bebas Data Leakage):
    1. Baseline (No Handling)
    2. Class Weight Balanced (Algorithmic)
    3. Random Oversampling / ROS (Data-level)
    4. Random Undersampling / RUS (Data-level)
    5. SMOTE (Data-level Synthetic)

    Menyimpan rekap komparasi ke: data/final/imbalance_experiment_comparison.csv
    """
    logger.info("Memulai Eksekusi 5 Skenario Eksperimen Handling Imbalance...")

    if "cleaned_text" not in df.columns or "sentiment_label" not in df.columns:
        raise ValueError("DataFrame wajib memiliki kolom 'cleaned_text' dan 'sentiment_label'.")

    X = df["cleaned_text"].astype(str).tolist()
    y = df["sentiment_label"].astype(int).tolist()

    test_size = getattr(settings, "TEST_SIZE", 0.2)
    random_state = getattr(settings, "RANDOM_STATE", 42)

    # 1. Stratified Split 80:20 (Bebas Leakage)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )

    # 2. Fit TF-IDF Vectorizer HANYA pada Training Set
    vectorizer = fit_tfidf(X_train)
    X_train_tfidf = transform_tfidf(X_train, vectorizer)
    X_test_tfidf = transform_tfidf(X_test, vectorizer)

    from imblearn.over_sampling import RandomOverSampler, SMOTE
    from imblearn.under_sampling import RandomUnderSampler

    # 3. Definisi 5 Skenario Eksperimen
    scenarios = {
        "1. Baseline (No Handling)": {"resampler": None, "class_weight": None},
        "2. Class Weight Balanced": {"resampler": None, "class_weight": "balanced"},
        "3. Random Oversampling (ROS)": {"resampler": RandomOverSampler(random_state=random_state), "class_weight": None},
        "4. Random Undersampling (RUS)": {"resampler": RandomUnderSampler(random_state=random_state), "class_weight": None},
        "5. SMOTE": {"resampler": SMOTE(random_state=random_state, k_neighbors=5), "class_weight": None}
    }

    c_param = getattr(settings, "SVM_C", 0.2)
    results = []
    trained_models = {}

    for name, cfg in scenarios.items():
        logger.info(f"Menjalankan Eksperimen: {name}...")

        if cfg["resampler"] is not None:
            X_tr_fit, y_tr_fit = cfg["resampler"].fit_resample(X_train_tfidf, y_train)
        else:
            X_tr_fit, y_tr_fit = X_train_tfidf, y_train

        model = LinearSVC(
            C=c_param,
            class_weight=cfg["class_weight"],
            random_state=random_state,
            dual=False
        )
        model.fit(X_tr_fit, y_tr_fit)
        trained_models[name] = model

        y_pred = model.predict(X_test_tfidf)

        acc = accuracy_score(y_test, y_pred)
        bal_acc = balanced_accuracy_score(y_test, y_pred)
        p_cls, r_cls, f_cls, _ = precision_recall_fscore_support(y_test, y_pred, labels=[0, 1, 2], zero_division=0)
        macro_p, macro_r, macro_f, _ = precision_recall_fscore_support(y_test, y_pred, average="macro", zero_division=0)

        results.append({
            "Skenario Eksperimen": name,
            "Accuracy": f"{acc*100:.2f}%",
            "Balanced Acc": f"{bal_acc*100:.2f}%",
            "Neg F1 (0)": f"{f_cls[0]:.4f}",
            "Neu F1 (1)": f"{f_cls[1]:.4f}",
            "Pos F1 (2)": f"{f_cls[2]:.4f}",
            "Macro Precision": f"{macro_p:.4f}",
            "Macro Recall": f"{macro_r:.4f}",
            "Macro F1-Score": f"{macro_f:.4f}",
            "raw_macro_f1": macro_f
        })

    df_results = pd.DataFrame(results)

    output_path = settings.FINAL_DATA_DIR / "imbalance_experiment_comparison.csv"
    settings.FINAL_DATA_DIR.mkdir(parents=True, exist_ok=True)
    df_results.drop(columns=["raw_macro_f1"]).to_csv(output_path, index=False)
    logger.info(f"✅ Tabel komparasi 5 skenario berhasil disimpan ke: {output_path}")

    # Generate Grid Confusion Matrix untuk 5 Skenario
    from sentiment.evaluation import plot_all_scenarios_confusion_matrices
    plot_all_scenarios_confusion_matrices(trained_models, X_test_tfidf, y_test)

    return df_results


def train_svm_classifier(df: pd.DataFrame) -> Tuple[Any, Any, Dict[str, Any]]:
    """
    Melatih Linear SVM utama berbasis konfigurasi dari settings.py.
    Murni tanpa CalibratedClassifierCV atau Threshold Moving tambahan,
    sehingga Confusion Matrix 100% konsisten antara Bab 4 dan Dashboard.
    """
    logger.info("Initializing train/test split and modeling process...")
    
    X = df["cleaned_text"].astype(str).tolist()
    y = df["sentiment_label"].astype(int).tolist()
    
    # Hitung statistik distribusi kelas
    dist_overall = pd.Series(y).value_counts().sort_index()
    
    test_size = getattr(settings, "TEST_SIZE", 0.2)
    random_state = getattr(settings, "RANDOM_STATE", 42)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    
    dist_train = pd.Series(y_train).value_counts().sort_index()
    dist_test = pd.Series(y_test).value_counts().sort_index()
    
    # TF-IDF Feature Extraction
    vectorizer = fit_tfidf(X_train)
    X_train_tfidf = transform_tfidf(X_train, vectorizer)
    X_test_tfidf = transform_tfidf(X_test, vectorizer)
    
    # Melatih Linear SVM
    c_param = getattr(settings, "SVM_C", 0.2)
    class_weight = getattr(settings, "SVM_CLASS_WEIGHT", None)  # Dinamis membaca dari settings.py
    
    logger.info(f"Training Primary Linear SVM (C={c_param}, class_weight={class_weight})...")
    primary_svm = LinearSVC(
        C=c_param,
        class_weight=class_weight,
        random_state=random_state,
        dual=False
    )
    primary_svm.fit(X_train_tfidf, y_train)
    logger.info("Primary Linear SVM model training complete.")
    
    # Simpan Model dan Vectorizer
    model_path = settings.MODELS_DIR / "svm_model.joblib"
    settings.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(primary_svm, model_path)
    logger.info(f"Model SVM disimpan ke {model_path}")
    
    save_vectorizer(vectorizer)
    
    # Pack training metadata secara lengkap untuk main.py
    train_meta = {
        "dataset_size": len(y),
        "train_size": len(y_train),
        "test_size": len(y_test),
        "class_weights": class_weight,
        "use_smote": getattr(settings, "USE_SMOTE", False),
        "use_threshold_moving": getattr(settings, "USE_THRESHOLD_MOVING", False),
        "class_thresholds": getattr(settings, "CLASS_PROB_THRESHOLDS", {0: 0.333, 1: 0.333, 2: 0.333}),
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "X_train_tfidf": X_train_tfidf,
        "X_test_tfidf": X_test_tfidf,
        "class_distribution": {
            "Negative": int(dist_overall.get(0, 0)),
            "Neutral": int(dist_overall.get(1, 0)),
            "Positive": int(dist_overall.get(2, 0))
        },
        "train_class_distribution": {
            "Negative": int(dist_train.get(0, 0)),
            "Neutral": int(dist_train.get(1, 0)),
            "Positive": int(dist_train.get(2, 0))
        },
        "test_class_distribution": {
            "Negative": int(dist_test.get(0, 0)),
            "Neutral": int(dist_test.get(1, 0)),
            "Positive": int(dist_test.get(2, 0))
        }
    }
    
    return primary_svm, vectorizer, train_meta