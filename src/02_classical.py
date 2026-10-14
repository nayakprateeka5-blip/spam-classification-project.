import json
import time
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, classification_report)
from src.common import load_dataset, prepare_labels, clean_text, MODELS_DIR, REPORTS_DIR, OUTPUTS_DIR

RANDOM_STATE = 42
TEST_SIZE = 0.20

def metrics_for(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
    }

def save_confusion(y_true, y_pred, name):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    plt.figure(figsize=(5.5, 4.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Ham", "Spam"], yticklabels=["Ham", "Spam"])
    plt.title(f"Confusion Matrix — {name}")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / f"confusion_{name.lower().replace(' ', '_')}.png", dpi=200)
    plt.close()

def main():
    df = prepare_labels(load_dataset())
    df["text"] = df["text"].map(clean_text)
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=TEST_SIZE, random_state=RANDOM_STATE,
        stratify=df["label"]
    )
    print(f"Train rows: {len(X_train)} | Test rows: {len(X_test)}")
    print("Split: 80/20, stratified; random_state=42")

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Multinomial Naive Bayes": MultinomialNB(alpha=0.5),
    }
    rows = []
    for name, classifier in models.items():
        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1,
                                      max_features=100000, sublinear_tf=True)),
            ("classifier", classifier),
        ])
        start = time.perf_counter()
        pipeline.fit(X_train, y_train)
        train_seconds = time.perf_counter() - start
        predictions = pipeline.predict(X_test)
        scores = metrics_for(y_test, predictions)
        scores.update({"model": name, "train_time_seconds": train_seconds})
        rows.append(scores)
        save_confusion(y_test, predictions, name)
        joblib.dump(pipeline, MODELS_DIR / f"{name.lower().replace(' ', '_')}.joblib")
        print(f"\n--- {name} ---")
        print(json.dumps(scores, indent=2))
        print(classification_report(y_test, predictions, target_names=["Ham", "Spam"], zero_division=0))

    result = pd.DataFrame(rows)
    result.to_csv(REPORTS_DIR / "classical_metrics.csv", index=False)
    print("\nSaved models and reports.")
    print(result.to_string(index=False))

if __name__ == "__main__":
    main()
