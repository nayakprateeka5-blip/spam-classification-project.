import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from src.common import REPORTS_DIR, OUTPUTS_DIR

METRICS = ["accuracy", "precision", "recall", "f1_score"]

def main():
    paths = [REPORTS_DIR / "classical_metrics.csv", REPORTS_DIR / "distilbert_metrics.json"]
    if not paths[0].exists():
        raise FileNotFoundError("Run: python -m src.02_classical first")
    if not paths[1].exists():
        raise FileNotFoundError("Run: python -m src.03_distilbert first")
    classical = pd.read_csv(paths[0])
    with open(paths[1], encoding="utf-8") as f:
        transformer = pd.DataFrame([json.load(f)])
    all_results = pd.concat([classical, transformer], ignore_index=True, sort=False)
    all_results.to_csv(REPORTS_DIR / "all_model_metrics.csv", index=False)
    print("\n--- Final comparison ---")
    print(all_results[["model"] + METRICS + ["train_time_seconds"]].to_string(index=False))

    plot_df = all_results.melt(id_vars="model", value_vars=METRICS,
                               var_name="Metric", value_name="Score")
    plt.figure(figsize=(10, 6))
    sns.barplot(data=plot_df, x="Metric", y="Score", hue="model")
    plt.ylim(0, 1)
    plt.title("Spam Classification: Model Metric Comparison")
    plt.ylabel("Score (0–1)")
    plt.xlabel("Evaluation metric")
    plt.legend(title="Model", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / "metric_comparison.png", dpi=200)
    plt.close()

    plt.figure(figsize=(9, 5))
    sns.barplot(data=all_results, x="model", y="train_time_seconds")
    plt.title("Training Time Comparison")
    plt.ylabel("Training time (seconds)")
    plt.xlabel("Model")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / "training_time_comparison.png", dpi=200)
    plt.close()
    print("\nSaved outputs/metric_comparison.png and outputs/training_time_comparison.png")
    print("Important: training time depends on CPU/GPU hardware, so report your own measured times.")

if __name__ == "__main__":
    main()
