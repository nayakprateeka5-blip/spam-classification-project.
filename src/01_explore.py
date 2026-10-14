import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from src.common import load_dataset, prepare_labels, OUTPUTS_DIR, REPORTS_DIR

def main():
    df = prepare_labels(load_dataset())
    print("\n--- Dataset overview ---")
    print(df.head(5).to_string(index=False))
    print("\nShape:", df.shape)
    print("\nMissing values:\n", df.isna().sum())
    print("\nDuplicate rows:", int(df.duplicated().sum()))
    print("\nClass counts:\n", df["label"].map({0: "Ham", 1: "Spam"}).value_counts())
    df["text_length"] = df["text"].str.len()
    print("\nText length summary:\n", df["text_length"].describe())

    sns.set_theme(style="whitegrid")
    counts = df["label"].map({0: "Ham", 1: "Spam"}).value_counts().reindex(["Ham", "Spam"])
    plt.figure(figsize=(7, 5))
    sns.barplot(x=counts.index, y=counts.values)
    plt.title("Email Class Distribution")
    plt.xlabel("Class")
    plt.ylabel("Number of emails")
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / "class_distribution.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.histplot(data=df, x="text_length", hue=df["label"].map({0: "Ham", 1: "Spam"}),
                 bins=50, element="step", stat="count", common_norm=False)
    plt.title("Email Text Length Distribution")
    plt.xlabel("Characters per email")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / "text_length_distribution.png", dpi=200)
    plt.close()

    df.drop(columns=["text_length"]).to_csv(REPORTS_DIR / "cleaned_dataset.csv", index=False)
    print(f"\nSaved EDA charts in {OUTPUTS_DIR} and cleaned data in {REPORTS_DIR}")

if __name__ == "__main__":
    main()
