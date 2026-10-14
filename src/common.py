from pathlib import Path
import re
import pandas as pd

# ==========================================
# 1. PROJECT DIRECTORIES
# ==========================================

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
OUTPUTS_DIR = ROOT / "outputs"

for folder in [MODELS_DIR, REPORTS_DIR, OUTPUTS_DIR]:
    folder.mkdir(parents=True, exist_ok=True)


# ==========================================
# 2. DATASET COLUMN NAMES
# ==========================================

TEXT_COLUMN = "Message"
LABEL_COLUMN = "Category"


# ==========================================
# 3. LOAD AND CLEAN DATASET
# ==========================================


def load_dataset():

    csv_files = sorted(DATA_DIR.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(
            f"No CSV file found in {DATA_DIR}. "
            "Place email.csv inside the data folder."
        )

    # Read the largest CSV file in the data folder.
    # This avoids accidentally choosing a smaller metadata file.
    frames = []

    for path in csv_files:
        try:
            df = pd.read_csv(path, encoding="utf-8", low_memory=False)
        except UnicodeDecodeError:
            df = pd.read_csv(path, encoding="latin-1", low_memory=False)

        if len(df.columns) >= 2:
            frames.append((path, df))

    if not frames:
        raise ValueError("No valid CSV with at least two columns was found.")

    path, df = max(frames, key=lambda item: len(item[1]))

    # Clean column names
    df.columns = [str(col).strip() for col in df.columns]

    # Verify expected columns exist
    if TEXT_COLUMN not in df.columns:
        raise ValueError(
            f"Text column '{TEXT_COLUMN}' not found. "
            f"Available columns: {list(df.columns)}"
        )

    if LABEL_COLUMN not in df.columns:
        raise ValueError(
            f"Label column '{LABEL_COLUMN}' not found. "
            f"Available columns: {list(df.columns)}"
        )

    # Keep only the required columns
    result = df[[TEXT_COLUMN, LABEL_COLUMN]].copy()

    result.columns = ["text", "label"]

    # Remove missing values
    result = result.dropna(subset=["text", "label"])

    # Convert values to strings and remove surrounding spaces
    result["text"] = result["text"].astype(str).str.strip()
    result["label"] = result["label"].astype(str).str.strip()

    # Remove empty messages and labels
    result = result[result["text"].ne("") & result["label"].ne("")]

    # Standardize label capitalization
    result["label"] = result["label"].str.lower()

    # IMPORTANT:
    # Keep only genuine ham and spam labels.
    # This removes malformed labels such as {"mode":"full".
    valid_labels = ["ham", "spam"]

    before_filter = len(result)

    result = result[result["label"].isin(valid_labels)].copy()

    removed = before_filter - len(result)

    # Remove duplicate email records
    duplicate_count = int(result.duplicated(subset=["text", "label"]).sum())

    result = result.drop_duplicates(subset=["text", "label"]).reset_index(drop=True)

    if result.empty:
        raise ValueError(
            "No valid email records remain. " "Check the CSV columns and label values."
        )

    print("\n========== DATASET LOADING ==========")
    print(f"File: {path.name}")
    print(f"Valid records: {len(result):,}")
    print(f"Invalid-label records removed: {removed:,}")
    print(f"Duplicate records removed: {duplicate_count:,}")
    print(f"Text column: {TEXT_COLUMN}")
    print(f"Label column: {LABEL_COLUMN}")
    print(f"Label values: {sorted(result['label'].unique())}")
    print("=====================================\n")

    return result


# ==========================================
# 4. CONVERT LABELS TO NUMBERS
# ==========================================


def normalize_label(value):

    label = str(value).strip().lower()

    if label == "spam":
        return 1

    if label == "ham":
        return 0

    return None


def prepare_labels(df):

    df = df.copy()

    df["label"] = df["label"].map(normalize_label)

    # Remove any labels that could not be converted
    df = df.dropna(subset=["label"])

    df["label"] = df["label"].astype(int)

    if df.empty:
        raise ValueError("No valid ham or spam labels were found.")

    if df["label"].nunique() != 2:
        raise ValueError("Both ham and spam classes are required.")

    print("Label encoding:")
    print("Ham  = 0")
    print("Spam = 1")

    return df


# ==========================================
# 5. TEXT PREPROCESSING
# ==========================================


def clean_text(text):

    # Convert text to lowercase
    text = str(text).lower()

    # Replace URLs with a common token
    text = re.sub(r"https?://\S+|www\.\S+", " URL ", text)

    # Replace email addresses with a common token
    text = re.sub(r"\b[\w.+-]+@[\w.-]+\.[a-z]{2,}\b", " EMAIL ", text)

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text
