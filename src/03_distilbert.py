import json
import time
import random

import numpy as np
import pandas as pd
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from src.common import (
    load_dataset,
    prepare_labels,
    MODELS_DIR,
    REPORTS_DIR,
    OUTPUTS_DIR,
)

# ---------------- CONFIGURATION ----------------
MODEL_NAME = "distilbert-base-uncased"
RANDOM_STATE = 42

TEST_SIZE = 0.20
VAL_SIZE_FROM_TRAIN = 0.10

MAX_LENGTH = 128
BATCH_SIZE = 8
EPOCHS = 1
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01


# ---------------- DATASET ----------------
class EmailDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        encoded = self.tokenizer(
            str(self.texts[index]),
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )

        item = {key: value.squeeze(0) for key, value in encoded.items()}

        item["labels"] = torch.tensor(
            int(self.labels[index]),
            dtype=torch.long,
        )

        return item


# ---------------- REPRODUCIBILITY ----------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------- EVALUATION ----------------
def evaluate(model, loader, device):
    model.eval()

    truths = []
    predictions = []
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}

            outputs = model(**batch)

            batch_size = batch["labels"].size(0)
            total_loss += outputs.loss.item() * batch_size
            total_samples += batch_size

            preds = torch.argmax(outputs.logits, dim=1)

            predictions.extend(preds.cpu().tolist())
            truths.extend(batch["labels"].cpu().tolist())

    average_loss = total_loss / max(total_samples, 1)

    return average_loss, truths, predictions


# ---------------- MAIN ----------------
def main():
    set_seed(RANDOM_STATE)

    # Ensure output directories exist.
    for folder in (MODELS_DIR, REPORTS_DIR, OUTPUTS_DIR):
        folder.mkdir(parents=True, exist_ok=True)

    print("\n========== DATASET LOADING ==========")

    df = prepare_labels(load_dataset())

    # Separate test data before creating train/validation sets.
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        df["text"],
        df["label"],
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["label"],
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=VAL_SIZE_FROM_TRAIN,
        random_state=RANDOM_STATE,
        stratify=y_trainval,
    )

    print(
        f"Train: {len(X_train)} | "
        f"Validation: {len(X_val)} | "
        f"Test: {len(X_test)}"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    if device.type == "cpu":
        print("CPU training may be slow. " f"Testing with EPOCHS={EPOCHS}.")

    # ---------------- TOKENIZER AND MODEL ----------------
    print("\nLoading tokenizer and DistilBERT...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        id2label={0: "Ham", 1: "Spam"},
        label2id={"Ham": 0, "Spam": 1},
    )

    model.to(device)

    # ---------------- DATA LOADERS ----------------
    train_loader = DataLoader(
        EmailDataset(X_train, y_train, tokenizer),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        EmailDataset(X_val, y_val, tokenizer),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    test_loader = DataLoader(
        EmailDataset(X_test, y_test, tokenizer),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    # ---------------- OPTIMIZER ----------------
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    total_steps = len(train_loader) * EPOCHS

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    # ---------------- CUSTOM PYTORCH TRAINING LOOP ----------------
    history = []
    training_start = time.perf_counter()

    for epoch in range(EPOCHS):
        model.train()

        running_loss = 0.0
        total_samples = 0
        epoch_start = time.perf_counter()

        print(f"\nEpoch {epoch + 1}/{EPOCHS}")

        for step, batch in enumerate(train_loader, start=1):
            batch = {key: value.to(device) for key, value in batch.items()}

            optimizer.zero_grad(set_to_none=True)

            outputs = model(**batch)
            loss = outputs.loss

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            optimizer.step()
            scheduler.step()

            batch_size = batch["labels"].size(0)
            running_loss += loss.item() * batch_size
            total_samples += batch_size

            if step == 1 or step % 25 == 0:
                print(
                    f"Batch {step}/{len(train_loader)} | " f"Loss: {loss.item():.4f}",
                    flush=True,
                )

        train_loss = running_loss / max(total_samples, 1)

        val_loss, val_true, val_pred = evaluate(
            model,
            val_loader,
            device,
        )

        val_accuracy = accuracy_score(val_true, val_pred)
        epoch_seconds = time.perf_counter() - epoch_start

        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "validation_loss": val_loss,
                "validation_accuracy": val_accuracy,
                "epoch_time_seconds": epoch_seconds,
            }
        )

        print(
            f"Epoch {epoch + 1} complete | "
            f"Train loss: {train_loss:.4f} | "
            f"Validation loss: {val_loss:.4f} | "
            f"Validation accuracy: {val_accuracy:.4f} | "
            f"Time: {epoch_seconds:.1f}s",
            flush=True,
        )

    train_seconds = time.perf_counter() - training_start

    # ---------------- FINAL TEST EVALUATION ----------------
    print("\nEvaluating on the test set...")

    test_loss, y_true, y_pred = evaluate(
        model,
        test_loader,
        device,
    )

    scores = {
        "model": "DistilBERT",
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "train_time_seconds": train_seconds,
        "test_loss": test_loss,
        "device": str(device),
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "max_length": MAX_LENGTH,
    }

    print("\n========== DISTILBERT TEST METRICS ==========")
    print(json.dumps(scores, indent=2))

    print("\nClassification report:")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=["Ham", "Spam"],
            zero_division=0,
        )
    )

    # ---------------- SAVE RESULTS ----------------
    history_df = pd.DataFrame(history)

    history_df.to_csv(
        REPORTS_DIR / "distilbert_training_history.csv",
        index=False,
    )

    with open(
        REPORTS_DIR / "distilbert_metrics.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(scores, file, indent=2)

    torch.save(
        model.state_dict(),
        MODELS_DIR / "distilbert_spam_classifier.pt",
    )

    tokenizer.save_pretrained(MODELS_DIR / "distilbert_tokenizer")

    model.config.to_json_file(MODELS_DIR / "distilbert_config.json")

    # ---------------- TRAINING LOSS GRAPH ----------------
    plt.figure(figsize=(7, 5))

    plt.plot(
        history_df["epoch"],
        history_df["train_loss"],
        marker="o",
        label="Training loss",
    )

    plt.plot(
        history_df["epoch"],
        history_df["validation_loss"],
        marker="o",
        label="Validation loss",
    )

    plt.title("DistilBERT Training Behaviour")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.xticks(history_df["epoch"])
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        OUTPUTS_DIR / "distilbert_training_loss.png",
        dpi=200,
    )
    plt.close()

    # ---------------- CONFUSION MATRIX ----------------
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    )

    plt.figure(figsize=(5.5, 4.5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Ham", "Spam"],
        yticklabels=["Ham", "Spam"],
    )

    plt.title("Confusion Matrix — DistilBERT")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.tight_layout()

    plt.savefig(
        OUTPUTS_DIR / "confusion_distilbert.png",
        dpi=200,
    )
    plt.close()

    print("\n========== SAVING COMPLETE ==========")
    print(f"Training time: {train_seconds:.1f} seconds")
    print(f"Metrics: {REPORTS_DIR / 'distilbert_metrics.json'}")
    print(f"Training history: {REPORTS_DIR / 'distilbert_training_history.csv'}")
    print(f"Plots saved in: {OUTPUTS_DIR}")
    print("DistilBERT pipeline completed successfully.")


if __name__ == "__main__":
    main()
