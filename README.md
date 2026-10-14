# Spam Email Classification: TF-IDF vs DistilBERT

## Project objective
Compare two traditional ML models (Logistic Regression and Multinomial Naive Bayes) with a fine-tuned DistilBERT model for ham/spam email classification.

## 1. Dataset
Download the assignment's required dataset:
https://www.kaggle.com/datasets/ashfakyeafi/spam-email-classification

Place the CSV file in `data/`. The scripts print the detected text and label columns. If detection is wrong, edit `TEXT_COLUMN` and `LABEL_COLUMN` in `src/common.py`.

Labels must be recognizable as spam/ham, 1/0, junk/legitimate, etc. If the dataset uses different names, edit `normalize_label()` in `src/common.py` so spam=1 and ham=0.

## 2. Create environment (Windows PowerShell)
```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```
If PowerShell blocks activation, use:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## 3. Run in this order
```powershell
python -m src.01_explore
python -m src.02_classical
python -m src.03_distilbert
python -m src.04_compare
```
Run each command from the project root. The first DistilBERT run downloads model files and needs internet access.

## 4. Outputs
- `reports/`: metrics, cleaned dataset, training history
- `models/`: saved classical pipelines and DistilBERT weights/tokenizer
- `outputs/`: labeled plots and confusion matrices

## 5. Methodology decisions
- Use an 80/20 stratified train/test split to preserve the ham/spam ratio.
- Split off a validation set from the training data for DistilBERT; do not tune on the test set.
- TF-IDF uses unigrams and bigrams, which can represent individual words and short phrases.
- Logistic Regression uses class weighting to reduce bias when classes are imbalanced.
- Multinomial Naive Bayes is a strong text-classification baseline; alpha=0.5 applies smoothing.
- DistilBERT uses max length 192, batch size 8, learning rate 2e-5, AdamW, weight decay, gradient clipping, and a linear learning-rate schedule.
- Two epochs are a practical starting point; compare validation loss and available compute before increasing epochs.

## 6. Limitations to discuss
- Results depend on dataset size, label quality, and class balance.
- Emails longer than the maximum token length are truncated.
- DistilBERT requires more compute and memory than TF-IDF models.
- Dataset patterns may not represent current real-world spam.
- A model can produce false positives (legitimate email marked spam) and false negatives (spam marked ham).
- Training times vary by hardware.

## Troubleshooting
- **Column detection wrong:** set `TEXT_COLUMN` and `LABEL_COLUMN` in `src/common.py`.
- **Unknown label:** edit `normalize_label()` to map the dataset's actual labels.
- **Out of memory:** lower `BATCH_SIZE` to 4 and `MAX_LENGTH` to 128 in `src/03_distilbert.py`.
- **Slow CPU:** set `EPOCHS = 1` for a pipeline test first, then run more epochs if time permits.
- **Transformers/PyTorch install issue:** use a compatible Python version (Python 3.10 or 3.11 is a safe starting point) and install PyTorch according to the official PyTorch instructions for your machine.
