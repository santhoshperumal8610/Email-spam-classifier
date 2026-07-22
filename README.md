# Email-spam-classifier

A minimal AI project that detects and classifies spam emails using an existing labeled dataset.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Dataset format

Use any existing CSV dataset with:
- A text column (supported names: `text`, `message`, `email`, `body`, `v2`)
- A label column (supported names: `label`, `class`, `target`, `category`, `v1`)

Label values supported: `spam`/`ham` (or `1`/`0`).

## Train

```bash
python spam_classifier.py train --dataset /path/to/your_dataset.csv --model-output models/spam_classifier.joblib
```

## Predict

```bash
python spam_classifier.py predict --model-path models/spam_classifier.joblib --text "Congratulations! You have won a free ticket."
```

Output is either `spam` or `ham`.
