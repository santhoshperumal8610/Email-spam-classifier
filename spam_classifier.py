import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


TEXT_COLUMN_CANDIDATES = ["text", "message", "email", "body", "v2"]
LABEL_COLUMN_CANDIDATES = ["label", "class", "target", "category", "v1"]



def _find_column(columns, candidates):
    lower_map = {c.lower(): c for c in columns}
    for candidate in candidates:
        if candidate in lower_map:
            return lower_map[candidate]
    return None



def _to_binary_label(value):
    normalized = str(value).strip().lower()
    spam_values = {"spam", "junk", "1", "true", "yes"}
    ham_values = {"ham", "not spam", "0", "false", "no"}

    if normalized in spam_values:
        return 1
    if normalized in ham_values:
        return 0

    raise ValueError(f"Unsupported label value '{value}'. Use spam/ham or 1/0 labels.")



def load_dataset(csv_path):
    data = pd.read_csv(csv_path)
    text_column = _find_column(data.columns, TEXT_COLUMN_CANDIDATES)
    label_column = _find_column(data.columns, LABEL_COLUMN_CANDIDATES)

    if text_column is None or label_column is None:
        raise ValueError(
            "Dataset must include a text column (e.g., text/message/email) "
            "and a label column (e.g., label/class/target)."
        )

    dataset = data[[text_column, label_column]].dropna()
    if dataset.empty:
        raise ValueError("Dataset is empty after dropping missing values.")

    dataset[text_column] = dataset[text_column].astype(str)
    dataset[label_column] = dataset[label_column].map(_to_binary_label)

    return dataset[text_column], dataset[label_column]



def train_model(dataset_path, output_model_path):
    texts, labels = load_dataset(dataset_path)

    x_train, x_test, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )

    model = Pipeline(
        [
            ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1, 2))),
            ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions, zero_division=0)

    artifact = {
        "model": model,
        "metrics": {
            "accuracy": float(accuracy),
            "f1": float(f1),
            "classification_report": classification_report(y_test, predictions, zero_division=0),
        },
    }

    output_path = Path(output_model_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output_path)

    return artifact["metrics"]



def predict_text(model_path, text):
    artifact = joblib.load(model_path)
    model = artifact["model"]

    prediction = int(model.predict([text])[0])
    return "spam" if prediction == 1 else "ham"



def main():
    parser = argparse.ArgumentParser(
        description="Train and use a spam email classifier with existing CSV datasets."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train a model on a labeled dataset")
    train_parser.add_argument("--dataset", required=True, help="Path to CSV dataset")
    train_parser.add_argument(
        "--model-output",
        default="models/spam_classifier.joblib",
        help="Where to save the trained model",
    )

    predict_parser = subparsers.add_parser("predict", help="Classify a message")
    predict_parser.add_argument(
        "--model-path",
        default="models/spam_classifier.joblib",
        help="Path to the trained model",
    )
    predict_parser.add_argument("--text", required=True, help="Email text to classify")

    args = parser.parse_args()

    if args.command == "train":
        metrics = train_model(args.dataset, args.model_output)
        print(f"Model saved to: {args.model_output}")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"F1 Score: {metrics['f1']:.4f}")
        print("Classification report:\n")
        print(metrics["classification_report"])
        return

    if args.command == "predict":
        label = predict_text(args.model_path, args.text)
        print(label)


if __name__ == "__main__":
    main()
