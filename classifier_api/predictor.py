import os
from collections.abc import Sequence
from threading import Lock

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .preprocessing import preprocess_text
from .types import CategoryPrediction, Prediction

tokenizer = None
model = None
_model_load_lock = Lock()
PREDICTION_THRESHOLD = 0.5


def ensure_model_loaded() -> None:
    global tokenizer, model
    if model is None or tokenizer is None:
        with _model_load_lock:
            if model is not None and tokenizer is not None:
                return
            model_path = os.environ.get("MODEL_PATH", "classifier_api/model")
            if not os.path.isdir(model_path):
                raise FileNotFoundError(
                    f"Model not found at {model_path}; run train_model first"
                )
            loaded_tokenizer = AutoTokenizer.from_pretrained(model_path)
            loaded_model = AutoModelForSequenceClassification.from_pretrained(
                model_path
            )
            loaded_model.eval()
            tokenizer, model = loaded_tokenizer, loaded_model


def predict_batch(abstracts: Sequence[str]) -> list[Prediction]:
    ensure_model_loaded()
    assert tokenizer is not None
    assert model is not None

    inputs = tokenizer(
        [
            preprocess_text(
                abstract,
                getattr(model.config, "preprocessing_variant", "clean"),
            )
            for abstract in abstracts
        ],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    )
    with torch.inference_mode():
        logits = model(**inputs).logits
        multilabel = model.config.problem_type == "multi_label_classification"
        probabilities = (
            torch.sigmoid(logits) if multilabel else torch.softmax(logits, dim=-1)
        )
    predictions: list[Prediction] = []
    for row in probabilities:
        if multilabel:
            selected_indices = torch.where(row >= PREDICTION_THRESHOLD)[0].tolist()
        else:
            selected_indices = []
        if not selected_indices:
            selected_indices = [int(row.argmax())]
        selected_indices.sort(key=lambda index: float(row[index]), reverse=True)
        labels = model.config.id2label
        categories: list[CategoryPrediction] = [
            {
                "category": labels[index],
                "confidence": round(float(row[index]), 4),
            }
            for index in selected_indices
        ]
        predictions.append(
            {
                "category": categories[0]["category"],
                "confidence": categories[0]["confidence"],
                "categories": categories,
            }
        )
    return predictions
