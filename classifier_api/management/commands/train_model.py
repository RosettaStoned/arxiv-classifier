import json
import math
import os
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from datasets import Dataset
from django.core.management.base import BaseCommand, CommandError
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from classifier_api.preprocessing import (
    PREPROCESSING_VARIANTS,
    clean_text,
    preprocess_text,
)

MODEL_NAME = "google-bert/bert-base-uncased"


def load_dataset_sample(path, max_per_label, seed, max_records=None):
    """Read eligible papers and keep a small reservoir for each arXiv code."""
    random_generator = random.Random(seed)
    samples = defaultdict(list)
    label_counts = Counter()
    report = {
        "records_read": 0,
        "invalid_records": 0,
        "missing_categories": 0,
        "missing_abstracts": 0,
        "short_abstracts": 0,
    }
    length_count = 0
    total_length = 0
    shortest_length = None
    longest_length = 0

    try:
        with open(path, encoding="utf-8") as dataset_file:
            for line_number, line in enumerate(dataset_file, start=1):
                if max_records is not None and line_number > max_records:
                    break

                report["records_read"] += 1
                try:
                    paper = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise CommandError(
                        f"Invalid JSON on line {line_number}: {exc}"
                    ) from exc

                if not isinstance(paper, dict):
                    report["invalid_records"] += 1
                    continue

                abstract = paper.get("abstract")
                raw_categories = paper.get("categories")
                categories = (
                    sorted(set(raw_categories.split()))
                    if isinstance(raw_categories, str)
                    else []
                )
                if not categories:
                    report["missing_categories"] += 1
                if not isinstance(abstract, str) or not abstract.strip():
                    report["missing_abstracts"] += 1
                    continue
                if not categories:
                    continue

                length = len(clean_text(abstract))
                if length < 20:
                    report["short_abstracts"] += 1
                    continue

                length_count += 1
                total_length += length
                shortest_length = (
                    length if shortest_length is None else min(shortest_length, length)
                )
                longest_length = max(longest_length, length)
                paper_sample = {
                    "id": line_number,
                    "abstract": abstract,
                    "categories": categories,
                }
                for category in categories:
                    label_counts[category] += 1
                    reservoir = samples[category]
                    if len(reservoir) < max_per_label:
                        reservoir.append(paper_sample)
                    else:
                        index = random_generator.randrange(label_counts[category])
                        if index < max_per_label:
                            reservoir[index] = paper_sample
    except (OSError, UnicodeError) as exc:
        raise CommandError(f"Could not read dataset: {exc}") from exc

    # A paper sampled for multiple codes is included once with every code.
    papers = {
        paper["id"]: paper for reservoir in samples.values() for paper in reservoir
    }
    label_names = sorted(samples)
    label_to_id = {label: index for index, label in enumerate(label_names)}
    texts = []
    targets = []
    for paper in sorted(papers.values(), key=lambda item: item["id"]):
        texts.append(paper["abstract"])
        targets.append([label_to_id[label] for label in paper["categories"]])

    report.update(
        labels=label_names,
        eligible_documents=length_count,
        eligible_examples_by_label=dict(sorted(label_counts.items())),
        reservoir_examples_by_label={
            label: len(samples[label]) for label in label_names
        },
        sampled_examples_by_label=dict(
            sorted(
                Counter(
                    label for paper in papers.values() for label in paper["categories"]
                ).items()
            )
        ),
        sampled_documents=len(texts),
        cleaned_abstract_length={
            "min": shortest_length,
            "max": longest_length if length_count else None,
            "mean": round(total_length / length_count, 1) if length_count else None,
        },
    )
    return texts, targets, label_names, report


def calculate_metrics(result):
    """Score multi-label predictions using a simple 0.5 probability threshold."""
    probabilities = 1 / (1 + np.exp(-np.clip(result.predictions, -80, 80)))
    predictions = probabilities >= 0.5

    # Make sure every paper gets at least one predicted subject code.
    empty_rows = ~predictions.any(axis=1)
    predictions[empty_rows, probabilities[empty_rows].argmax(axis=1)] = True

    targets = result.label_ids.astype(bool)
    labels = np.flatnonzero(targets.any(axis=0) | predictions.any(axis=0))
    precision, recall, f1, _ = precision_recall_fscore_support(
        targets,
        predictions,
        labels=labels,
        average="macro",
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(targets, predictions)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


class Command(BaseCommand):
    help = "Train BERT to predict the arXiv subject codes on each abstract."

    def add_arguments(self, parser):
        parser.add_argument("dataset", help="Path to the arXiv JSONL dataset")
        parser.add_argument(
            "--max-per-label",
            "--max-per-class",
            dest="max_per_label",
            type=int,
            default=25,
            help="Reservoir sample size per subject code (default: 25)",
        )
        parser.add_argument(
            "--max-records",
            type=int,
            help="Stop after this many input lines; useful for a quick demo",
        )
        parser.add_argument("--epochs", type=int, default=1)
        parser.add_argument("--batch-size", type=int, default=8)
        parser.add_argument("--max-length", type=int, default=512)
        parser.add_argument(
            "--learning-rates",
            type=float,
            nargs="+",
            default=[1e-5, 2e-5],
        )
        parser.add_argument(
            "--preprocessing",
            nargs="+",
            choices=PREPROCESSING_VARIANTS,
            default=list(PREPROCESSING_VARIANTS),
            help="Text variants to compare (default: all)",
        )
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument(
            "--output",
            default=os.environ.get("MODEL_PATH", "classifier_api/model"),
        )

    def handle(self, *args, **options):
        if options["max_per_label"] < 1:
            raise CommandError("--max-per-label must be positive")
        if options["max_records"] is not None and options["max_records"] < 1:
            raise CommandError("--max-records must be positive")
        if options["epochs"] < 1 or options["batch_size"] < 1:
            raise CommandError("--epochs and --batch-size must be positive")
        if not 1 <= options["max_length"] <= 512:
            raise CommandError("--max-length must be between 1 and 512")

        learning_rates = list(dict.fromkeys(options["learning_rates"]))
        if not learning_rates or any(
            not math.isfinite(rate) or rate <= 0 for rate in learning_rates
        ):
            raise CommandError("--learning-rates must contain positive values")

        texts, category_ids, label_names, data_report = load_dataset_sample(
            options["dataset"],
            options["max_per_label"],
            options["seed"],
            options["max_records"],
        )
        if len(texts) < 5:
            raise CommandError("Need at least five sampled abstracts to train")

        # BERT expects one 0/1 target value for every known subject code.
        targets = [
            [float(label_id in ids) for label_id in range(len(label_names))]
            for ids in category_ids
        ]
        train_texts, validation_texts, train_targets, validation_targets = (
            train_test_split(
                texts,
                targets,
                test_size=0.2,
                random_state=options["seed"],
            )
        )

        self.stdout.write("Dataset summary:")
        self.stdout.write(json.dumps(data_report, indent=2))
        self.stdout.write(
            f"Training papers: {len(train_texts)}; "
            f"validation papers: {len(validation_texts)}"
        )

        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        output_path = Path(options["output"]).resolve()
        output_path.mkdir(parents=True, exist_ok=True)
        experiment_results = []
        best_f1 = float("-inf")
        best_experiment = None

        for variant in options["preprocessing"]:
            train_data = Dataset.from_dict(
                {
                    "text": [preprocess_text(text, variant) for text in train_texts],
                    "labels": train_targets,
                }
            )
            validation_data = Dataset.from_dict(
                {
                    "text": [
                        preprocess_text(text, variant) for text in validation_texts
                    ],
                    "labels": validation_targets,
                }
            )

            def tokenize(batch):
                return tokenizer(
                    batch["text"],
                    truncation=True,
                    max_length=options["max_length"],
                )

            train_data = train_data.map(tokenize, batched=True)
            validation_data = validation_data.map(tokenize, batched=True)

            for learning_rate in learning_rates:
                model = AutoModelForSequenceClassification.from_pretrained(
                    MODEL_NAME,
                    num_labels=len(label_names),
                    id2label=dict(enumerate(label_names)),
                    label2id={label: index for index, label in enumerate(label_names)},
                )
                model.config.problem_type = "multi_label_classification"
                model.config.preprocessing_variant = variant

                trainer = Trainer(
                    model=model,
                    args=TrainingArguments(
                        output_dir=str(output_path / "training-runs"),
                        learning_rate=learning_rate,
                        per_device_train_batch_size=options["batch_size"],
                        per_device_eval_batch_size=options["batch_size"],
                        num_train_epochs=options["epochs"],
                        weight_decay=0.01,
                        eval_strategy="epoch",
                        save_strategy="no",
                        report_to="none",
                        seed=options["seed"],
                        data_seed=options["seed"],
                    ),
                    train_dataset=train_data,
                    eval_dataset=validation_data,
                    tokenizer=tokenizer,
                    data_collator=DataCollatorWithPadding(tokenizer),
                    compute_metrics=calculate_metrics,
                )
                trainer.train()
                metrics = trainer.evaluate()
                experiment = {
                    "preprocessing": variant,
                    "learning_rate": learning_rate,
                    "metrics": {
                        key: float(value)
                        for key, value in metrics.items()
                        if isinstance(value, (int, float))
                    },
                }
                experiment_results.append(experiment)
                self.stdout.write(json.dumps(experiment, indent=2))

                if metrics["eval_f1"] > best_f1:
                    best_f1 = metrics["eval_f1"]
                    best_experiment = experiment
                    trainer.save_model(str(output_path))
                    tokenizer.save_pretrained(str(output_path))

                del trainer, model

        report_path = output_path / "training_report.json"
        report_path.write_text(
            json.dumps(
                {
                    "model": MODEL_NAME,
                    "seed": options["seed"],
                    "epochs": options["epochs"],
                    "dataset": data_report,
                    "split": {
                        "train_examples": len(train_texts),
                        "validation_examples": len(validation_texts),
                    },
                    "labels": label_names,
                    "selection_metric": "macro multi-label F1",
                    "experiments": experiment_results,
                    "selected_experiment": best_experiment,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        self.stdout.write(self.style.SUCCESS(f"Saved best model to {output_path}"))
        self.stdout.write(f"Training report saved to {report_path}")
