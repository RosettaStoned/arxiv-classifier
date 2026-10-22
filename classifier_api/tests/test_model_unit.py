import asyncio
import json
import tempfile
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch
from django.test import AsyncClient, SimpleTestCase, override_settings

from classifier_api.exceptions import InferenceOverloaded
from classifier_api.management.commands.train_model import (
    Command,
    calculate_metrics,
    load_dataset_sample,
)
from classifier_api.preprocessing import clean_text, preprocess_text
from classifier_api.views import ModelHealthView


class DataPreparationTests(SimpleTestCase):
    def test_clean_text_lowercases_and_collapses_whitespace(self):
        self.assertEqual(clean_text("  A  Study of CELLS. "), "a study of cells.")

    def test_clean_text_removes_latex_commands_and_symbols(self):
        self.assertEqual(clean_text(r"Let \alpha = \textbf{CELLS}!"), "let cells !")

    def test_preprocessing_variants_remove_stopwords_and_stem_words(self):
        text = "The studies of algorithms."

        self.assertEqual(preprocess_text(text, "clean"), "the studies of algorithms.")
        self.assertEqual(preprocess_text(text, "stopwords"), "studies algorithms .")
        self.assertEqual(
            preprocess_text(text, "stopwords_stemmed"), "studi algorithm ."
        )

    def test_multilabel_metrics_compare_full_subject_code_sets(self):
        result = SimpleNamespace(
            predictions=np.array([[3.0, 2.0, -2.0], [-2.0, 3.0, -1.0]]),
            label_ids=np.array([[1.0, 1.0, 0.0], [0.0, 1.0, 0.0]]),
        )

        self.assertEqual(
            calculate_metrics(result),
            {"accuracy": 1.0, "precision": 1.0, "recall": 1.0, "f1": 1.0},
        )

    def test_dataset_loader_keeps_every_subject_and_samples_deterministically(self):
        records = [
            {
                "categories": "cs.AI math.ST",
                "abstract": "A sufficiently long computer science abstract.",
            },
            {
                "categories": "cs.LG",
                "abstract": "A sufficiently long machine learning abstract.",
            },
            {
                "categories": "physics.gen-ph",
                "abstract": "A sufficiently long physics abstract.",
            },
            {"categories": "math.ST", "abstract": "A sufficiently long math abstract."},
            {"categories": "cs.AI", "abstract": ""},
            {"categories": "chem.ph", "abstract": "short"},
        ]
        with tempfile.TemporaryDirectory() as temporary_directory:
            dataset_path = Path(temporary_directory) / "sample.jsonl"
            dataset_path.write_text(
                "\n".join(json.dumps(record) for record in records),
                encoding="utf-8",
            )

            texts, labels, label_names, report = load_dataset_sample(
                str(dataset_path), max_per_label=5, seed=42
            )

        self.assertEqual(
            texts,
            [
                "A sufficiently long computer science abstract.",
                "A sufficiently long machine learning abstract.",
                "A sufficiently long physics abstract.",
                "A sufficiently long math abstract.",
            ],
        )
        self.assertEqual(label_names, ["cs.AI", "cs.LG", "math.ST", "physics.gen-ph"])
        self.assertEqual(labels, [[0, 2], [1], [3], [2]])
        self.assertEqual(report["records_read"], 6)
        self.assertEqual(report["missing_categories"], 0)
        self.assertEqual(report["missing_abstracts"], 1)
        self.assertEqual(report["short_abstracts"], 1)
        self.assertEqual(report["eligible_examples_by_label"]["cs.AI"], 1)
        self.assertEqual(report["eligible_examples_by_label"]["math.ST"], 2)
        self.assertEqual(report["sampled_examples_by_label"]["cs.AI"], 1)
        self.assertEqual(report["sampled_documents"], 4)
        self.assertIsNotNone(report["cleaned_abstract_length"]["mean"])

    def test_dataset_loader_stops_at_max_records(self):
        records = [
            {"categories": "cs.AI", "abstract": "A sufficiently long first abstract."},
            {
                "categories": "math.ST",
                "abstract": "A sufficiently long second abstract.",
            },
        ]
        with tempfile.TemporaryDirectory() as temporary_directory:
            dataset_path = Path(temporary_directory) / "sample.jsonl"
            dataset_path.write_text(
                "\n".join(json.dumps(record) for record in records),
                encoding="utf-8",
            )

            texts, _, labels, report = load_dataset_sample(
                str(dataset_path), max_per_label=5, seed=42, max_records=1
            )

        self.assertEqual(len(texts), 1)
        self.assertEqual(labels, ["cs.AI"])
        self.assertEqual(report["records_read"], 1)

    def test_sampled_document_keeps_all_codes_even_when_sampled_for_one_code(self):
        record = {
            "categories": "cs.AI math.ST",
            "abstract": "A sufficiently long multi-subject research abstract.",
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            dataset_path = Path(temporary_directory) / "sample.jsonl"
            dataset_path.write_text(json.dumps(record), encoding="utf-8")

            texts, labels, label_names, report = load_dataset_sample(
                str(dataset_path), max_per_label=1, seed=42
            )

        self.assertEqual(len(texts), 1)
        self.assertEqual(label_names, ["cs.AI", "math.ST"])
        self.assertEqual(labels, [[0, 1]])
        self.assertEqual(
            report["sampled_examples_by_label"], {"cs.AI": 1, "math.ST": 1}
        )

    def test_training_selects_and_saves_the_best_validation_configuration(self):
        records = []
        categories_by_label = [
            "q-bio.BM",
            "chem.ph",
            "physics.gen-ph",
            "cs.AI",
            "econ.GN",
        ]
        for label_id, categories in enumerate(categories_by_label):
            for index in range(5):
                records.append(
                    {
                        "categories": categories,
                        "abstract": (
                            f"A sufficiently long sample abstract for class "
                            f"{label_id} item {index}."
                        ),
                    }
                )

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            dataset_path = root / "sample.jsonl"
            output_path = root / "model"
            dataset_path.write_text(
                "\n".join(json.dumps(record) for record in records),
                encoding="utf-8",
            )

            tokenizer = Mock(
                side_effect=lambda texts, **_kwargs: {"input_ids": [[1] for _ in texts]}
            )
            model = SimpleNamespace(config=SimpleNamespace())
            trainer = Mock()
            trainer.evaluate.side_effect = [
                {"eval_f1": score, "eval_accuracy": score}
                for score in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
            ]

            def save_model(path):
                model_path = Path(path)
                model_path.mkdir(parents=True, exist_ok=True)
                (model_path / "config.json").write_text(
                    json.dumps(
                        {"preprocessing_variant": (model.config.preprocessing_variant)}
                    ),
                    encoding="utf-8",
                )

            trainer.save_model.side_effect = save_model
            with (
                patch(
                    "classifier_api.management.commands.train_model.AutoTokenizer.from_pretrained",
                    return_value=tokenizer,
                ),
                patch(
                    "classifier_api.management.commands.train_model.DataCollatorWithPadding"
                ),
                patch(
                    "classifier_api.management.commands.train_model.AutoModelForSequenceClassification.from_pretrained",
                    return_value=model,
                ) as load_model,
                patch(
                    "classifier_api.management.commands.train_model.TrainingArguments"
                ),
                patch(
                    "classifier_api.management.commands.train_model.Trainer",
                    return_value=trainer,
                ),
            ):
                Command().handle(
                    dataset=str(dataset_path),
                    max_per_label=5,
                    max_records=None,
                    epochs=1,
                    batch_size=8,
                    max_length=512,
                    learning_rates=[1e-5, 2e-5],
                    preprocessing=["clean", "stopwords", "stopwords_stemmed"],
                    seed=42,
                    output=str(output_path),
                )

            report = json.loads(
                (output_path / "training_report.json").read_text(encoding="utf-8")
            )
            saved_config = json.loads(
                (output_path / "config.json").read_text(encoding="utf-8")
            )

        self.assertEqual(trainer.train.call_count, 6)
        self.assertEqual(
            report["selected_experiment"]["preprocessing"], "stopwords_stemmed"
        )
        self.assertEqual(report["selected_experiment"]["learning_rate"], 2e-5)
        self.assertEqual(saved_config["preprocessing_variant"], "stopwords_stemmed")
        self.assertEqual(load_model.call_args.kwargs["num_labels"], 5)
        self.assertEqual(load_model.call_args.kwargs["id2label"][0], "chem.ph")


class ModelHealthTests(SimpleTestCase):
    @override_settings(
        CACHES={
            "default": {
                "BACKEND": "django.core.cache.backends.locmem.LocMemCache",
                "LOCATION": "health-check-tests",
            }
        }
    )
    def test_redis_check_rejects_local_memory_cache(self):
        with self.assertRaises(ConnectionError):
            ModelHealthView._check_redis()

    async def test_health_check_reports_all_dependencies(self):
        with (
            patch("classifier_api.views.ModelHealthView._check_redis"),
            patch("classifier_api.views.ModelHealthView._check_celery"),
            patch("classifier_api.views.ensure_model_loaded"),
        ):
            response = await AsyncClient().get("/api/health/")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {
                "status": "ok",
                "checks": {"redis": "ok", "celery": "ok", "model": "ok"},
            },
        )

    async def test_health_check_fails_when_redis_is_unavailable(self):
        with (
            patch(
                "classifier_api.views.ModelHealthView._check_redis",
                side_effect=ConnectionError,
            ),
            patch("classifier_api.views.ModelHealthView._check_celery"),
            patch("classifier_api.views.ensure_model_loaded"),
        ):
            response = await AsyncClient().get("/api/health/")

        self.assertEqual(response.status_code, 503)
        self.assertEqual(
            response.json()["checks"],
            {"redis": "unavailable", "celery": "ok", "model": "ok"},
        )

    async def test_health_check_fails_when_celery_is_unavailable(self):
        with (
            patch("classifier_api.views.ModelHealthView._check_redis"),
            patch(
                "classifier_api.views.ModelHealthView._check_celery",
                side_effect=ConnectionError,
            ),
            patch("classifier_api.views.ensure_model_loaded"),
        ):
            response = await AsyncClient().get("/api/health/")

        self.assertEqual(response.status_code, 503)
        self.assertEqual(
            response.json()["checks"],
            {"redis": "ok", "celery": "unavailable", "model": "ok"},
        )

    async def test_health_check_fails_when_model_cannot_load(self):
        with (
            patch("classifier_api.views.ModelHealthView._check_redis"),
            patch("classifier_api.views.ModelHealthView._check_celery"),
            patch(
                "classifier_api.views.ensure_model_loaded",
                side_effect=FileNotFoundError,
            ),
        ):
            response = await AsyncClient().get("/api/health/")

        self.assertEqual(response.status_code, 503)
        self.assertEqual(
            response.json()["checks"],
            {"redis": "ok", "celery": "ok", "model": "unavailable"},
        )

class PredictorTests(SimpleTestCase):
    @patch("classifier_api.predictor.model")
    @patch("classifier_api.predictor.tokenizer")
    def test_predictor_returns_all_codes_above_threshold_in_score_order(
        self, tokenizer, model
    ):
        tokenizer.return_value = {"input_ids": torch.tensor([[1]])}
        model.config = SimpleNamespace(
            preprocessing_variant="clean",
            problem_type="multi_label_classification",
            id2label={0: "cs.AI", 1: "math.ST", 2: "physics.gen-ph"},
        )
        model.return_value.logits = torch.tensor([[2.0, -1.0, 1.5]])

        from classifier_api.predictor import predict_batch

        predictions = predict_batch(["A sufficiently long research abstract."])

        self.assertEqual(
            predictions,
            [
                {
                    "category": "cs.AI",
                    "confidence": 0.8808,
                    "categories": [
                        {"category": "cs.AI", "confidence": 0.8808},
                        {"category": "physics.gen-ph", "confidence": 0.8176},
                    ],
                }
            ],
        )


class InferenceBackpressureTests(SimpleTestCase):
    async def test_rejects_work_when_the_bounded_inference_slots_are_full(self):
        from classifier_api import services

        started = threading.Event()
        release = threading.Event()

        def slow_prediction(_abstracts):
            started.set()
            release.wait(timeout=2)
            return []

        with (
            patch(
                "classifier_api.services.inference_slots",
                threading.BoundedSemaphore(1),
            ),
            patch("classifier_api.services.run_prediction", side_effect=slow_prediction),
        ):
            first_request = asyncio.create_task(
                services.run_prediction_async(["first abstract"])
            )
            self.assertTrue(await asyncio.to_thread(started.wait, 1))

            with self.assertRaises(InferenceOverloaded):
                await services.run_prediction_async(["second abstract"])

            release.set()
            self.assertEqual(await asyncio.wait_for(first_request, timeout=2), [])
