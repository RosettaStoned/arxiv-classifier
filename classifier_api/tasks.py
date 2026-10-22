import logging

from celery import current_app, shared_task
from celery_batches import Batches, SimpleRequest
from django.conf import settings
from django_redis.exceptions import ConnectionInterrupted
from redis.exceptions import RedisError

from .prediction_cache import classification_cache_key, set_cached_prediction
from .services import run_prediction

logger = logging.getLogger(__name__)


def process_prediction_batch(batch: list[SimpleRequest]) -> None:
    try:
        predictions = run_prediction([item.args[0] for item in batch])
        if len(predictions) != len(batch):
            raise ValueError("Model returned an unexpected number of predictions")
    except Exception:
        logger.exception("Model inference failed for %s tasks", len(batch))
        for item in batch:
            try:
                current_app.backend.mark_as_failure(
                    item.id, RuntimeError("Inference service unavailable")
                )
            except Exception:
                logger.exception("Could not store failure state for task %s", item.id)
        return

    for item, prediction in zip(batch, predictions, strict=True):
        try:
            set_cached_prediction(
                classification_cache_key(item.args[0]), prediction
            )
        except (ConnectionInterrupted, RedisError, ValueError, TypeError):
            logger.warning("Could not cache async classification result")

        try:
            current_app.backend.mark_as_done(
                item.id, {**prediction, "cached": False}
            )
        except Exception:
            logger.exception("Could not store result for task %s", item.id)


@shared_task(
    base=Batches,
    flush_every=settings.ASYNC_PREDICTION_BATCH_SIZE,
    flush_interval=settings.ASYNC_PREDICTION_BATCH_FLUSH_INTERVAL,
    name="classifier_api.classify_abstract_async",
)
def classify_abstract_async(batch: list[SimpleRequest]) -> None:
    process_prediction_batch(batch)
