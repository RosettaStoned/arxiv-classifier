import asyncio
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from threading import BoundedSemaphore, Event

from asgiref.sync import sync_to_async
from django.conf import settings

from .exceptions import InferenceOverloaded
from .types import Prediction

prediction_executor = ThreadPoolExecutor(
    max_workers=1, thread_name_prefix="model-inference"
)
inference_slots = BoundedSemaphore(1 + settings.CLASSIFICATION_MAX_PENDING)


def run_prediction(abstracts: Sequence[str]) -> list[Prediction]:
    from .predictor import predict_batch

    return predict_batch(abstracts)


async def run_prediction_async(abstracts: Sequence[str]) -> list[Prediction]:
    if not inference_slots.acquire(blocking=False):
        raise InferenceOverloaded()

    started = Event()

    def predict_and_release(abstracts: Sequence[str]) -> list[Prediction]:
        started.set()
        try:
            return run_prediction(abstracts)
        finally:
            inference_slots.release()

    run_prediction_in_executor = sync_to_async(
        predict_and_release,
        thread_sensitive=False,
        executor=prediction_executor,
    )
    try:
        return await asyncio.shield(run_prediction_in_executor(abstracts))
    except Exception:
        if not started.is_set():
            inference_slots.release()
        raise
