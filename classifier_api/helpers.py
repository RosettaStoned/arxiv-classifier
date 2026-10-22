import hashlib
import json
import logging
from typing import Any
from uuid import UUID, uuid4

from asgiref.sync import sync_to_async
from celery import current_app
from celery.exceptions import OperationalError
from celery.result import AsyncResult
from django.conf import settings
from django.core.cache import cache
from django_redis.exceptions import ConnectionInterrupted
from redis.exceptions import RedisError

from .exceptions import IdempotencyConflict, QueueUnavailable, ResultBackendUnavailable
from .prediction_cache import (
    get_cached_prediction as get_cached_prediction_sync,
)
from .prediction_cache import (
    set_cached_prediction as set_cached_prediction_sync,
)
from .tasks import classify_abstract_async

logger = logging.getLogger(__name__)

ASYNC_TASK_MARKER_PREFIX = "classifier_api:async_task"
IDEMPOTENCY_KEY_PREFIX = "classifier_api:idempotency"


async def get_cached_prediction(cache_key: str) -> dict[str, Any] | None:
    return await sync_to_async(get_cached_prediction_sync, thread_sensitive=True)(
        cache_key
    )


async def set_cached_prediction(cache_key: str, prediction: dict[str, Any]) -> None:
    await sync_to_async(set_cached_prediction_sync, thread_sensitive=True)(
        cache_key, prediction
    )


async def reserve_idempotency_key(
    owner_id: str, idempotency_key: UUID, abstract: str
) -> tuple[str, bool]:
    task_id = str(uuid4())
    marker_key = f"{ASYNC_TASK_MARKER_PREFIX}:{task_id}"
    await _register_task_owner(marker_key, owner_id)

    idempotency_cache_key = _idempotency_cache_key(owner_id, idempotency_key)
    created = False
    value = {
        "abstract_hash": hashlib.sha256(abstract.encode("utf-8")).hexdigest(),
        "task_id": task_id,
    }
    try:
        created = await sync_to_async(cache.add, thread_sensitive=True)(
            idempotency_cache_key,
            json.dumps(value),
            timeout=settings.CELERY_RESULT_EXPIRES,
        )
        if created:
            return task_id, True

        existing_value = await sync_to_async(cache.get, thread_sensitive=True)(
            idempotency_cache_key
        )
        existing = json.loads(existing_value) if existing_value else None
        if not isinstance(existing, dict) or not isinstance(
            existing.get("task_id"), str
        ):
            raise ResultBackendUnavailable()
        if existing.get("abstract_hash") != value["abstract_hash"]:
            raise IdempotencyConflict()
        return existing["task_id"], False
    except (ConnectionInterrupted, RedisError, ValueError, TypeError) as exc:
        raise ResultBackendUnavailable() from exc
    finally:
        if not created:
            await _remove_task_marker(marker_key)


async def release_idempotency_key(owner_id: str, idempotency_key: UUID) -> None:
    try:
        await sync_to_async(cache.delete, thread_sensitive=True)(
            _idempotency_cache_key(owner_id, idempotency_key)
        )
    except (ConnectionInterrupted, RedisError):
        logger.exception("Could not remove async idempotency key")


def _idempotency_cache_key(owner_id: str, idempotency_key: UUID) -> str:
    digest = hashlib.sha256(
        f"{owner_id}:{idempotency_key}".encode("utf-8")
    ).hexdigest()
    return f"{IDEMPOTENCY_KEY_PREFIX}:{digest}"


async def enqueue_prediction(
    abstract: str,
    owner_id: str,
    task_id: str | None = None,
    register_owner: bool = True,
) -> str:
    task_id = task_id or str(uuid4())
    marker_key = f"{ASYNC_TASK_MARKER_PREFIX}:{task_id}"
    if register_owner:
        await _register_task_owner(marker_key, owner_id)
    try:
        enqueue = sync_to_async(
            classify_abstract_async.apply_async, thread_sensitive=False
        )
        await enqueue(args=[abstract], task_id=task_id)
    except (OperationalError, RedisError) as exc:
        await _remove_task_marker(marker_key)
        raise QueueUnavailable() from exc
    except Exception:
        await _remove_task_marker(marker_key)
        raise
    return task_id


async def complete_cached_prediction(
    prediction: dict[str, Any],
    owner_id: str,
    task_id: str | None = None,
    register_owner: bool = True,
) -> str:
    task_id = task_id or str(uuid4())
    marker_key = f"{ASYNC_TASK_MARKER_PREFIX}:{task_id}"
    if register_owner:
        await _register_task_owner(marker_key, owner_id)
    try:
        mark_as_done = sync_to_async(
            current_app.backend.mark_as_done, thread_sensitive=False
        )
        await mark_as_done(task_id, {**prediction, "cached": True})
    except (OperationalError, ConnectionInterrupted, RedisError) as exc:
        await _remove_task_marker(marker_key)
        raise ResultBackendUnavailable() from exc
    except Exception:
        await _remove_task_marker(marker_key)
        raise
    return task_id


async def _register_task_owner(marker_key: str, owner_id: str) -> None:
    try:
        marker_created = await sync_to_async(cache.add, thread_sensitive=True)(
            marker_key, str(owner_id), timeout=settings.CELERY_RESULT_EXPIRES
        )
    except (ConnectionInterrupted, RedisError) as exc:
        raise ResultBackendUnavailable() from exc
    if not marker_created:
        raise ResultBackendUnavailable()


async def _remove_task_marker(marker_key: str) -> None:
    try:
        await sync_to_async(cache.delete, thread_sensitive=True)(marker_key)
    except (ConnectionInterrupted, RedisError):
        logger.exception("Could not remove async task marker %s", marker_key)


async def task_was_submitted(task_id: str, owner_id: str) -> bool:
    marker = await sync_to_async(cache.get, thread_sensitive=True)(
        f"{ASYNC_TASK_MARKER_PREFIX}:{task_id}"
    )
    return marker is not None and str(marker) == str(owner_id)


async def get_task_result(task_id: str) -> tuple[str, Any]:
    return await sync_to_async(_get_task_result, thread_sensitive=False)(task_id)


def _get_task_result(task_id: str) -> tuple[str, Any]:
    result = AsyncResult(task_id)
    return result.state, result.result if result.ready() else None
