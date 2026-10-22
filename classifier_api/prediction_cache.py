import hashlib
from typing import Any

from django.conf import settings
from django.core.cache import cache


def classification_cache_key(abstract: str) -> str:
    digest = hashlib.sha256(abstract.lower().encode("utf-8")).hexdigest()
    return "classify:" + digest


def get_cached_prediction(cache_key: str) -> dict[str, Any] | None:
    return cache.get(cache_key)


def set_cached_prediction(cache_key: str, prediction: dict[str, Any]) -> None:
    cache.set(cache_key, prediction, timeout=settings.CLASSIFICATION_CACHE_TTL)
