import asyncio
import logging
from typing import Any
from uuid import UUID, uuid4

from adrf.views import APIView
from celery import current_app
from celery.exceptions import OperationalError
from django.conf import settings
from django.core.cache import cache
from django.urls import reverse
from django_redis.exceptions import ConnectionInterrupted
from drf_spectacular.utils import OpenApiResponse, extend_schema
from redis.exceptions import RedisError
from rest_framework import status
from rest_framework.request import Request
from rest_framework.response import Response

from .exceptions import InferenceOverloaded, QueueUnavailable, ResultBackendUnavailable
from .helpers import (
    complete_cached_prediction,
    enqueue_prediction,
    get_cached_prediction,
    get_task_result,
    release_idempotency_key,
    reserve_idempotency_key,
    set_cached_prediction,
    task_was_submitted,
)
from .prediction_cache import classification_cache_key
from .predictor import ensure_model_loaded
from .serializers import (
    AbstractBatchSerializer,
    AbstractSerializer,
    AsyncAbstractSerializer,
    AsyncTaskAcceptedSerializer,
    AsyncTaskStatusSerializer,
    PredictionBatchSerializer,
    PredictionSerializer,
)
from .services import run_prediction_async
from .throttles import ActorScopedRateThrottle
from .types import Prediction

logger = logging.getLogger(__name__)


async def classify_abstracts(abstracts: list[str]) -> list[Prediction] | Response:
    try:
        predictions = await asyncio.wait_for(
            run_prediction_async(abstracts),
            timeout=settings.CLASSIFICATION_REQUEST_TIMEOUT,
        )
    except asyncio.TimeoutError:
        return Response(
            {"error": "Timed out waiting for model inference"},
            status=status.HTTP_504_GATEWAY_TIMEOUT,
        )
    except InferenceOverloaded as exc:
        response = Response(
            {"error": str(exc.detail)}, status=status.HTTP_503_SERVICE_UNAVAILABLE
        )
        response["Retry-After"] = "1"
        return response
    except FileNotFoundError:
        logger.exception("Model files are unavailable")
        return Response(
            {"error": "Model is unavailable"},
            status=status.HTTP_503_SERVICE_UNAVAILABLE,
        )
    except Exception:
        logger.exception("Model inference failed")
        response = Response(
            {"error": "Model inference failed"},
            status=status.HTTP_503_SERVICE_UNAVAILABLE,
        )
        response["Retry-After"] = "1"
        return response

    if len(predictions) != len(abstracts):
        logger.error("Model returned an unexpected number of predictions")
        return Response(
            {"error": "Model inference failed"},
            status=status.HTTP_503_SERVICE_UNAVAILABLE,
        )
    return [{**prediction, "cached": False} for prediction in predictions]


class ClassifyView(APIView):
    throttle_scope = "classify_single"

    @extend_schema(
        request=AbstractSerializer,
        responses={200: PredictionSerializer},
        description="Classify one abstract.",
    )
    async def post(self, request: Request) -> Response:
        serializer = AbstractSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        abstract = " ".join(serializer.validated_data["abstract"].split())
        cache_key = classification_cache_key(abstract)
        try:
            cached = await get_cached_prediction(cache_key)
            if cached:
                return Response({**cached, "cached": True})
        except (ConnectionInterrupted, RedisError, ValueError, TypeError):
            logger.warning("Could not read classification cache")

        result = await classify_abstracts([abstract])
        if isinstance(result, Response):
            return result
        prediction = result[0]
        try:
            await set_cached_prediction(cache_key, prediction)
        except (ConnectionInterrupted, RedisError):
            logger.warning("Could not cache classification result")
        return Response(prediction)


class ClassifyBatchView(APIView):
    throttle_scope = "classify_batch"

    @extend_schema(
        request=AbstractBatchSerializer,
        responses={200: PredictionBatchSerializer},
        description="Classify up to the configured batch size of abstracts.",
    )
    async def post(self, request: Request) -> Response:
        serializer = AbstractBatchSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        results = await classify_abstracts(serializer.validated_data["abstracts"])
        if isinstance(results, Response):
            return results
        return Response({"predictions": results})


class ClassifyAsyncView(APIView):
    throttle_scope = "classify_async"
    throttle_classes = [ActorScopedRateThrottle]

    @extend_schema(
        request=AsyncAbstractSerializer,
        responses={202: AsyncTaskAcceptedSerializer},
        description="Queue one abstract for asynchronous classification.",
    )
    async def post(self, request: Request) -> Response:
        serializer = AsyncAbstractSerializer(
            data=request.data,
            context={"idempotency_key": request.headers.get("X-Idempotency-Key")},
        )
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        abstract = serializer.validated_data["abstract"]
        idempotency_key: UUID | None = serializer.validated_data.get(
            "idempotency_key"
        )
        owner_id = str(request.user.pk)
        task_id = None
        idempotency_reserved = False
        if idempotency_key is not None:
            task_id, idempotency_reserved = await reserve_idempotency_key(
                owner_id, idempotency_key, abstract
            )
            if not idempotency_reserved:
                status_url = request.build_absolute_uri(
                    reverse(
                        "classify-async-status",
                        kwargs={"task_id": UUID(task_id)},
                    )
                )
                return Response(
                    {
                        "task_id": task_id,
                        "status": "PENDING",
                        "status_url": status_url,
                    },
                    status=status.HTTP_202_ACCEPTED,
                )

        try:
            cached = await get_cached_prediction(classification_cache_key(abstract))
        except (ConnectionInterrupted, RedisError, ValueError, TypeError):
            logger.warning("Could not read async classification cache")
            cached = None

        try:
            if cached:
                task_id = await complete_cached_prediction(
                    cached,
                    owner_id,
                    task_id=task_id,
                    register_owner=not idempotency_reserved,
                )
                task_status = "SUCCESS"
            else:
                task_id = await enqueue_prediction(
                    abstract,
                    owner_id,
                    task_id=task_id,
                    register_owner=not idempotency_reserved,
                )
                task_status = "PENDING"
        except QueueUnavailable as exc:
            if idempotency_reserved and idempotency_key is not None:
                await release_idempotency_key(owner_id, idempotency_key)
            response = Response(
                {"error": str(exc.detail)}, status=status.HTTP_503_SERVICE_UNAVAILABLE
            )
            response["Retry-After"] = "1"
            return response
        except Exception:
            if idempotency_reserved and idempotency_key is not None:
                await release_idempotency_key(owner_id, idempotency_key)
            raise

        status_url = request.build_absolute_uri(
            reverse("classify-async-status", kwargs={"task_id": UUID(task_id)})
        )
        return Response(
            {
                "task_id": task_id,
                "status": task_status,
                "status_url": status_url,
            },
            status=status.HTTP_202_ACCEPTED,
        )


class ClassifyAsyncStatusView(APIView):
    throttle_scope = "classify_async_status"
    throttle_classes = [ActorScopedRateThrottle]

    @extend_schema(
        responses={
            200: AsyncTaskStatusSerializer,
            202: AsyncTaskStatusSerializer,
            404: OpenApiResponse(description="No task was found with this ID."),
        },
        description="Check an asynchronous classification task.",
    )
    async def get(self, request: Request, task_id: UUID) -> Response:
        try:
            submitted = await task_was_submitted(
                str(task_id), str(request.user.pk)
            )
        except (ConnectionInterrupted, RedisError) as exc:
            raise ResultBackendUnavailable() from exc
        if not submitted:
            return Response(
                {"detail": "No task was found with this ID."},
                status=status.HTTP_404_NOT_FOUND,
            )

        try:
            state, result = await get_task_result(str(task_id))
        except (OperationalError, RedisError) as exc:
            raise ResultBackendUnavailable() from exc

        payload: dict[str, Any] = {"task_id": str(task_id), "status": state}
        if state == "SUCCESS":
            payload["result"] = result
            return Response(payload)
        if state == "FAILURE":
            payload["error"] = "Prediction failed. Submit the abstract again to retry."
            return Response(payload, status=status.HTTP_200_OK)
        return Response(payload, status=status.HTTP_202_ACCEPTED)


class ModelHealthView(APIView):
    authentication_classes = []
    permission_classes = []

    @extend_schema(exclude=True)
    def get(self, request: Request) -> Response:
        checks: dict[str, str] = {}
        health_checks = (
            ("redis", self._check_redis),
            ("celery", self._check_celery),
            ("model", ensure_model_loaded),
        )

        for name, check in health_checks:
            try:
                check()
                checks[name] = "ok"
            except Exception:
                logger.exception("Health check failed: %s", name)
                checks[name] = "unavailable"

        healthy = all(result == "ok" for result in checks.values())
        return Response(
            {"status": "ok" if healthy else "unavailable", "checks": checks},
            status=(
                status.HTTP_200_OK
                if healthy
                else status.HTTP_503_SERVICE_UNAVAILABLE
            ),
        )

    @staticmethod
    def _check_redis() -> None:
        if (
            settings.CACHES["default"]["BACKEND"]
            != "django_redis.cache.RedisCache"
        ):
            raise ConnectionError("Redis cache is not configured")
        key = f"classifier_api:health:{uuid4()}"
        if not cache.set(key, key, timeout=1) or cache.get(key) != key:
            raise ConnectionError("Redis cache round-trip failed")

    @staticmethod
    def _check_celery() -> None:
        if not current_app.control.ping(timeout=1):
            raise ConnectionError("No Celery worker responded to ping")
