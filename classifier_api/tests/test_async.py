import asyncio
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from types import SimpleNamespace
from unittest.mock import Mock, patch
from uuid import UUID, uuid4

from celery.exceptions import OperationalError
from django.conf import settings
from django.core.cache import cache
from django.test import AsyncClient, SimpleTestCase, override_settings
from django_redis.exceptions import ConnectionInterrupted
from rest_framework import serializers
from rest_framework.request import Request
from rest_framework.test import APIRequestFactory, force_authenticate

from classifier_api.exceptions import QueueUnavailable
from classifier_api.helpers import (
    ASYNC_TASK_MARKER_PREFIX,
    enqueue_prediction,
    reserve_idempotency_key,
)
from classifier_api.serializers import IdempotencyKeyMixin
from classifier_api.tasks import classify_abstract_async, process_prediction_batch
from classifier_api.tests._test_support import AllowAnonymousApiTests
from classifier_api.throttles import (
    SLIDING_WINDOW_SCRIPT,
    ActorScopedRateThrottle,
    RateLimitUnavailable,
)
from classifier_api.views import ClassifyAsyncView

LOCAL_CACHE = {
    "default": {
        "BACKEND": "django.core.cache.backends.locmem.LocMemCache",
        "LOCATION": "classifier-api-async-tests",
    }
}
REDIS_CACHE = {
    "default": {
        "BACKEND": "django_redis.cache.RedisCache",
        "LOCATION": "redis://redis:6379/0",
    }
}

TEST_REST_FRAMEWORK = {
    "DEFAULT_SCHEMA_CLASS": "drf_spectacular.openapi.AutoSchema",
    "DEFAULT_AUTHENTICATION_CLASSES": [],
    "DEFAULT_PERMISSION_CLASSES": ["rest_framework.permissions.AllowAny"],
    "DEFAULT_THROTTLE_CLASSES": ["classifier_api.throttles.ActorScopedRateThrottle"],
    "DEFAULT_THROTTLE_RATES": {
        "classify_single": "600/min",
        "classify_batch": "300/min",
        "classify_async": "600/min",
        "classify_async_status": "1200/min",
        "auth_token": "10/min",
        "auth_refresh": "60/min",
    },
}


@override_settings(CACHES=LOCAL_CACHE, REST_FRAMEWORK=TEST_REST_FRAMEWORK)
class AsyncApiUnitTests(AllowAnonymousApiTests, SimpleTestCase):
    async def asyncSetUp(self):
        cache.clear()

    def test_idempotency_key_mixin_adds_validated_uuid_to_data(self):
        class ExampleSerializer(IdempotencyKeyMixin, serializers.Serializer):
            value = serializers.CharField()

        key = uuid4()
        serializer = ExampleSerializer(
            data={"value": "example"},
            context={"idempotency_key": str(key)},
        )

        self.assertTrue(serializer.is_valid(), serializer.errors)
        self.assertEqual(serializer.validated_data["idempotency_key"], key)

    @patch("classifier_api.views.enqueue_prediction")
    async def test_accepts_one_abstract_and_returns_task_id(self, enqueue):
        task_id = str(uuid4())
        enqueue.return_value = task_id

        response = await AsyncClient().post(
            "/api/classify/async/",
            {"abstract": "A sufficiently long research abstract."},
            content_type="application/json",
        )

        self.assertEqual(response.status_code, 202)
        self.assertEqual(response.json()["task_id"], task_id)
        self.assertEqual(response.json()["status"], "PENDING")
        UUID(response.json()["task_id"])
        self.assertTrue(response.json()["status_url"].endswith(f"/{task_id}/"))
        enqueue.assert_called_once()

    @patch("classifier_api.views.enqueue_prediction")
    async def test_idempotent_retry_returns_original_task_without_requeue(self, enqueue):
        enqueue.side_effect = lambda *args, **kwargs: kwargs["task_id"]
        client = AsyncClient()
        headers = {"X-Idempotency-Key": str(uuid4())}
        payload = {"abstract": "A sufficiently long research abstract."}

        first = await client.post(
            "/api/classify/async/",
            payload,
            content_type="application/json",
            headers=headers,
        )
        retry = await client.post(
            "/api/classify/async/",
            payload,
            content_type="application/json",
            headers=headers,
        )

        task_id = enqueue.call_args.kwargs["task_id"]
        self.assertEqual(first.status_code, 202)
        self.assertEqual(retry.status_code, 202)
        self.assertEqual(first.json()["task_id"], task_id)
        self.assertEqual(retry.json()["task_id"], task_id)
        enqueue.assert_called_once()

    @patch("classifier_api.views.enqueue_prediction")
    async def test_idempotency_key_rejects_a_different_abstract(self, enqueue):
        enqueue.return_value = str(uuid4())
        client = AsyncClient()
        headers = {"X-Idempotency-Key": str(uuid4())}

        first = await client.post(
            "/api/classify/async/",
            {"abstract": "A sufficiently long research abstract."},
            content_type="application/json",
            headers=headers,
        )
        conflict = await client.post(
            "/api/classify/async/",
            {"abstract": "A different sufficiently long abstract."},
            content_type="application/json",
            headers=headers,
        )

        self.assertEqual(first.status_code, 202)
        self.assertEqual(conflict.status_code, 409)
        enqueue.assert_called_once()

    @patch("classifier_api.views.enqueue_prediction")
    @patch("classifier_api.views.complete_cached_prediction")
    @patch("classifier_api.views.get_cached_prediction")
    async def test_idempotent_cache_hit_uses_one_completed_task(
        self, get_cached_prediction, complete_cached_prediction, enqueue
    ):
        get_cached_prediction.return_value = {
            "category": "cs.AI",
            "confidence": 0.9,
            "categories": [{"category": "cs.AI", "confidence": 0.9}],
        }
        complete_cached_prediction.side_effect = (
            lambda *args, **kwargs: kwargs["task_id"]
        )
        client = AsyncClient()
        request = {
            "abstract": "A sufficiently long research abstract.",
        }
        idempotency_key = str(uuid4())

        first = await client.post(
            "/api/classify/async/",
            request,
            content_type="application/json",
            headers={"X-Idempotency-Key": idempotency_key},
        )
        retry = await client.post(
            "/api/classify/async/",
            request,
            content_type="application/json",
            headers={"X-Idempotency-Key": idempotency_key},
        )

        task_id = complete_cached_prediction.call_args.kwargs["task_id"]
        self.assertEqual(first.status_code, 202)
        self.assertEqual(first.json()["status"], "SUCCESS")
        self.assertEqual(retry.json()["task_id"], task_id)
        complete_cached_prediction.assert_awaited_once()
        enqueue.assert_not_called()

    @patch("classifier_api.views.enqueue_prediction")
    async def test_idempotency_reservation_is_released_when_enqueue_fails(self, enqueue):
        async def enqueue_prediction_side_effect(*args, **kwargs):
            if enqueue.call_count == 1:
                raise QueueUnavailable()
            return kwargs["task_id"]

        enqueue.side_effect = enqueue_prediction_side_effect
        client = AsyncClient()
        payload = {"abstract": "A sufficiently long research abstract."}
        headers = {"X-Idempotency-Key": str(uuid4())}

        failed = await client.post(
            "/api/classify/async/",
            payload,
            content_type="application/json",
            headers=headers,
        )
        retried = await client.post(
            "/api/classify/async/",
            payload,
            content_type="application/json",
            headers=headers,
        )

        task_id = enqueue.call_args.kwargs["task_id"]
        self.assertEqual(failed.status_code, 503)
        self.assertEqual(retried.status_code, 202)
        self.assertEqual(retried.json()["task_id"], task_id)
        self.assertEqual(enqueue.call_count, 2)

    async def test_idempotency_key_must_be_canonical_uuid(self):
        payload = {"abstract": "A sufficiently long research abstract."}
        with patch("classifier_api.views.enqueue_prediction") as enqueue:
            client = AsyncClient()
            for invalid_key in (
                "",
                "not-a-uuid",
                "550E8400-E29B-41D4-A716-446655440000",
                "{550e8400-e29b-41d4-a716-446655440000}",
            ):
                response = await client.post(
                    "/api/classify/async/",
                    payload,
                    content_type="application/json",
                    headers={"X-Idempotency-Key": invalid_key},
                )
                self.assertEqual(response.status_code, 400)
        enqueue.assert_not_called()

    async def test_concurrent_idempotent_submissions_reserve_one_task_per_user(self):
        key = str(uuid4())
        abstract = "A sufficiently long research abstract."

        first, second = await asyncio.gather(
            reserve_idempotency_key("user-one", UUID(key), abstract),
            reserve_idempotency_key("user-one", UUID(key), abstract),
        )
        other_user = await reserve_idempotency_key("user-two", UUID(key), abstract)

        self.assertEqual(first[0], second[0])
        self.assertEqual(sum((first[1], second[1])), 1)
        self.assertTrue(other_user[1])
        self.assertNotEqual(first[0], other_user[0])

    @patch("classifier_api.tasks.run_prediction")
    @patch("classifier_api.helpers.AsyncResult")
    @patch("classifier_api.helpers.current_app")
    @patch("classifier_api.views.enqueue_prediction")
    @patch("classifier_api.views.get_cached_prediction")
    async def test_async_cache_hit_returns_task_id_and_pollable_result(
        self,
        get_cached_prediction,
        enqueue,
        celery_app,
        async_result,
        run_prediction,
    ):
        abstract = "A sufficiently long research abstract."
        prediction = {
            "category": "q-bio.BM",
            "confidence": 0.9,
            "categories": [{"category": "q-bio.BM", "confidence": 0.9}],
        }
        completed_results = {}
        get_cached_prediction.return_value = prediction
        celery_app.backend.mark_as_done.side_effect = (
            lambda task_id, result: completed_results.__setitem__(task_id, result)
        )
        async_result.side_effect = lambda task_id: SimpleNamespace(
            state="SUCCESS",
            ready=lambda: task_id in completed_results,
            result=completed_results.get(task_id),
        )
        client = AsyncClient()

        submitted = await client.post(
            "/api/classify/async/",
            {"abstract": abstract},
            content_type="application/json",
        )

        self.assertEqual(submitted.status_code, 202)
        self.assertEqual(submitted.json()["status"], "SUCCESS")
        task_id = submitted.json()["task_id"]
        enqueue.assert_not_awaited()
        run_prediction.assert_not_called()
        celery_app.backend.mark_as_done.assert_called_once_with(
            task_id, {**prediction, "cached": True}
        )

        polled = await client.get(f"/api/classify/async/{task_id}/")
        self.assertEqual(polled.status_code, 200)
        self.assertEqual(
            polled.json()["result"], {**prediction, "cached": True}
        )

    async def test_rejects_missing_short_and_oversized_abstracts_before_enqueue(self):
        client = AsyncClient()
        cases = [
            {},
            {"abstract": "too short"},
            {"abstract": "a" * 20_001},
        ]
        with patch("classifier_api.views.enqueue_prediction") as enqueue:
            for payload in cases:
                response = await client.post(
                    "/api/classify/async/",
                    payload,
                    content_type="application/json",
                )
                self.assertEqual(response.status_code, 400)
            enqueue.assert_not_called()

    async def test_rejects_non_object_and_unknown_fields_before_enqueue(self):
        with patch("classifier_api.views.enqueue_prediction") as enqueue:
            for payload in (
                ["A sufficiently long research abstract."],
                {
                    "abstract": "A sufficiently long research abstract.",
                    "unexpected": "value",
                },
            ):
                response = await AsyncClient().post(
                    "/api/classify/async/",
                    payload,
                    content_type="application/json",
                )
                self.assertEqual(response.status_code, 400)
            enqueue.assert_not_called()

    @patch("classifier_api.views.get_task_result")
    async def test_status_view_returns_completed_prediction(self, get_result):
        task_id = uuid4()
        prediction = {
            "category": "physics.gen-ph",
            "confidence": 0.91,
            "categories": [{"category": "physics.gen-ph", "confidence": 0.91}],
            "cached": False,
        }
        get_result.return_value = "SUCCESS", prediction
        cache.set(f"{ASYNC_TASK_MARKER_PREFIX}:{task_id}", "None", timeout=3600)

        response = await AsyncClient().get(f"/api/classify/async/{task_id}/")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["result"], prediction)
        get_result.assert_called_once_with(str(task_id))

    @patch("classifier_api.views.get_task_result")
    async def test_status_view_returns_202_while_pending(self, get_result):
        task_id = uuid4()
        get_result.return_value = "PENDING", None
        cache.set(f"{ASYNC_TASK_MARKER_PREFIX}:{task_id}", "None", timeout=3600)

        response = await AsyncClient().get(f"/api/classify/async/{task_id}/")

        self.assertEqual(response.status_code, 202)
        self.assertEqual(response.json()["status"], "PENDING")

    @patch("classifier_api.views.get_task_result")
    async def test_status_view_returns_completed_failure(self, get_result):
        task_id = uuid4()
        get_result.return_value = "FAILURE", RuntimeError("inference failed")
        cache.set(f"{ASYNC_TASK_MARKER_PREFIX}:{task_id}", "None", timeout=3600)

        response = await AsyncClient().get(f"/api/classify/async/{task_id}/")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "FAILURE")
        self.assertIn("error", response.json())

    @patch("classifier_api.views.get_task_result")
    async def test_status_view_returns_404_for_unknown_task(self, get_result):
        task_id = uuid4()
        get_result.return_value = "PENDING", None

        response = await AsyncClient().get(f"/api/classify/async/{task_id}/")

        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["detail"], "No task was found with this ID.")

    @patch(
        "classifier_api.views.task_was_submitted",
        side_effect=ConnectionInterrupted("cache unavailable"),
    )
    @patch("classifier_api.views.get_task_result", return_value=("PENDING", None))
    async def test_status_view_returns_503_when_task_tracking_cache_is_unavailable(
        self, get_result, task_was_submitted
    ):
        response = await AsyncClient().get(f"/api/classify/async/{uuid4()}/")

        self.assertEqual(response.status_code, 503)
        self.assertEqual(
            response.json()["detail"], "Prediction results are temporarily unavailable."
        )

    @patch("classifier_api.helpers.classify_abstract_async.apply_async")
    async def test_enqueue_registers_task_before_scheduling(self, apply_async):
        task_id = await enqueue_prediction(
            "A sufficiently long research abstract.", "test-user"
        )

        apply_async.assert_called_once_with(
            args=["A sufficiently long research abstract."], task_id=task_id
        )
        self.assertEqual(cache.get(f"{ASYNC_TASK_MARKER_PREFIX}:{task_id}"), "test-user")

    @patch(
        "classifier_api.helpers.classify_abstract_async.apply_async",
        side_effect=OperationalError("broker unavailable"),
    )
    async def test_enqueue_removes_marker_when_broker_fails(self, apply_async):
        with self.assertRaises(QueueUnavailable):
            await enqueue_prediction(
                "A sufficiently long research abstract.", "test-user"
            )

        task_id = apply_async.call_args.kwargs["task_id"]
        self.assertIsNone(cache.get(f"{ASYNC_TASK_MARKER_PREFIX}:{task_id}"))

    @patch("classifier_api.views.enqueue_prediction")
    @patch("classifier_api.views.get_cached_prediction", return_value=None)
    @patch.dict(
        ActorScopedRateThrottle.THROTTLE_RATES,
        {"classify_async": "1/min", "classify_async_status": "10/min"},
    )
    async def test_submit_scope_throttles_repeated_actor(
        self, get_cached_prediction, enqueue
    ):
        enqueue.return_value = str(uuid4())
        with (
            self.settings(CACHES=REDIS_CACHE),
            patch.object(
                ClassifyAsyncView,
                "throttle_classes",
                [ActorScopedRateThrottle],
            ),
            patch(
                "classifier_api.throttles.get_redis_connection"
            ) as get_redis_connection,
        ):
            get_redis_connection.return_value.eval.side_effect = [
                [1, 0],
                [0, 1000],
            ]
            client = AsyncClient()
            first = await client.post(
                "/api/classify/async/",
                {"abstract": "A sufficiently long research abstract."},
                content_type="application/json",
                headers={"X-Forwarded-For": "192.0.2.110"},
            )
            second = await client.post(
                "/api/classify/async/",
                {"abstract": "Another sufficiently long research abstract."},
                content_type="application/json",
                headers={"X-Forwarded-For": "192.0.2.110"},
            )

        self.assertEqual(first.status_code, 202)
        self.assertEqual(second.status_code, 429)
        self.assertIn("Retry-After", second)
        enqueue.assert_called_once()


@override_settings(CACHES=LOCAL_CACHE)
class ActorScopedThrottleUnitTests(SimpleTestCase):
    def setUp(self):
        cache.clear()
        self.factory = APIRequestFactory()
        self.view = SimpleNamespace(throttle_scope="test_scope")

    @patch.dict(ActorScopedRateThrottle.THROTTLE_RATES, {"test_scope": "2/min"})
    def test_refuses_non_redis_cache_backend(self):
        request = Request(self.factory.get("/", REMOTE_ADDR="192.0.2.20"))

        with self.assertRaises(RateLimitUnavailable):
            ActorScopedRateThrottle().allow_request(request, self.view)

    @override_settings(
        REST_FRAMEWORK={"DEFAULT_THROTTLE_RATES": {"test_scope": "2/min"}},
        CACHES=REDIS_CACHE,
    )
    @patch.dict(ActorScopedRateThrottle.THROTTLE_RATES, {"test_scope": "2/min"})
    @patch("classifier_api.throttles.get_redis_connection")
    def test_scoped_rate_allows_limit_then_rejects_excess(self, get_redis_connection):
        request = Request(self.factory.get("/", HTTP_X_API_KEY="client-one"))
        throttle = ActorScopedRateThrottle()
        get_redis_connection.return_value.eval.side_effect = [
            [1, 0],
            [1, 0],
            [0, 1500],
        ]

        self.assertTrue(throttle.allow_request(request, self.view))
        self.assertTrue(throttle.allow_request(request, self.view))
        self.assertFalse(throttle.allow_request(request, self.view))
        self.assertEqual(throttle.wait(), 1.5)

    @patch.dict(ActorScopedRateThrottle.THROTTLE_RATES, {"test_scope": "5/min"})
    @override_settings(CACHES=REDIS_CACHE)
    @patch("classifier_api.throttles.get_redis_connection")
    def test_concurrent_requests_use_the_same_atomic_redis_script(
        self, get_redis_connection
    ):
        count = 0
        lock = Lock()

        def atomic_eval(*_args):
            nonlocal count
            with lock:
                count += 1
                return [int(count <= 5), 0 if count <= 5 else 1000]

        get_redis_connection.return_value.eval.side_effect = atomic_eval

        def make_request(_):
            request = Request(
                self.factory.get("/", HTTP_X_API_KEY="same-client")
            )
            return ActorScopedRateThrottle().allow_request(request, self.view)

        with ThreadPoolExecutor(max_workers=12) as executor:
            results = list(executor.map(make_request, range(30)))

        self.assertEqual(sum(results), 5)
        self.assertEqual(get_redis_connection.return_value.eval.call_count, 30)
        self.assertTrue(
            all(
                call.args[0] == SLIDING_WINDOW_SCRIPT
                for call in get_redis_connection.return_value.eval.call_args_list
            )
        )

    async def test_task_marker_is_owned_by_the_submitting_user(self):
        from classifier_api.helpers import task_was_submitted

        task_id = str(uuid4())
        cache.set(f"{ASYNC_TASK_MARKER_PREFIX}:{task_id}", "user-1", timeout=3600)

        self.assertTrue(await task_was_submitted(task_id, "user-1"))
        self.assertFalse(await task_was_submitted(task_id, "user-2"))

    def test_actor_key_prefers_authenticated_user_then_ip_not_api_key(self):
        first = Request(self.factory.get("/", HTTP_X_API_KEY="client-one"))
        second = Request(self.factory.get("/", HTTP_X_API_KEY="client-two"))
        throttle = ActorScopedRateThrottle()
        self.view.throttle_scope = "test_scope"
        throttle.scope = self.view.throttle_scope
        first_key = throttle.get_cache_key(first, self.view)
        second_key = throttle.get_cache_key(second, self.view)
        self.assertTrue(first_key.startswith("classifier_api:throttle:{test_scope:"))
        self.assertEqual(first_key, second_key)
        different_ip = Request(
            self.factory.get("/", HTTP_X_API_KEY="client-one", REMOTE_ADDR="192.0.2.99")
        )
        self.assertNotEqual(first_key, throttle.get_cache_key(different_ip, self.view))

        authenticated = self.factory.get("/")
        force_authenticate(
            authenticated, user=SimpleNamespace(pk=42, is_authenticated=True)
        )
        user_request = Request(authenticated)
        another_authenticated = self.factory.get(
            "/", HTTP_X_API_KEY="different-key", REMOTE_ADDR="192.0.2.99"
        )
        force_authenticate(
            another_authenticated,
            user=SimpleNamespace(pk=42, is_authenticated=True),
        )
        self.assertNotEqual(
            throttle.get_cache_key(user_request, self.view),
            throttle.get_cache_key(first, self.view),
        )
        self.assertEqual(
            throttle.get_cache_key(user_request, self.view),
            throttle.get_cache_key(Request(another_authenticated), self.view),
        )

        ip_one = Request(self.factory.get("/", REMOTE_ADDR="192.0.2.10"))
        ip_two = Request(self.factory.get("/", REMOTE_ADDR="192.0.2.11"))
        self.assertNotEqual(
            throttle.get_cache_key(ip_one, self.view),
            throttle.get_cache_key(ip_two, self.view),
        )

    @patch.dict(ActorScopedRateThrottle.THROTTLE_RATES, {"test_scope": "2/min"})
    @override_settings(CACHES=REDIS_CACHE)
    @patch("classifier_api.throttles.time.time")
    @patch("classifier_api.throttles.get_redis_connection")
    def test_sliding_window_limits_requests_across_a_window_boundary(
        self, get_redis_connection, clock
    ):
        request = Request(self.factory.get("/", REMOTE_ADDR="192.0.2.25"))
        throttle = ActorScopedRateThrottle()
        get_redis_connection.return_value.eval.side_effect = [
            [1, 0],
            [1, 0],
            [0, 1000],
        ]

        clock.return_value = 59.9
        self.assertTrue(throttle.allow_request(request, self.view))
        self.assertTrue(throttle.allow_request(request, self.view))

        clock.return_value = 60.1
        self.assertFalse(throttle.allow_request(request, self.view))
        self.assertGreater(throttle.wait(), 0)


@override_settings(CACHES=LOCAL_CACHE)
class CeleryBatchUnitTests(SimpleTestCase):
    def setUp(self):
        cache.clear()

    def test_batch_flush_configuration_comes_from_django_settings(self):
        self.assertEqual(
            classify_abstract_async.flush_every,
            settings.ASYNC_PREDICTION_BATCH_SIZE,
        )
        self.assertEqual(
            classify_abstract_async.flush_interval,
            settings.ASYNC_PREDICTION_BATCH_FLUSH_INTERVAL,
        )

    @patch("classifier_api.tasks.current_app")
    @patch("classifier_api.tasks.run_prediction")
    def test_batch_stores_each_prediction_by_its_task_id(
        self, run_prediction, celery_app
    ):
        backend = Mock()
        celery_app.backend = backend
        batch = [
            SimpleNamespace(id="task-one", args=("first abstract",)),
            SimpleNamespace(id="task-two", args=("second abstract",)),
        ]
        predictions = [
            {
                "category": "q-bio.BM",
                "confidence": 0.9,
                "categories": [{"category": "q-bio.BM", "confidence": 0.9}],
                "cached": False,
            },
            {
                "category": "physics.gen-ph",
                "confidence": 0.8,
                "categories": [{"category": "physics.gen-ph", "confidence": 0.8}],
                "cached": False,
            },
        ]
        run_prediction.return_value = predictions

        process_prediction_batch(batch)

        run_prediction.assert_called_once_with(["first abstract", "second abstract"])
        backend.mark_as_done.assert_any_call("task-one", predictions[0])
        backend.mark_as_done.assert_any_call("task-two", predictions[1])

    @patch("classifier_api.tasks.current_app")
    @patch("classifier_api.tasks.run_prediction", side_effect=TimeoutError)
    def test_model_failure_marks_every_item_failed(self, run_prediction, celery_app):
        backend = Mock()
        celery_app.backend = backend
        batch = [
            SimpleNamespace(id="task-one", args=("first abstract",)),
            SimpleNamespace(id="task-two", args=("second abstract",)),
        ]

        process_prediction_batch(batch)

        self.assertEqual(run_prediction.call_count, 1)
        self.assertEqual(backend.mark_as_failure.call_count, 2)
        self.assertEqual(
            [call.args[0] for call in backend.mark_as_failure.call_args_list],
            ["task-one", "task-two"],
        )

    @patch("classifier_api.tasks.current_app")
    @patch("classifier_api.tasks.run_prediction", side_effect=TimeoutError)
    def test_failure_to_store_one_failure_does_not_skip_remaining_items(
        self, run_prediction, celery_app
    ):
        backend = Mock()
        backend.mark_as_failure.side_effect = [
            RuntimeError("backend unavailable"),
            None,
        ]
        celery_app.backend = backend
        batch = [
            SimpleNamespace(id="task-one", args=("first abstract",)),
            SimpleNamespace(id="task-two", args=("second abstract",)),
        ]

        process_prediction_batch(batch)

        self.assertEqual(backend.mark_as_failure.call_count, 2)
        self.assertEqual(
            [call.args[0] for call in backend.mark_as_failure.call_args_list],
            ["task-one", "task-two"],
        )


@override_settings(CACHES=LOCAL_CACHE, REST_FRAMEWORK=TEST_REST_FRAMEWORK)
class AsyncClassificationIntegrationTests(AllowAnonymousApiTests, SimpleTestCase):
    async def asyncSetUp(self):
        cache.clear()

    @patch("classifier_api.helpers.AsyncResult")
    @patch("classifier_api.tasks.current_app")
    @patch("classifier_api.tasks.run_prediction")
    @patch("classifier_api.helpers.classify_abstract_async.apply_async")
    async def test_submit_run_batch_and_poll_returns_individual_result(
        self, enqueue, run_prediction, celery_app, async_result
    ):
        prediction = {
            "category": "q-bio.BM",
            "confidence": 0.87,
            "categories": [{"category": "q-bio.BM", "confidence": 0.87}],
            "cached": False,
        }
        stored_results = {}
        backend = Mock()
        backend.mark_as_done.side_effect = lambda result_id, value: (
            stored_results.__setitem__(result_id, value)
        )
        celery_app.backend = backend
        async_result.side_effect = lambda result_id: SimpleNamespace(
            state="SUCCESS",
            ready=lambda: result_id in stored_results,
            result=stored_results.get(result_id),
        )

        run_prediction.return_value = [
            {
                "category": "q-bio.BM",
                "confidence": 0.87,
                "categories": [{"category": "q-bio.BM", "confidence": 0.87}],
            }
        ]

        client = AsyncClient()
        idempotency_key = str(uuid4())
        submitted = await client.post(
            "/api/classify/async/",
            {"abstract": "We study biological cells and their behavior."},
            content_type="application/json",
            headers={"X-Idempotency-Key": idempotency_key},
        )
        self.assertEqual(submitted.status_code, 202)
        task_id = submitted.json()["task_id"]
        enqueue.assert_called_once_with(
            args=["We study biological cells and their behavior."],
            task_id=task_id,
        )
        retried = await client.post(
            "/api/classify/async/",
            {"abstract": "We study biological cells and their behavior."},
            content_type="application/json",
            headers={"X-Idempotency-Key": idempotency_key},
        )
        self.assertEqual(retried.status_code, 202)
        self.assertEqual(retried.json()["task_id"], task_id)
        enqueue.assert_called_once()

        process_prediction_batch(
            [
                SimpleNamespace(
                    id=task_id, args=("We study biological cells and their behavior.",)
                )
            ]
        )
        self.assertEqual(stored_results[task_id], prediction)

        polled = await client.get(f"/api/classify/async/{task_id}/")
        self.assertEqual(polled.status_code, 200)
        self.assertEqual(polled.json()["result"], prediction)
        async_result.assert_called_once_with(task_id)
        run_prediction.assert_called_once_with(
            ["We study biological cells and their behavior."]
        )
