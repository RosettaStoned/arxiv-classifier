from unittest.mock import AsyncMock, patch

from asgiref.sync import sync_to_async
from django.core.cache import cache
from django.test import AsyncClient, SimpleTestCase, override_settings

from classifier_api.exceptions import InferenceOverloaded
from classifier_api.tests._test_support import AllowAnonymousApiTests


@override_settings(
    REDIS_URL="",
    CACHES={
        "default": {
            "BACKEND": "django.core.cache.backends.locmem.LocMemCache",
            "LOCATION": "classifier-api-unit",
        }
    },
    REST_FRAMEWORK={
        "DEFAULT_SCHEMA_CLASS": "drf_spectacular.openapi.AutoSchema",
        "DEFAULT_AUTHENTICATION_CLASSES": [],
        "DEFAULT_PERMISSION_CLASSES": ["rest_framework.permissions.AllowAny"],
        "DEFAULT_THROTTLE_CLASSES": [
            "classifier_api.throttles.ActorScopedRateThrottle"
        ],
        "DEFAULT_THROTTLE_RATES": {
            "classify_single": "600/min",
            "classify_batch": "300/min",
            "classify_async": "600/min",
            "classify_async_status": "1200/min",
        },
    },
)
class ClassificationApiTests(AllowAnonymousApiTests, SimpleTestCase):
    async def asyncSetUp(self):
        await sync_to_async(cache.clear, thread_sensitive=True)()

    async def test_rejects_short_abstract(self):
        response = await AsyncClient().post(
            "/api/classify/", {"abstract": "too short"}, content_type="application/json"
        )
        self.assertEqual(response.status_code, 400)

    async def test_rejects_missing_abstract(self):
        response = await AsyncClient().post(
            "/api/classify/", {}, content_type="application/json"
        )
        self.assertEqual(response.status_code, 400)

    async def test_rejects_abstract_over_maximum_length(self):
        response = await AsyncClient().post(
            "/api/classify/",
            {"abstract": "a" * 20_001},
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 400)

    @patch("classifier_api.views.run_prediction_async", new_callable=AsyncMock)
    async def test_returns_single_prediction(self, run_prediction):
        run_prediction.return_value = [
            {
                "category": "physics.gen-ph",
                "confidence": 0.91,
                "categories": [{"category": "physics.gen-ph", "confidence": 0.91}],
            }
        ]
        response = await AsyncClient().post(
            "/api/classify/",
            {"abstract": "A sufficiently long research abstract."},
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {
                "category": "physics.gen-ph",
                "confidence": 0.91,
                "categories": [{"category": "physics.gen-ph", "confidence": 0.91}],
                "cached": False,
            },
        )
        run_prediction.assert_awaited_once_with(
            ["A sufficiently long research abstract."]
        )

    @patch("classifier_api.views.run_prediction_async", new_callable=AsyncMock)
    async def test_batch_endpoint_preserves_prediction_order(self, run_prediction):
        abstracts = [
            "A sufficiently long biology research abstract.",
            "A sufficiently long physics research abstract.",
        ]
        run_prediction.return_value = [
            {
                "category": "q-bio.BM",
                "confidence": 0.9,
                "categories": [{"category": "q-bio.BM", "confidence": 0.9}],
            },
            {
                "category": "physics.gen-ph",
                "confidence": 0.8,
                "categories": [{"category": "physics.gen-ph", "confidence": 0.8}],
            },
        ]
        response = await AsyncClient().post(
            "/api/classify/batch/",
            {"abstracts": abstracts},
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            [
                item["categories"][0]["category"]
                for item in response.json()["predictions"]
            ],
            ["q-bio.BM", "physics.gen-ph"],
        )
        run_prediction.assert_awaited_once_with(abstracts)

    async def test_batch_endpoint_rejects_too_many_abstracts(self):
        response = await AsyncClient().post(
            "/api/classify/batch/",
            {"abstracts": ["A sufficiently long research abstract."] * 17},
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 400)

    @patch("classifier_api.views.run_prediction_async", new_callable=AsyncMock)
    async def test_model_failure_returns_retry_after(self, run_prediction):
        run_prediction.side_effect = RuntimeError("model failed")
        response = await AsyncClient().post(
            "/api/classify/",
            {"abstract": "A sufficiently long research abstract."},
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response["Retry-After"], "1")

    @patch("classifier_api.views.run_prediction_async", new_callable=AsyncMock)
    async def test_overloaded_inference_returns_retry_after(self, run_prediction):
        run_prediction.side_effect = InferenceOverloaded()
        response = await AsyncClient().post(
            "/api/classify/",
            {"abstract": "A sufficiently long research abstract."},
            content_type="application/json",
        )

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response["Retry-After"], "1")
