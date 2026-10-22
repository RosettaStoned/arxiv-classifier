from unittest.mock import AsyncMock, patch

from asgiref.sync import sync_to_async
from django.core.cache import cache
from django.test import AsyncClient, Client, SimpleTestCase, override_settings

from classifier_api.prediction_cache import classification_cache_key
from classifier_api.tests._test_support import AllowAnonymousApiTests

ABSTRACTS = [
    "We study biological systems and cellular mechanisms in detail.",
    "We study particle interactions and energy fields in detail.",
]


@override_settings(REDIS_URL="")
@override_settings(
    CACHES={
        "default": {
            "BACKEND": "django.core.cache.backends.locmem.LocMemCache",
            "LOCATION": "classifier-api-integration",
        }
    }
)
@override_settings(
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
    }
)
class ClassificationIntegrationTests(AllowAnonymousApiTests, SimpleTestCase):
    async def asyncSetUp(self):
        await sync_to_async(cache.clear, thread_sensitive=True)()

    def test_api_documentation_includes_both_sync_endpoints(self):
        client = Client()
        docs = client.get("/api/docs/")
        schema = client.get("/api/schema/")

        self.assertEqual(docs.status_code, 200)
        self.assertContains(docs, 'id="swagger-ui"')
        self.assertEqual(schema.status_code, 200)
        self.assertContains(schema, "/api/classify/")
        self.assertContains(schema, "/api/classify/batch/")
        self.assertContains(schema, "/api/classify/async/")
        self.assertContains(schema, "/api/classify/async/{task_id}/")

    @patch("classifier_api.views.run_prediction_async", new_callable=AsyncMock)
    async def test_batch_api_response_matches_model_batch(self, run_prediction):
        expected = [
            {
                "category": "q-bio.BM",
                "confidence": 0.87,
                "categories": [{"category": "q-bio.BM", "confidence": 0.87}],
            },
            {
                "category": "physics.gen-ph",
                "confidence": 0.82,
                "categories": [{"category": "physics.gen-ph", "confidence": 0.82}],
            },
        ]
        run_prediction.return_value = expected
        response = await AsyncClient().post(
            "/api/classify/batch/",
            {"abstracts": ABSTRACTS},
            content_type="application/json",
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {
                "predictions": [
                    {**expected[0], "cached": False},
                    {**expected[1], "cached": False},
                ]
            },
        )
        run_prediction.assert_awaited_once_with(ABSTRACTS)

    @patch("classifier_api.views.run_prediction_async", new_callable=AsyncMock)
    @patch("classifier_api.views.get_cached_prediction")
    async def test_single_api_uses_cached_prediction(
        self, get_cached_prediction, run_prediction
    ):
        get_cached_prediction.return_value = {
            "category": "q-bio.BM",
            "confidence": 0.87,
            "categories": [{"category": "q-bio.BM", "confidence": 0.87}],
        }
        response = await AsyncClient().post(
            "/api/classify/",
            {"abstract": ABSTRACTS[0]},
            content_type="application/json",
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {
                "category": "q-bio.BM",
                "confidence": 0.87,
                "categories": [{"category": "q-bio.BM", "confidence": 0.87}],
                "cached": True,
            },
        )
        run_prediction.assert_not_awaited()

    @patch("classifier_api.views.run_prediction_async", new_callable=AsyncMock)
    @patch("classifier_api.views.set_cached_prediction")
    @patch("classifier_api.views.get_cached_prediction")
    async def test_batch_api_does_not_do_per_abstract_cache_lookups(
        self, get_cached_prediction, set_cached_prediction, run_prediction
    ):
        run_prediction.return_value = [
            {
                "category": "q-bio.BM",
                "confidence": 0.87,
                "categories": [{"category": "q-bio.BM", "confidence": 0.87}],
            },
            {
                "category": "physics.gen-ph",
                "confidence": 0.82,
                "categories": [{"category": "physics.gen-ph", "confidence": 0.82}],
            },
        ]
        response = await AsyncClient().post(
            "/api/classify/batch/",
            {"abstracts": ABSTRACTS},
            content_type="application/json",
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json()["predictions"],
            [
                {
                    "category": "q-bio.BM",
                    "confidence": 0.87,
                    "categories": [{"category": "q-bio.BM", "confidence": 0.87}],
                    "cached": False,
                },
                {
                    "category": "physics.gen-ph",
                    "confidence": 0.82,
                    "categories": [{"category": "physics.gen-ph", "confidence": 0.82}],
                    "cached": False,
                },
            ],
        )
        run_prediction.assert_awaited_once_with(ABSTRACTS)
        get_cached_prediction.assert_not_awaited()
        set_cached_prediction.assert_not_awaited()

    @patch("classifier_api.views.run_prediction_async", new_callable=AsyncMock)
    async def test_single_api_stores_prediction_in_django_cache(self, run_prediction):
        prediction = {
            "category": "q-bio.BM",
            "confidence": 0.87,
            "categories": [{"category": "q-bio.BM", "confidence": 0.87}],
        }
        run_prediction.return_value = [prediction]
        client = AsyncClient()

        first_response = await client.post(
            "/api/classify/",
            {"abstract": ABSTRACTS[0]},
            content_type="application/json",
        )
        second_response = await client.post(
            "/api/classify/",
            {"abstract": ABSTRACTS[0]},
            content_type="application/json",
        )

        self.assertEqual(first_response.status_code, 200)
        self.assertEqual(second_response.status_code, 200)
        self.assertEqual(second_response.json(), {**prediction, "cached": True})
        run_prediction.assert_awaited_once_with([ABSTRACTS[0]])
        self.assertEqual(
            await sync_to_async(cache.get, thread_sensitive=True)(
                classification_cache_key(ABSTRACTS[0])
            ),
            {**prediction, "cached": False},
        )
