from unittest.mock import AsyncMock, patch

from django.contrib.auth import get_user_model
from django.test import Client, TestCase


class JwtAuthenticationTests(TestCase):
    def setUp(self):
        self.user = get_user_model().objects.create_user(
            username="api-client",
            password="safe-test-password",
        )
        self.client = Client()

    def test_prediction_requires_a_valid_jwt(self):
        payload = {"abstract": "A sufficiently long research abstract."}
        unauthorized = self.client.post(
            "/api/classify/", payload, content_type="application/json"
        )
        self.assertEqual(unauthorized.status_code, 401)

        token_response = self.client.post(
            "/api/auth/token/",
            {"username": "api-client", "password": "safe-test-password"},
            content_type="application/json",
        )
        self.assertEqual(token_response.status_code, 200)
        access_token = token_response.json()["access"]

        with patch(
            "classifier_api.views.run_prediction_async",
            new_callable=AsyncMock,
            return_value=[
                {
                    "category": "cs.AI",
                    "confidence": 0.9,
                    "categories": [{"category": "cs.AI", "confidence": 0.9}],
                }
            ],
        ):
            authorized = self.client.post(
                "/api/classify/",
                payload,
                content_type="application/json",
                HTTP_AUTHORIZATION=f"Bearer {access_token}",
            )

        self.assertEqual(authorized.status_code, 200)

    def test_refresh_endpoint_returns_a_new_access_token(self):
        token_response = self.client.post(
            "/api/auth/token/",
            {"username": "api-client", "password": "safe-test-password"},
            content_type="application/json",
        )
        response = self.client.post(
            "/api/auth/token/refresh/",
            {"refresh": token_response.json()["refresh"]},
            content_type="application/json",
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn("access", response.json())
