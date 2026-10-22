from django.urls import path

from .auth_views import ScopedTokenObtainPairView, ScopedTokenRefreshView
from .views import (
    ClassifyAsyncStatusView,
    ClassifyAsyncView,
    ClassifyBatchView,
    ClassifyView,
    ModelHealthView,
)

urlpatterns = [
    path("auth/token/", ScopedTokenObtainPairView.as_view(), name="token-obtain"),
    path("auth/token/refresh/", ScopedTokenRefreshView.as_view(), name="token-refresh"),
    path("classify/", ClassifyView.as_view(), name="classify-article"),
    path("classify/batch/", ClassifyBatchView.as_view(), name="classify-batch"),
    path("classify/async/", ClassifyAsyncView.as_view(), name="classify-async"),
    path(
        "classify/async/<uuid:task_id>/",
        ClassifyAsyncStatusView.as_view(),
        name="classify-async-status",
    ),
    path("health/", ModelHealthView.as_view(), name="model-health"),
]
