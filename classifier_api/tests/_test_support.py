from unittest.mock import patch

from classifier_api.views import (
    ClassifyAsyncStatusView,
    ClassifyAsyncView,
    ClassifyBatchView,
    ClassifyView,
)


class AllowAnonymousApiTests:
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        for view_class in (
            ClassifyView,
            ClassifyBatchView,
            ClassifyAsyncView,
            ClassifyAsyncStatusView,
        ):
            for attribute in ("authentication_classes", "permission_classes"):
                patcher = patch.object(view_class, attribute, [])
                patcher.start()
                cls.addClassCleanup(patcher.stop)
            patcher = patch.object(view_class, "throttle_classes", [])
            patcher.start()
            cls.addClassCleanup(patcher.stop)
