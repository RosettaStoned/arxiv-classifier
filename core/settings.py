import os
from datetime import timedelta
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

SECRET_KEY = os.environ.get(
    "DJANGO_SECRET_KEY", "local-development-key-not-for-production"
)
DEBUG = os.environ.get("DJANGO_DEBUG", "false").lower() == "true"
ALLOWED_HOSTS = os.environ.get("ALLOWED_HOSTS", "localhost,127.0.0.1,testserver").split(
    ","
)
DATA_UPLOAD_MAX_MEMORY_SIZE = int(
    os.environ.get("DATA_UPLOAD_MAX_MEMORY_SIZE", str(256 * 1024))
)
REDIS_URL = os.environ.get("REDIS_URL", "")
CELERY_BROKER_URL = os.environ.get(
    "CELERY_BROKER_URL", "redis://localhost:6379/1"
)
CELERY_RESULT_BACKEND = os.environ.get(
    "CELERY_RESULT_BACKEND", "redis://localhost:6379/2"
)
CELERY_RESULT_EXPIRES = 3600
CELERY_TASK_DEFAULT_QUEUE = "predictions"
CELERY_TASK_SERIALIZER = "json"
CELERY_RESULT_SERIALIZER = "json"
CELERY_ACCEPT_CONTENT = ["json"]
CELERY_TASK_TRACK_STARTED = True
CELERY_TASK_ACKS_LATE = True
CELERY_TASK_REJECT_ON_WORKER_LOST = True
CELERY_TASK_SOFT_TIME_LIMIT = 15
CELERY_TASK_TIME_LIMIT = 20
CELERY_WORKER_PREFETCH_MULTIPLIER = 0
CELERY_WORKER_MAX_TASKS_PER_CHILD = 500
CELERY_BROKER_TRANSPORT_OPTIONS = {"visibility_timeout": 60}
MODEL_PATH = os.environ.get("MODEL_PATH", "classifier_api/model")
CLASSIFICATION_CACHE_TTL = int(os.environ.get("CLASSIFICATION_CACHE_TTL", "86400"))
CLASSIFICATION_REQUEST_TIMEOUT = float(
    os.environ.get("CLASSIFICATION_REQUEST_TIMEOUT", "30")
)
CLASSIFICATION_BATCH_SIZE = int(os.environ.get("CLASSIFICATION_BATCH_SIZE", "16"))
ASYNC_PREDICTION_BATCH_SIZE = int(
    os.environ.get("ASYNC_PREDICTION_BATCH_SIZE", str(CLASSIFICATION_BATCH_SIZE))
)
ASYNC_PREDICTION_BATCH_FLUSH_INTERVAL = float(
    os.environ.get("ASYNC_PREDICTION_BATCH_FLUSH_INTERVAL", "0.03")
)
CLASSIFICATION_MAX_PENDING = int(os.environ.get("CLASSIFICATION_MAX_PENDING", "2"))
if CLASSIFICATION_MAX_PENDING < 0:
    raise ValueError("CLASSIFICATION_MAX_PENDING must be zero or greater.")
if ASYNC_PREDICTION_BATCH_SIZE < 1:
    raise ValueError("ASYNC_PREDICTION_BATCH_SIZE must be positive.")
if ASYNC_PREDICTION_BATCH_FLUSH_INTERVAL <= 0:
    raise ValueError("ASYNC_PREDICTION_BATCH_FLUSH_INTERVAL must be positive.")
INSTALLED_APPS = [
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "rest_framework",
    "adrf",
    "drf_spectacular",
    "classifier_api",
]

MIDDLEWARE = [
    "django.middleware.common.CommonMiddleware",
]

ROOT_URLCONF = "core.urls"
WSGI_APPLICATION = "core.wsgi.application"
ASGI_APPLICATION = "core.asgi.application"
TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": BASE_DIR / "db.sqlite3",
    }
}

LANGUAGE_CODE = "en-us"
TIME_ZONE = "UTC"
USE_I18N = True
USE_TZ = True

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

REST_FRAMEWORK = {
    "DEFAULT_SCHEMA_CLASS": "drf_spectacular.openapi.AutoSchema",
    "NUM_PROXIES": 1,
    "DEFAULT_AUTHENTICATION_CLASSES": [
        "rest_framework_simplejwt.authentication.JWTAuthentication",
    ],
    "DEFAULT_PERMISSION_CLASSES": [
        "rest_framework.permissions.IsAuthenticated",
    ],
    "DEFAULT_THROTTLE_CLASSES": [
        "classifier_api.throttles.ActorScopedRateThrottle",
    ],
    "DEFAULT_THROTTLE_RATES": {
        "classify_single": "600/min",
        "classify_batch": "300/min",
        "classify_async": "600/min",
        "classify_async_status": "1200/min",
        "auth_token": "10/min",
        "auth_refresh": "60/min",
    },
}

SIMPLE_JWT = {
    "ACCESS_TOKEN_LIFETIME": timedelta(minutes=15),
    "REFRESH_TOKEN_LIFETIME": timedelta(days=1),
    "AUTH_HEADER_TYPES": ("Bearer",),
}

CACHES = (
    {
        "default": {
            "BACKEND": "django_redis.cache.RedisCache",
            "LOCATION": REDIS_URL,
            "OPTIONS": {
                "CLIENT_CLASS": "django_redis.client.DefaultClient",
                "CONNECTION_POOL_KWARGS": {
                    "max_connections": 100,
                    "socket_connect_timeout": 0.2,
                    "socket_timeout": 0.2,
                    "health_check_interval": 30,
                },
            },
        }
    }
    if REDIS_URL
    else {
        "default": {
            "BACKEND": "django.core.cache.backends.locmem.LocMemCache",
            "LOCATION": "classifier-api",
        }
    }
)

SPECTACULAR_SETTINGS = {
    "TITLE": "arXiv Research Classifier API",
    "DESCRIPTION": "Predict arXiv subject codes from an abstract.",
    "VERSION": "1.0.0",
    "SERVE_INCLUDE_SCHEMA": False,
}

LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {"format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"},
    },
    "handlers": {
        "console": {
            "level": "INFO",
            "class": "logging.StreamHandler",
            "formatter": "standard",
        },
    },
    "loggers": {
        "": {
            "handlers": ["console"],
            "level": "INFO",
            "propagate": True,
        },
    },
}
