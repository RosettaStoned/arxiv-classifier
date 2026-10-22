import hashlib
import logging
import math
import time

from django.conf import settings
from django_redis import get_redis_connection
from django_redis.exceptions import ConnectionInterrupted
from redis.exceptions import RedisError
from rest_framework.exceptions import APIException
from rest_framework.request import Request
from rest_framework.throttling import ScopedRateThrottle

logger = logging.getLogger(__name__)

SLIDING_WINDOW_SCRIPT = """
local window_ms = tonumber(ARGV[1])
local limit = tonumber(ARGV[2])
local elapsed_ms = tonumber(ARGV[3])
local previous_count = tonumber(redis.call("GET", KEYS[1])) or 0
local current_count = tonumber(redis.call("GET", KEYS[2])) or 0
local estimated_count = previous_count * (window_ms - elapsed_ms) / window_ms

if estimated_count + current_count + 1 > limit then
    local retry_ms = window_ms - elapsed_ms
    local remaining_previous = limit - current_count - 1
    if previous_count > 0 and remaining_previous >= 0 then
        local threshold_ms = window_ms * (1 - remaining_previous / previous_count)
        if threshold_ms > elapsed_ms then
            retry_ms = math.min(retry_ms, math.ceil(threshold_ms - elapsed_ms))
        end
    end
    return {0, math.max(1, retry_ms)}
end

local new_count = redis.call("INCR", KEYS[2])
if new_count == 1 then
    redis.call("PEXPIRE", KEYS[2], window_ms * 2)
end
return {1, 0}
"""


class RateLimitUnavailable(APIException):
    status_code = 503
    default_detail = "Rate limiting is temporarily unavailable."
    default_code = "rate_limit_unavailable"


def request_actor(request: Request, throttle: ScopedRateThrottle) -> str:
    user = getattr(request, "user", None)
    if user and user.is_authenticated:
        actor = f"user:{user.pk}"
    else:
        actor = f"ip:{throttle.get_ident(request)}"
    return hashlib.sha256(actor.encode("utf-8")).hexdigest()


class ActorScopedRateThrottle(ScopedRateThrottle):
    cache_key_prefix = "classifier_api:throttle"
    redis_cache_backend = "django_redis.cache.RedisCache"

    def get_cache_key(self, request: Request, view: object) -> str:
        identity = request_actor(request, self)
        return f"{self.cache_key_prefix}:{{{self.scope}:{identity}}}"

    def allow_request(self, request: Request, view: object) -> bool:
        self.scope = getattr(view, "throttle_scope", None)
        if not self.scope:
            return True

        self.rate = self.get_rate()
        self.num_requests, self.duration = self.parse_rate(self.rate)
        now = time.time()
        window = math.floor(now / self.duration)
        elapsed_ms = int((now - window * self.duration) * 1000)
        window_ms = self.duration * 1000
        base_key = self.get_cache_key(request, view)
        previous_key = f"{base_key}:{window - 1}"
        current_key = f"{base_key}:{window}"

        if settings.CACHES["default"]["BACKEND"] != self.redis_cache_backend:
            logger.error("Scoped rate limiting requires the Redis cache backend.")
            raise RateLimitUnavailable()

        try:
            allowed, retry_ms = get_redis_connection("default").eval(
                SLIDING_WINDOW_SCRIPT,
                2,
                previous_key,
                current_key,
                window_ms,
                self.num_requests,
                elapsed_ms,
            )
        except (ConnectionInterrupted, RedisError, ValueError) as exc:
            logger.exception("Could not apply scoped rate limit %s", self.scope)
            raise RateLimitUnavailable() from exc

        self._wait = retry_ms / 1000
        return bool(allowed)

    def wait(self) -> float:
        return self._wait
