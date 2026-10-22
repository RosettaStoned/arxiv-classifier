from rest_framework_simplejwt.views import TokenObtainPairView, TokenRefreshView


class ScopedTokenObtainPairView(TokenObtainPairView):
    throttle_scope = "auth_token"


class ScopedTokenRefreshView(TokenRefreshView):
    throttle_scope = "auth_refresh"
