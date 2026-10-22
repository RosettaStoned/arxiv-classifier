from typing import Any

from django.conf import settings
from rest_framework import serializers


class IdempotencyKeyMixin:
    def validate(self, attrs: dict[str, Any]) -> dict[str, Any]:
        attrs = super().validate(attrs)
        raw_idempotency_key = self.context.get("idempotency_key")
        if raw_idempotency_key is None:
            return attrs

        try:
            idempotency_key = serializers.UUIDField().run_validation(
                raw_idempotency_key
            )
        except serializers.ValidationError as exc:
            raise serializers.ValidationError(
                {"X-Idempotency-Key": "Must be a canonical UUID."}
            ) from exc
        if str(idempotency_key) != raw_idempotency_key:
            raise serializers.ValidationError(
                {"X-Idempotency-Key": "Must be a canonical UUID."}
            )

        attrs["idempotency_key"] = idempotency_key
        return attrs


class AbstractSerializer(serializers.Serializer):
    abstract = serializers.CharField(
        min_length=20, max_length=20_000, trim_whitespace=True
    )


class AsyncAbstractSerializer(IdempotencyKeyMixin, serializers.Serializer):
    abstract = serializers.CharField(
        min_length=20, max_length=20_000, trim_whitespace=True
    )

    def validate(self, attrs: dict[str, Any]) -> dict[str, Any]:
        attrs = super().validate(attrs)
        extra_fields = set(self.initial_data) - set(self.fields)
        if extra_fields:
            raise serializers.ValidationError(
                {field: "Unexpected field." for field in sorted(extra_fields)}
            )
        return attrs

    def validate_abstract(self, abstract: str) -> str:
        return " ".join(abstract.split())


class AbstractBatchSerializer(serializers.Serializer):
    abstracts = serializers.ListField(
        child=serializers.CharField(
            min_length=20, max_length=20_000, trim_whitespace=True
        ),
        min_length=1,
        max_length=settings.CLASSIFICATION_BATCH_SIZE,
    )

    def validate_abstracts(self, abstracts: list[str]) -> list[str]:
        return [" ".join(abstract.split()) for abstract in abstracts]


class CategoryPredictionSerializer(serializers.Serializer):
    category = serializers.CharField()
    confidence = serializers.FloatField()


class PredictionSerializer(serializers.Serializer):
    category = serializers.CharField()
    confidence = serializers.FloatField()
    categories = CategoryPredictionSerializer(many=True)
    cached = serializers.BooleanField()


class PredictionBatchSerializer(serializers.Serializer):
    predictions = PredictionSerializer(many=True)


class AsyncTaskAcceptedSerializer(serializers.Serializer):
    task_id = serializers.UUIDField()
    status = serializers.CharField()
    status_url = serializers.URLField()


class AsyncTaskStatusSerializer(serializers.Serializer):
    task_id = serializers.UUIDField()
    status = serializers.CharField()
    result = serializers.DictField(required=False)
    error = serializers.CharField(required=False)
