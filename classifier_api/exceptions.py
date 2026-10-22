from rest_framework.exceptions import APIException


class QueueUnavailable(APIException):
    status_code = 503
    default_detail = "The prediction queue is temporarily unavailable."
    default_code = "prediction_queue_unavailable"


class IdempotencyConflict(APIException):
    status_code = 409
    default_detail = "This idempotency key was already used for a different abstract."
    default_code = "idempotency_key_conflict"


class ResultBackendUnavailable(APIException):
    status_code = 503
    default_detail = "Prediction results are temporarily unavailable."
    default_code = "result_backend_unavailable"


class InferenceOverloaded(APIException):
    status_code = 503
    default_detail = "Model inference is busy. Please retry shortly."
    default_code = "inference_overloaded"
