from typing import NotRequired, TypedDict


class CategoryPrediction(TypedDict):
    category: str
    confidence: float


class Prediction(TypedDict):
    category: str
    confidence: float
    categories: list[CategoryPrediction]
    cached: NotRequired[bool]
