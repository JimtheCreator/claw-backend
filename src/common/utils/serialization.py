from typing import Any

def serialize_result(result: Any) -> Any:
    """
    Converts a pydantic result entity (or anything else) to a plain,
    JSON-safe structure for Redis pub/sub / SSE payloads.

    Tries pydantic v2's model_dump(mode="json") first (handles datetime,
    nested models, etc. natively), falls back to v1's .dict(), and passes
    through anything that's already plain (dict, None, primitives)
    unchanged. Explicit conversion here rather than relying on a generic
    json.dumps(default=...) fallback chain - a pydantic model's __dict__
    isn't a reliable, version-stable way to get a clean nested
    representation, and guessing wrong silently produces a payload that
    looks fine until a client tries to read a field that got mangled.
    """
    if result is None:
        return None
    if hasattr(result, "model_dump"):
        return result.model_dump(mode="json")
    if hasattr(result, "dict"):
        return result.dict()
    return result