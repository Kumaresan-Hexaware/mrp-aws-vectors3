from __future__ import annotations

from typing import Any, Dict
import json
import math
from decimal import Decimal


def json_safe(obj: Any) -> Any:
    """Convert arbitrary objects into JSON-serializable structures (best-effort)."""
    try:
        json.dumps(obj)
        return obj
    except Exception:
        pass

    # Common special cases
    if hasattr(obj, "to_dict"):
        try:
            return obj.to_dict()  # type: ignore[attr-defined]
        except Exception:
            pass

    if hasattr(obj, "dict"):
        try:
            return obj.dict()  # type: ignore[attr-defined]
        except Exception:
            pass

    if hasattr(obj, "__dict__"):
        try:
            return dict(obj.__dict__)
        except Exception:
            pass

    return str(obj)


def to_ddb_safe(obj: Any) -> Any:
    """Convert Python objects into DynamoDB-safe values.

    - DynamoDB (boto3 serializer) does not accept float -> convert to Decimal
    - Recursively process dicts/lists
    - NaN/Inf -> None
    """
    if obj is None:
        return None
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, int):
        return obj
    if isinstance(obj, float):
        if not math.isfinite(obj):
            return None
        # Use str() to avoid binary float representation issues.
        return Decimal(str(obj))
    if isinstance(obj, Decimal):
        return obj
    if isinstance(obj, str):
        return obj
    if isinstance(obj, list):
        return [to_ddb_safe(v) for v in obj]
    if isinstance(obj, tuple):
        return [to_ddb_safe(v) for v in obj]
    if isinstance(obj, dict):
        return {str(k): to_ddb_safe(v) for k, v in obj.items()}
    # Fallback to json_safe first, then recurse.
    return to_ddb_safe(json_safe(obj))


def scrub_large_strings(payload: Any, max_len: int = 20_000) -> Any:
    """Trim very large strings recursively to keep DynamoDB items within size limits.

    Accepts dict/list/str/any and returns the same general shape.
    """
    if payload is None:
        return None

    if isinstance(payload, str):
        return payload if len(payload) <= max_len else payload[:max_len] + "...<trimmed>"

    if isinstance(payload, list):
        return [scrub_large_strings(v, max_len=max_len) for v in payload]

    if isinstance(payload, dict):
        out: Dict[str, Any] = {}
        for k, v in payload.items():
            out[str(k)] = scrub_large_strings(v, max_len=max_len)
        return out

    return payload
