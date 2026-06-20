import hashlib


def generate_cache_key(
    *parts: str,
) -> str:
    """
    Generate deterministic Redis-safe cache keys.

    Same input -> same key
    Different input -> different key
    """

    raw_key = ":".join(parts)

    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()
