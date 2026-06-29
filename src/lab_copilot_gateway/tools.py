"""Curated lab tool registry scaffold."""


def list_tools() -> list[dict[str, object]]:
    """Return currently exposed curated tools.

    The registry intentionally starts empty: exposing real lab tools belongs to
    later chunks after policy, audit, identity mapping, and approval are in place.
    """
    return []
