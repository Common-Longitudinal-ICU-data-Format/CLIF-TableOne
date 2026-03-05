"""Simple in-memory session store for single-user local tool."""

_store: dict = {
    "analyzed_tables": {},
    "cache_metadata": {},
    "config": None,
    "hosp_years": None,
}


def get_store() -> dict:
    """Return the full session store."""
    return _store


def get(key: str, default=None):
    """Get a value from the session store."""
    return _store.get(key, default)


def set(key: str, value) -> None:
    """Set a value in the session store."""
    _store[key] = value


def delete(key: str) -> None:
    """Delete a key from the session store."""
    _store.pop(key, None)


def clear() -> None:
    """Clear all session data and reset to defaults."""
    _store.clear()
    _store.update({
        "analyzed_tables": {},
        "cache_metadata": {},
        "config": None,
        "hosp_years": None,
    })
