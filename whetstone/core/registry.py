from collections.abc import Callable


class Registry[T]:
    def __init__(self) -> None:
        self._items: dict[str, T] = {}

    def register(self, name: str, item: T) -> T:
        """Register ``item`` under ``name`` and return it. Raise a KeyError if ``name`` is already registered."""
        if name in self._items:
            msg = f"Duplicate registry entry: {name}"
            raise KeyError(msg)
        self._items[name] = item
        return item

    def decorator(self, name: str) -> Callable[[T], T]:
        """Return a decorator that registers the decorated object under ``name``."""

        def _register(item: T) -> T:
            return self.register(name, item)

        return _register

    def get(self, name: str) -> T:
        """Look up a registered item by name. Raise a KeyError if ``name`` is unknown; the message lists known entries."""
        try:
            return self._items[name]
        except KeyError as exc:
            known = ", ".join(sorted(self._items)) or "<none>"
            msg = f"Unknown registry entry {name!r}; known entries: {known}"
            raise KeyError(msg) from exc

    def names(self) -> list[str]:
        """Return all registered names, sorted."""
        return sorted(self._items)
