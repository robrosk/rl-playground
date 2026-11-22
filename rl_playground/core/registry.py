"""Name to class lookup (Registry pattern).

Entries are objects or lazy ``"package.module:Attr"`` strings, so a heavy module such as one
that imports torch loads only when someone asks for it by name.
"""

from __future__ import annotations

import importlib
from typing import Any, Callable, Dict, List


class Registry:
    def __init__(self, kind: str):
        self.kind = kind
        self._entries: Dict[str, Any] = {}

    def add(self, name: str, target: Any) -> None:
        if name in self._entries:
            raise KeyError(f"{self.kind} {name!r} is already registered")
        self._entries[name] = target

    def register(self, name: str) -> Callable[[Any], Any]:
        """Decorator form of ``add``."""

        def decorate(obj):
            self.add(name, obj)
            return obj

        return decorate

    def get(self, name: str) -> Any:
        try:
            target = self._entries[name]
        except KeyError:
            known = ", ".join(self.names()) or "none"
            raise KeyError(f"unknown {self.kind} {name!r}; known: {known}") from None
        if isinstance(target, str):
            module_name, _, attr = target.partition(":")
            target = getattr(importlib.import_module(module_name), attr)
            self._entries[name] = target
        return target

    def make(self, name: str, *args, **kwargs) -> Any:
        return self.get(name)(*args, **kwargs)

    def names(self) -> List[str]:
        return sorted(self._entries)

    def __contains__(self, name: str) -> bool:
        return name in self._entries
