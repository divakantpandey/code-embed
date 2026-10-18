"""Registry for programming language handlers in CodeEmbed."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.data.languages.base import BaseLanguageHandler

logger = logging.getLogger(__name__)


class LanguageRegistry:
    """Central registry mapping language names and file extensions to handlers."""

    _handlers: dict[str, BaseLanguageHandler] = {}
    _ext_map: dict[str, BaseLanguageHandler] = {}
    _initialized: bool = False

    @classmethod
    def _bootstrap(cls) -> None:
        """Register built-in language handlers."""
        if cls._initialized:
            return

        from src.data.languages.go import GoLanguageHandler
        from src.data.languages.java import JavaLanguageHandler
        from src.data.languages.javascript import JavaScriptLanguageHandler
        from src.data.languages.python import PythonLanguageHandler

        builtins: list[BaseLanguageHandler] = [
            PythonLanguageHandler(),
            JavaScriptLanguageHandler(),
            JavaLanguageHandler(),
            GoLanguageHandler(),
        ]

        for h in builtins:
            cls.register(h)

        # Common aliases
        cls._handlers["js"] = cls._handlers["javascript"]
        cls._handlers["ts"] = cls._handlers["javascript"]
        cls._handlers["typescript"] = cls._handlers["javascript"]
        cls._handlers["py"] = cls._handlers["python"]
        cls._handlers["golang"] = cls._handlers["go"]

        cls._initialized = True

    @classmethod
    def register(cls, handler: BaseLanguageHandler) -> None:
        """Register a language handler instance."""
        cls._handlers[handler.name.lower()] = handler
        for ext in handler.file_extensions:
            cls._ext_map[ext.lower()] = handler

    @classmethod
    def get(cls, name_or_ext: str) -> BaseLanguageHandler:
        """Look up handler by language name (e.g. 'python') or extension (e.g. '.js').

        Falls back to PythonLanguageHandler if unrecognized.
        """
        cls._bootstrap()
        cleaned = name_or_ext.lower().strip()
        if cleaned in cls._handlers:
            return cls._handlers[cleaned]
        if cleaned in cls._ext_map:
            return cls._ext_map[cleaned]
        if not cleaned.startswith(".") and f".{cleaned}" in cls._ext_map:
            return cls._ext_map[f".{cleaned}"]

        # Default fallback: Python
        return cls._handlers["python"]

    @classmethod
    def supported_languages(cls) -> list[str]:
        """Return list of canonical supported language names."""
        cls._bootstrap()
        return ["python", "javascript", "java", "go"]
