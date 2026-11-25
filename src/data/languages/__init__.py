"""Pluggable multi-language parsing and filtering support for CodeEmbed."""

from __future__ import annotations

from src.data.languages.base import BaseLanguageHandler
from src.data.languages.go import GoLanguageHandler
from src.data.languages.java import JavaLanguageHandler
from src.data.languages.javascript import JavaScriptLanguageHandler
from src.data.languages.php import PhpLanguageHandler
from src.data.languages.python import PythonLanguageHandler
from src.data.languages.registry import LanguageRegistry
from src.data.languages.ruby import RubyLanguageHandler

__all__ = [
    "BaseLanguageHandler",
    "GoLanguageHandler",
    "JavaLanguageHandler",
    "JavaScriptLanguageHandler",
    "LanguageRegistry",
    "PhpLanguageHandler",
    "PythonLanguageHandler",
    "RubyLanguageHandler",
]
