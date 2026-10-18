"""Pluggable multi-language parsing and filtering support for CodeEmbed."""

from __future__ import annotations

from src.data.languages.base import BaseLanguageHandler
from src.data.languages.registry import LanguageRegistry

__all__ = ["BaseLanguageHandler", "LanguageRegistry"]
