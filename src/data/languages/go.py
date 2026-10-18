"""Go language handler using Tree-sitter."""

from __future__ import annotations

from typing import Any

from src.data.languages.base import BaseLanguageHandler


class GoLanguageHandler(BaseLanguageHandler):
    """Language handler for Go."""

    name = "go"
    tree_sitter_name = "go"
    file_extensions = [".go"]

    def extract_docstring_and_code(self, item: dict[str, Any]) -> tuple[str, str]:
        """Extract (godoc, code) pair."""
        docstring = (
            item.get("func_documentation_string")
            or item.get("docstring")
            or item.get("query")
            or ""
        ).strip()
        code = (
            item.get("func_code_string")
            or item.get("code")
            or item.get("function")
            or ""
        ).strip()
        return docstring, code
