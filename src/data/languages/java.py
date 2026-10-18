"""Java language handler using Tree-sitter."""

from __future__ import annotations

from typing import Any

from src.data.languages.base import BaseLanguageHandler


class JavaLanguageHandler(BaseLanguageHandler):
    """Language handler for Java."""

    name = "java"
    tree_sitter_name = "java"
    file_extensions = [".java"]

    def extract_docstring_and_code(self, item: dict[str, Any]) -> tuple[str, str]:
        """Extract (Javadoc, code) pair."""
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
