"""JavaScript and TypeScript language handler using Tree-sitter."""

from __future__ import annotations

from typing import Any

from src.data.languages.base import BaseLanguageHandler


class JavaScriptLanguageHandler(BaseLanguageHandler):
    """Language handler for JavaScript and TypeScript."""

    name = "javascript"
    tree_sitter_name = "javascript"
    file_extensions = [".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx"]

    def extract_docstring_and_code(self, item: dict[str, Any]) -> tuple[str, str]:
        """Extract (JSDoc, code) pair."""
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
