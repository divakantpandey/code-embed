"""Python language handler using Tree-sitter and ast.parse fallback."""

from __future__ import annotations

import ast
from typing import Any

from src.data.languages.base import BaseLanguageHandler


class PythonLanguageHandler(BaseLanguageHandler):
    """Language handler for Python."""

    name = "python"
    tree_sitter_name = "python"
    file_extensions = [".py", ".pyw", ".pyi"]

    def _fallback_syntax_check(self, code: str) -> bool:
        """Validate syntax via standard library ast.parse."""
        try:
            ast.parse(code)
            return True
        except SyntaxError:
            return False

    def extract_docstring_and_code(self, item: dict[str, Any]) -> tuple[str, str]:
        """Extract (docstring, code) from CodeSearchNet or generic dataset row."""
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
