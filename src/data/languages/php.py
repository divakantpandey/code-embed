"""PHP language handler using Tree-sitter."""

from __future__ import annotations

import re
from typing import Any

from src.data.languages.base import BaseLanguageHandler

_PHP_FUNC_RE = re.compile(r"\bfunction\s+[a-zA-Z0-9_]+")


class PhpLanguageHandler(BaseLanguageHandler):
    """Language handler for PHP."""

    name = "php"
    tree_sitter_name = "php"
    file_extensions = [".php", ".phtml"]
    function_node_types = {"function_definition", "method_declaration"}
    comment_prefixes = ("//", "#", "/*", "*")

    def _fallback_has_function_definition(self, code: str) -> bool:
        """Fallback check for function keyword in PHP."""
        return bool(_PHP_FUNC_RE.search(code))

    def extract_docstring_and_code(self, item: dict[str, Any]) -> tuple[str, str]:
        """Extract (PHPDoc, code) pair."""
        docstring = (
            item.get("func_documentation_string")
            or item.get("docstring")
            or item.get("query")
            or ""
        ).strip()
        code = (
            item.get("func_code_string") or item.get("code") or item.get("function") or ""
        ).strip()
        return docstring, code
