"""Java language handler using Tree-sitter."""

from __future__ import annotations

import re
from typing import Any

from src.data.languages.base import BaseLanguageHandler

_JAVA_METHOD_RE = re.compile(
    r"\b(public|private|protected|static|final|native|synchronized|abstract|default|\w+)\s+[\w<>\[\],\s]+\s+\w+\s*\([^)]*\)\s*\{"
)


class JavaLanguageHandler(BaseLanguageHandler):
    """Language handler for Java."""

    name = "java"
    tree_sitter_name = "java"
    file_extensions = [".java"]
    function_node_types = {"method_declaration", "constructor_declaration"}
    comment_prefixes = ("//", "/*", "*")

    def _fallback_has_function_definition(self, code: str) -> bool:
        """Fallback check for method definition in Java."""
        return bool(_JAVA_METHOD_RE.search(code)) or ("(" in code and ")" in code and "{" in code)

    def extract_docstring_and_code(self, item: dict[str, Any]) -> tuple[str, str]:
        """Extract (Javadoc, code) pair."""
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
