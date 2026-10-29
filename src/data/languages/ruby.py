"""Ruby language handler using Tree-sitter."""

from __future__ import annotations

import re
from typing import Any

from src.data.languages.base import BaseLanguageHandler

_RUBY_DEF_RE = re.compile(r"\bdef\s+[a-zA-Z0-9_!?]+")


class RubyLanguageHandler(BaseLanguageHandler):
    """Language handler for Ruby."""

    name = "ruby"
    tree_sitter_name = "ruby"
    file_extensions = [".rb", ".rake", ".gemspec"]
    function_node_types = {"method", "singleton_method"}
    comment_prefixes = ("#", "=begin")

    def _fallback_has_function_definition(self, code: str) -> bool:
        """Fallback check for def keyword in Ruby."""
        return bool(_RUBY_DEF_RE.search(code))

    def extract_docstring_and_code(self, item: dict[str, Any]) -> tuple[str, str]:
        """Extract (RDoc/YARD, code) pair."""
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
