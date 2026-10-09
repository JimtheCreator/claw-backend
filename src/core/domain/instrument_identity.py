"""Safe exchange symbol vocabulary, shared by URLs, payloads and storage.

Unicode letters/digits are legitimate exchange identifiers. Whitespace, control
characters, punctuation and ASCII lowercase are not canonical ticker symbols.
The pattern also works with Pydantic's Rust regex engine (no lookarounds).
"""
import re

SYMBOL_PATTERN = r'^(?:[A-Z0-9]|[^\x00-\x7F\W_]){2,30}$'
SYMBOL = re.compile(SYMBOL_PATTERN)
