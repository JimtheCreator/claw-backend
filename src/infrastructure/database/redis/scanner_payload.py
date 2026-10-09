"""Compact internal scanner data; retain reads of existing JSON cache entries."""
import base64
import json
import zlib

PREFIX = 'zjson1:'


def encode(value):
    raw = json.dumps(value, allow_nan=False, separators=(',', ':'))
    if len(raw) < 2048:
        return raw
    compressed = PREFIX + base64.b64encode(zlib.compress(raw.encode(), level=1)).decode('ascii')
    return compressed if len(compressed) < len(raw) else raw


def decode(raw):
    if isinstance(raw, bytes):
        raw = raw.decode('utf-8')
    if raw.startswith(PREFIX):
        raw = zlib.decompress(base64.b64decode(raw[len(PREFIX):], validate=True))
    return json.loads(raw)
