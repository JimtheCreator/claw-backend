import json

import pytest

from infrastructure.database.redis.scanner_payload import encode, decode, PREFIX


def test_candle_payload_is_lossless_compact_and_legacy_readable():
    data = {'chart': [dict(index=i, timestamp=1700000000+i*60,
                          open=1.12345, high=1.12456, low=1.123, close=1.124, volume=0)
                      for i in range(250)], 'matches': [], 'status': 'ready'}
    legacy = json.dumps(data)
    packed = encode(data)
    assert packed.startswith(PREFIX)
    assert len(packed) < len(legacy) / 3
    assert decode(packed) == decode(packed.encode()) == decode(legacy) == data
    assert decode(encode({'status': 'warming'})) == {'status': 'warming'}
    with pytest.raises(ValueError):
        encode({'price': float('nan')})
