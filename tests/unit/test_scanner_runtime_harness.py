from datetime import datetime, timezone

from tests.integration.scanner_runtime_support import stable_window


def test_fixture_window_avoids_coincident_close_boundaries():
    def seconds(hour, minute, second=0):
        return int(datetime(2026, 9, 17, hour, minute, second, tzinfo=timezone.utc).timestamp())
    assert stable_window(seconds(12, 1))[0] == "15m"
    assert stable_window(seconds(12, 58))[0] == "4h"
    assert stable_window(seconds(15, 58))[0] == "1d"
    assert stable_window(seconds(23, 58)) is None
    assert stable_window(seconds(0, 0, 15)) is None
    assert stable_window(seconds(0, 0, 31))[0] == "15m"
