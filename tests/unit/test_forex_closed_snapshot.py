from datetime import datetime, timezone
from core.scanner.market_sessions import closed_session_snapshot


def test_closed_session_retains_latest_results_without_hiding_stale_input():
    now=datetime(2026,10,3,12,tzinfo=timezone.utc)
    snapshot=dict(market='forex',interval='15m',data_as_of='2026-10-02T21:00:00+00:00',fresh_until='2026-10-02T21:15:05+00:00')
    result=closed_session_snapshot(snapshot,now)
    assert result['fresh_until']=='2026-10-04T21:15:05+00:00'
    assert result['session']['market_state']=='closed'
    stale=dict(snapshot,data_as_of='2026-10-02T20:45:00+00:00')
    assert closed_session_snapshot(stale,now)==stale
    crypto=dict(snapshot,market='spot')
    assert closed_session_snapshot(crypto,now)==crypto
    assert closed_session_snapshot(snapshot,datetime(2026,10,5,12,tzinfo=timezone.utc))==snapshot
