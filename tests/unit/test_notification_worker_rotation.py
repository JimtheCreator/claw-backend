import subprocess
import sys

import pytest

from core.scanner import notification_summary as copy
from scripts.worker_lock import worker_lock


def test_plural_rotation_includes_every_template_and_counts():
    copy._body_bags.clear()
    summary = {'matches': [{}, {}], 'crypto_count': 2, 'forex_count': 0}
    expected = {'2 symbols now match this pattern.',
                '2 symbols are showing this pattern.',
                'New matches detected across multiple symbols.',
                'This pattern is showing up across multiple markets.'}
    for _ in range(3):
        assert {copy.summary_body(summary) for _ in range(4)} == expected


def test_singular_rotation_never_uses_plural_copy():
    copy._body_bags.clear()
    summary = {'matches': [{}]}
    assert len({copy.summary_body(summary) for _ in range(3)}) == 3
    with pytest.raises(ValueError):
        copy.summary_body({'matches': []})


def test_worker_lock_blocks_other_process_and_releases(tmp_path):
    command = [sys.executable, '-c',
               'from scripts.worker_lock import worker_lock; '
               f'lock = worker_lock("delivery", {str(tmp_path)!r}); lock.__enter__()']
    with worker_lock('delivery', tmp_path):
        blocked = subprocess.run(command, capture_output=True, text=True)
        assert blocked.returncode != 0
        assert 'already running' in blocked.stderr
        with worker_lock('inbox', tmp_path):
            pass
    assert subprocess.run(command, capture_output=True).returncode == 0
