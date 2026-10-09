"""Compare exact detector outputs and CPU work with/without per-window reuse.

Synthetic geometry is a regression/CPU benchmark, not market qualification.
"""
import asyncio
from contextlib import nullcontext
import json
from pathlib import Path
import time
import numpy as np

from core.scanner.engine import closed_window, detector_version, load_registry
from core.use_cases.market_analysis.detect_patterns_engine.shared_features import shared_features
from tests.fixtures.scanner_geometry import geometry_rows

ROOT = Path(__file__).resolve().parents[1]


def numeric_json(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


async def run():
    corpus = json.loads((ROOT / 'tests/fixtures/scanner/geometry.json').read_text())
    detectors = json.loads((ROOT / 'config/scanner/binance-spot-pilot.json').read_text())['detectors']
    registry = load_registry()
    totals = {'uncached_seconds': 0., 'shared_seconds': 0., 'computed_features': 0, 'reused_features': 0, 'windows': 0}
    for case in corpus['cases']:
        for scale in corpus['scales']:
            rows = geometry_rows(case['recipe'], cutoff=1789646400, scale=scale)
            status, ohlcv = closed_window(rows, '15m', 1789646400)
            assert status == 'ready'
            results = []
            for enabled in (False, True):
                started = time.perf_counter()
                with shared_features() if enabled else nullcontext() as features:
                    raw = [await registry[d]['strict_function'](ohlcv) for d in detectors]
                    results.append(json.dumps(raw, sort_keys=True, allow_nan=False, default=numeric_json))
                    if features:
                        totals['computed_features'] += features['computed']
                        totals['reused_features'] += features['hits']
                totals['shared_seconds' if enabled else 'uncached_seconds'] += time.perf_counter() - started
            assert results[0] == results[1], (case['id'], scale)
            totals['windows'] += 1
    totals.update(detector_version=detector_version(), output_parity=True,
                  cpu_scope='one process, synthetic windows, no network or user load')
    path = ROOT / 'logs/scanner-feature-benchmark.json'
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(totals, indent=2) + '\n')
    print(json.dumps(totals, indent=2))


if __name__ == '__main__':
    asyncio.run(run())
