"""Evaluate versioned synthetic geometry cases through the actual scanner contract."""
import argparse
import asyncio
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace
import warnings

if __name__ == "__main__":
    os.environ.setdefault("PYTHON_DOTENV_DISABLED", "1")

from core.scanner.catalog import pattern_catalog
from core.scanner.engine import detector_version, scan_instrument
from tests.fixtures.scanner_geometry import geometry_rows

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "tests/fixtures/scanner/geometry.json"
CUTOFF = 1789646400
PILOT = json.loads((ROOT / "config/scanner/binance-spot-pilot.json").read_text())


async def evaluate_case(case, scale):
    rows = geometry_rows(case["recipe"], scale=scale, cutoff=CUTOFF)
    async def load(*args):
        return rows
    started = time.perf_counter()
    with warnings.catch_warnings(record=True) as emitted:
        warnings.simplefilter("always", RuntimeWarning)
        result = await scan_instrument("FIXTUREUSDT", "15m", CUTOFF,
            PILOT["detectors"], SimpleNamespace(load=load))
    found = sorted(match["pattern_id"] for match in result["matches"]
                   if match["detector_id"] == case["detector"])
    expected = [case["expected"]] if case["expected"] else []
    numerical_warnings = [str(item.message) for item in emitted if issubclass(item.category, RuntimeWarning)]
    return {"case": case["id"], "detector": case["detector"], "scale": scale,
            "expected": expected, "found": found, "status": result["status"],
            "passed": result["status"] == "ready" and found == expected and not numerical_warnings,
            "warnings": numerical_warnings, "issues": result["issues"],
            "milliseconds": round((time.perf_counter() - started) * 1000, 3)}


async def qualify():
    corpus = json.loads(CORPUS.read_text())
    results = [await evaluate_case(case, scale) for case in corpus["cases"] for scale in corpus["scales"]]
    enabled = {item["id"] for item in pattern_catalog() if item["detector_id"] in PILOT["detectors"]}
    positives = {case["expected"] for case in corpus["cases"] if case["expected"]}
    # Fixture inventory alone is not evidence that a detector passed it.
    failing_positives = {pattern for row in results if not row["passed"] for pattern in row["expected"]}
    return {"evidence": corpus["evidence"], "detector_version": detector_version(),
            "total": len(results), "passed": sum(row["passed"] for row in results),
            "positive_variants_covered": sorted(positives),
            "positive_variants_passing_all_scales": sorted(positives - failing_positives),
            "detector_evaluations": len(results) * len(PILOT["detectors"]),
            "co_evaluation": "Every case runs all enabled detectors; recognition labels judge only the named detector.",
            "enabled_variants_without_positive_fixture": sorted(enabled - positives),
            "market_precision": None, "market_recall": None,
            "note": "Synthetic engineering cases. Unlabeled real-market accuracy remains unmeasured.",
            "results": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=ROOT / "logs/scanner-qualification.json")
    args = parser.parse_args()
    report = asyncio.run(qualify())
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"{report['passed']}/{report['total']} geometry checks passed; "
          f"{len(report['positive_variants_covered'])} variants have synthetic positive fixtures")
    for result in report["results"]:
        if not result["passed"]:
            print(result["case"], result["scale"], result["status"], result["found"], result["warnings"])
    print("Report:", args.report)
    raise SystemExit(0 if report["passed"] == report["total"] else 1)


if __name__ == "__main__":
    main()
