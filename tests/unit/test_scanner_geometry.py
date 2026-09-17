import asyncio
import json

import pytest

from scripts.qualify_scanner_patterns import CORPUS, evaluate_case

CORPUS_DATA = json.loads(CORPUS.read_text())


@pytest.mark.parametrize("case", CORPUS_DATA["cases"], ids=lambda case: case["id"])
@pytest.mark.parametrize("scale", CORPUS_DATA["scales"])
def test_synthetic_geometry_and_price_scale(case, scale):
    result = asyncio.run(evaluate_case(case, scale))
    assert result["passed"], result
