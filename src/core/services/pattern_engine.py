import asyncio
from core.use_cases.market_analysis import detect_patterns
import numpy as np
from typing import Any

class PatternEngine:
    CATEGORY_WINDOW_SIZES = {
        "harmonic": [20, 30, 50, 100, 150],
        "candlestick": [1, 2, 3, 5, 7],
        "chart": [12, 15, 20, 30, 50],
    }

    def __init__(self, category: str = "all"):
        self.category = category.lower()
        self.detector = detect_patterns.PatternDetector()

    async def analyze_patterns(self, ohlcv: dict) -> dict:
        if self.category == "all":
            cats = ["harmonic", "candlestick", "chart"]
            results = await asyncio.gather(*[
                self._analyze_category(cat, ohlcv) for cat in cats
            ])
            return {cat: res for cat, res in zip(cats, results)}
        else:
            return {self.category: await self._analyze_category(self.category, ohlcv)}

    async def _analyze_category(self, category, ohlcv):
        window_sizes = self.CATEGORY_WINDOW_SIZES.get(category, [10, 20, 30])
        n = len(ohlcv["close"])
        detected_patterns = []
        for window_size in window_sizes:
            if window_size > n:
                continue
            step_size = max(1, window_size // (8 if window_size < 30 else 5))
            for start in range(0, n - window_size + 1, step_size):
                window = {k: v[start:start+window_size] for k, v in ohlcv.items()}
                patterns = await self.detector.detect_by_category(category, window)
                if patterns:
                    if isinstance(patterns, np.ndarray):
                        patterns = patterns.tolist()
                    if isinstance(patterns, list):
                        dict_patterns = [p for p in patterns if isinstance(p, dict) and isinstance(p.get('confidence', None), (int, float))]
                        if dict_patterns:
                            try:
                                best = max(dict_patterns, key=lambda p: p.get('confidence', 0))
                                detected_patterns.append(best)
                            except Exception:
                                detected_patterns.extend(dict_patterns)
                        else:
                            detected_patterns.extend(patterns)
                    elif isinstance(patterns, dict):
                        detected_patterns.append(patterns)
        return detected_patterns 