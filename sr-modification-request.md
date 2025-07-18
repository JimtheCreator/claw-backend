# Cursor AI Prompt: Enhance Support/Resistance Engine with Multi-Touch Clustering

## Task Overview
I need you to enhance my existing `SupportResistanceEngine` to implement professional-grade multi-touch clustering for S/R levels. Currently, the engine finds individual pivot points, but I want it to cluster nearby price touches and prioritize levels based on touch frequency, volume, and recency.

## Current Engine Status
- ✅ Finds local extrema using `scipy.signal.argrelextrema`
- ✅ Weights levels by volume and recency
- ✅ Uses ATR for distance filtering
- ❌ Missing multi-touch clustering (the main enhancement needed)
- ❌ Missing touch frequency scoring

## Required Enhancements

### 1. Multi-Touch Clustering Algorithm
Add a new method `_cluster_touches()` that:
- Groups nearby price touches within a tolerance (0.1% of price)
- Counts how many times each price level was touched
- Calculates average price for each cluster
- Tracks timestamps of all touches in each cluster

### 2. Enhanced Level Scoring System
Replace the current simple weighting with a comprehensive scoring system:
- **Touch Count Weight**: 3+ touches = highest priority
- **Volume Weight**: Average volume at cluster touches
- **Recency Weight**: More recent touches weighted higher
- **Duration Weight**: How long the level held significance
- **Strength Score**: Combined metric for ranking levels

### 3. Hybrid Level Detection
Modify `_find_levels()` to return both:
- **Multi-touch clustered levels** (highest priority)
- **Single high-volume pivots** (secondary priority)
- Combine and rank by strength score

## Implementation Requirements

### New Methods to Add:
```python
def _cluster_touches(self, df, price_col, tolerance=0.001):
    """Cluster nearby price touches into significant levels"""
    # Find all potential touch points (not just extrema)
    # Group touches within tolerance percentage
    # Return clusters with touch_count >= 2

def _calculate_level_strength(self, cluster, df):
    """Calculate comprehensive strength score for a level cluster"""
    # Touch count scoring (most important)
    # Volume weighting
    # Recency weighting  
    # Duration weighting
    # Return total strength score

def _find_all_touch_candidates(self, df, price_col):
    """Find all potential touch points, not just local extrema"""
    # Include: local extrema, high volume bars, significant price reactions
    # Return comprehensive list of touch candidates
