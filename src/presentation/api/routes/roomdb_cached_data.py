from infrastructure.database.supabase.crypto_repository import SupabaseCryptoRepository
from core.use_cases.market_analysis.detect_patterns_engine import initialized_pattern_registry
from fastapi import APIRouter, HTTPException, Query
from typing import List
from pydantic import BaseModel

class PatternResponse(BaseModel):
    name: str
    displayName: str

class SymbolResponse(BaseModel):
    symbol: str
    asset: str  # or make this optional if needed

router = APIRouter(tags=["roomdb_cached_data"])

# Dependency injection function
def get_supabase_repo():
    return SupabaseCryptoRepository()

# --- Initialize your CryptoRepository ---
try:
    supabase_repo = get_supabase_repo()
    print("Successfully connected to Supabase.")
except Exception as e:
    print(f"Error initializing CryptoRepository: {e}")
    supabase_repo = None


def format_display_name(pattern_type: str) -> str:
    """
    Convert snake_case pattern type to a proper display name.
    Examples:
    - bullish_engulfing -> Bullish Engulfing
    - three_white_soldiers -> Three White Soldiers
    - standard_doji -> Standard Doji
    """
    return pattern_type.replace('_', ' ').title()


# ==============================================================================
# ENDPOINT 1: Get All Available Patterns (Flattened Structure)
# ==============================================================================
@router.get("/cache/patterns", response_model=List[PatternResponse])
def get_patterns():
    """
    Returns a flattened list of all pattern types, where each pattern type
    has its own name and displayName properties.
    """
    pattern_list = []
    
    # Iterate over the registry's items
    for pattern_name, details in initialized_pattern_registry.items():
        pattern_types = details.get("types", [pattern_name])
        
        # Create a separate entry for each pattern type
        for pattern_type in pattern_types:
            pattern_list.append({
                "name": pattern_type,
                "displayName": format_display_name(pattern_type)
            })
    
    # Remove duplicates if any (in case same pattern type appears in multiple groups)
    seen_names = set()
    unique_patterns = []
    for pattern in pattern_list:
        if pattern["name"] not in seen_names:
            seen_names.add(pattern["name"])
            unique_patterns.append(pattern)
    
    # Sort by name for consistent ordering
    unique_patterns.sort(key=lambda x: x["name"])
    
    return unique_patterns


# ==============================================================================
# ENDPOINT 2: Get All Symbols from Supabase (Paginated)
# ==============================================================================
@router.get("/cache/symbols", response_model=List[SymbolResponse])
async def get_symbols(
    page: int = Query(1, gt=0, description="Page number, starting from 1"),
    limit: int = Query(200, gt=0, le=1000, description="Number of items per page")
):
    """
    Fetches a paginated list of symbols from Supabase using your CryptoRepository.
    """
    if not supabase_repo:
        raise HTTPException(status_code=500, detail="Supabase connection not configured")

    try:
        symbols = await supabase_repo.get_symbols_paginated(page, limit)

        if symbols is not None:
            return symbols
        else:
            raise HTTPException(status_code=500, detail="Failed to fetch symbols from Supabase")

    except Exception as e:
        print(f"Error in /cache/symbols endpoint: {e}")
        raise HTTPException(status_code=500, detail="An internal error occurred")


@router.get("/cache/symbols/search", response_model=List[SymbolResponse])
async def search_symbols(
    q: str = Query(..., min_length=2, description="Search query for symbol or name")
):
    """
    Searches for symbols in the database based on a query string.
    This enables on-demand fetching from the client.
    """
    if not supabase_repo:
        raise HTTPException(status_code=500, detail="Database connection not configured")

    try:
        # We use the new repository method here
        symbols = await supabase_repo.search_cryptos(query=q, limit=2)

        if symbols is not None:
            return symbols
        else:
            # This could be a 404 if you prefer, but 500 signals a search failure
            raise HTTPException(status_code=500, detail="Failed to execute symbol search")

    except Exception as e:
        print(f"Error in /cache/symbols/search endpoint: {e}")
        raise HTTPException(status_code=500, detail="An internal server error occurred")