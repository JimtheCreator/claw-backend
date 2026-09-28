"""Saved pattern watch contract. Public browsing never depends on this module."""
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator
from .catalog import ScanInterval


class WatchCreate(BaseModel):
    model_config = ConfigDict(extra='forbid')
    universe: str = Field(default='binance-spot-pilot', pattern=r'^[a-z0-9][a-z0-9-]{0,63}$')
    pattern_id: str = Field(pattern=r'^[a-z0-9_]{1,80}$')
    interval: ScanInterval = '15m'
    symbols: list[str] = Field(default_factory=list, max_length=50)
    mode: Literal['once', 'repeat'] = 'repeat'

    @field_validator('symbols')
    @classmethod
    def normalize_symbols(cls, values):
        import re
        normalized = sorted(set(values))
        if any(not re.fullmatch(r'[A-Z0-9]{2,30}', value) for value in normalized):
            raise ValueError('Invalid symbol')
        return normalized


class WatchAction(BaseModel):
    model_config = ConfigDict(extra='forbid')
    action: Literal['pause', 'resume']


class WatchLimitReached(Exception):
    pass


class EventConflict(Exception):
    pass


class FollowCreate(BaseModel):
    model_config = ConfigDict(extra='forbid')
    universe: str = Field(default='binance-spot-pilot', pattern=r'^[a-z0-9][a-z0-9-]{0,63}$')
    # Accepted for older clients; saved follows always cover every scanner interval.
    interval: ScanInterval | None = Field(default=None, deprecated=True)
    muted: bool = False


class DeviceRegistration(BaseModel):
    model_config = ConfigDict(extra='forbid')
    token: str = Field(min_length=20, max_length=4096)
