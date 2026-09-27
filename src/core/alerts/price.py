"""Price rules use a fixed reference, never a rolling 24-hour percentage."""
from decimal import Decimal
from typing import Literal
from uuid import UUID
from pydantic import BaseModel, ConfigDict, Field, model_validator

class PriceAlertCreate(BaseModel):
    model_config = ConfigDict(extra='forbid')
    request_id: UUID
    symbol: str = Field(pattern=r'^[A-Z0-9]{2,30}$')
    kind: Literal['price', 'percentage']
    direction: Literal['above', 'below']
    amount: Decimal = Field(gt=0, le=Decimal('1000000000000'), allow_inf_nan=False)
    reference_price: Decimal = Field(gt=0, le=Decimal('1000000000000'), allow_inf_nan=False)

    @model_validator(mode='after')
    def valid_target(self):
        if self.kind == 'percentage' and (self.amount > 10000 or self.direction == 'below' and self.amount >= 100):
            raise ValueError('Choose a decrease below 100% or an increase up to 10,000%')
        return self

    @property
    def target(self):
        if self.kind == 'price':
            return self.amount
        sign = 1 if self.direction == 'above' else -1
        return self.reference_price * (1 + sign * self.amount / 100)


def valid_ticks(ticks, now_ms):
    """Only fresh, finite positive exchange prices may trigger alerts."""
    import re
    result = []
    for tick in ticks if isinstance(ticks, list) else []:
        try:
            symbol, price, stamp = tick['s'], Decimal(str(tick['c'])), int(tick['E'])
            if (re.fullmatch(r'[A-Z0-9]{2,30}', symbol) and price.is_finite() and price > 0
                    and now_ms - 30000 <= stamp <= now_ms + 5000):
                result.append({'symbol': symbol, 'price': str(price), 'time': stamp})
        except (KeyError, TypeError, ValueError, ArithmeticError):
            continue
    return result
