# src/core/domain/entities/UserEntity.py
from pydantic import BaseModel, Field, EmailStr
from typing import Optional, Dict, Any
from datetime import datetime
from uuid import UUID

class UserEntity(BaseModel):
    id: UUID
    email: EmailStr
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    display_name: Optional[str] = None
    avatar_url: Optional[str] = None
    created_at: datetime
    updated_at: Optional[datetime] = None
    last_login: Optional[datetime] = None
    telegram_id: Optional[str] = None
    telegram_username: Optional[str] = None
    auth_provider: str = "email"  # email, google, telegram
    email_verified: bool = False
    is_active: bool = True
    extra_data: Optional[Dict[str, Any]] = None
    
    class Config:
        from_attributes = True