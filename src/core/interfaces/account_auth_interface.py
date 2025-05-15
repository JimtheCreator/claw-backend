from typing import Dict, Any, Optional
from pydantic import BaseModel, EmailStr

# Models for request validation
class UserRegistration(BaseModel):
    email: EmailStr
    password: str
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    display_name: Optional[str] = None
    
class UserLogin(BaseModel):
    email: EmailStr
    password: str

class TelegramAuth(BaseModel):
    id: int
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    username: Optional[str] = None
    photo_url: Optional[str] = None
    auth_date: int
    hash: str

class EmailRequest(BaseModel):
    email: EmailStr
    
class ProfileUpdate(BaseModel):
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    display_name: Optional[str] = None
    avatar_url: Optional[str] = None