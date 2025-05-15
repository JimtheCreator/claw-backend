# src/core/interfaces/auth_repository.py
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

class AuthRepository(ABC):
    @abstractmethod
    async def register_user_with_email(self, email: str, password: str, 
                                user_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Register a new user with email and password"""
        pass
    
    @abstractmethod
    async def login_with_email(self, email: str, password: str) -> Dict[str, Any]:
        """Login user with email and password"""
        pass
    
    @abstractmethod
    async def sign_in_with_google(self, redirect_url: str) -> Dict[str, Any]:
        """Generate URL for Google OAuth sign-in"""
        pass
    
    @abstractmethod
    async def sign_in_with_telegram(self, telegram_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle Telegram authentication"""
        pass
    
    @abstractmethod
    async def send_password_reset_email(self, email: str) -> Dict[str, Any]:
        """Send password reset email"""
        pass
    
    @abstractmethod
    async def verify_email(self, token: str) -> Dict[str, Any]:
        """Verify email address with token"""
        pass
    
    @abstractmethod
    async def send_magic_link(self, email: str) -> Dict[str, Any]:
        """Send magic link for passwordless login"""
        pass
    
    @abstractmethod
    async def refresh_session(self, refresh_token: str) -> Dict[str, Any]:
        """Refresh authentication session"""
        pass
    
    @abstractmethod
    async def logout(self, access_token: str) -> Dict[str, Any]:
        """Logout user and invalidate session"""
        pass
    
    @abstractmethod
    async def get_user(self, user_id: str) -> Dict[str, Any]:
        """Get user data"""
        pass
    
    @abstractmethod
    async def update_user(self, user_id: str, user_data: Dict[str, Any]) -> Dict[str, Any]:
        """Update user profile information"""
        pass