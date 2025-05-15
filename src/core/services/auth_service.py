# src/core/services/auth_service.py
from typing import Dict, Any, Optional
from core.interfaces.auth_repository import AuthRepository

class AuthService:
    def __init__(self, auth_repository: AuthRepository):
        self.auth_repository = auth_repository
        
    async def register_user(self, email: str, password: str, user_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Register a new user with email and password"""
        return await self.auth_repository.register_user_with_email(email, password, user_data)
        
    async def login(self, email: str, password: str) -> Dict[str, Any]:
        """Login with email and password"""
        return await self.auth_repository.login_with_email(email, password)
        
    async def google_sign_in(self, redirect_url: str) -> Dict[str, Any]:
        """Begin Google OAuth sign-in process"""
        return await self.auth_repository.sign_in_with_google(redirect_url)
        
    async def google_auth_callback(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle Google OAuth callback
        
        In Supabase, the callback is handled automatically by the client
        and not by the backend. This method is included for completion.
        """
        # This would typically process the callback, but with Supabase
        # the callback processing happens on the client side
        return {"status": "success", "message": "Use Supabase client to process callback"}
        
    async def telegram_sign_in(self, telegram_data: Dict[str, Any]) -> Dict[str, Any]:
        """Authenticate with Telegram data"""
        return await self.auth_repository.sign_in_with_telegram(telegram_data)
        
    async def send_reset_password_email(self, email: str) -> Dict[str, Any]:
        """Send password reset email"""
        return await self.auth_repository.send_password_reset_email(email)
        
    async def verify_user_email(self, token: str) -> Dict[str, Any]:
        """Verify user email with token"""
        return await self.auth_repository.verify_email(token)
        
    async def send_magic_link_email(self, email: str) -> Dict[str, Any]:
        """Send magic link for passwordless login"""
        return await self.auth_repository.send_magic_link(email)
        
    async def refresh_auth_token(self, refresh_token: str) -> Dict[str, Any]:
        """Refresh authentication token"""
        return await self.auth_repository.refresh_session(refresh_token)
        
    async def logout_user(self, access_token: str) -> Dict[str, Any]:
        """Logout user"""
        return await self.auth_repository.logout(access_token)
        
    async def get_user_profile(self, user_id: str) -> Dict[str, Any]:
        """Get user profile data"""
        return await self.auth_repository.get_user(user_id)
        
    async def update_user_profile(self, user_id: str, 
                           user_data: Dict[str, Any]) -> Dict[str, Any]:
        """Update user profile data"""
        return await self.auth_repository.update_user(user_id, user_data)