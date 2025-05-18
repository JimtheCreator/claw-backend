# src/dependencies.py
from infrastructure.database.supabase.crypto_repository import SupabaseCryptoRepository
from archived_code.user_accounts.auth_service import AuthService

# Singleton pattern for repositories and services
_auth_repository = None
_auth_service = None

def get_auth_repository():
    """Get or create the auth repository instance"""
    global _auth_repository
    if _auth_repository is None:
        _auth_repository = SupabaseCryptoRepository()
    return _auth_repository

def get_auth_service():
    """Get or create the auth service instance"""
    global _auth_service
    if _auth_service is None:
        auth_repository = get_auth_repository()
        _auth_service = AuthService(auth_repository)
    return _auth_service