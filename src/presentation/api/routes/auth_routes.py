# src/api/routes/auth_routes.py
from fastapi import APIRouter, Depends, HTTPException, Body, Query
from fastapi.responses import JSONResponse, RedirectResponse, HTMLResponse
from typing import Dict, Any, Optional
from pydantic import BaseModel, EmailStr
from core.services.auth_service import AuthService
from infrastructure.database.supabase.crypto_repository import SupabaseCryptoRepository
from dependencies import get_auth_service
from core.interfaces.account_auth_interface import (
    UserRegistration,
    UserLogin,
    TelegramAuth,
    EmailRequest,
    ProfileUpdate
)

router = APIRouter(
    prefix="/auth",
    tags=["authentication"],
    responses={404: {"description": "Not found"}},
)


# Authentication endpoints
@router.post("/register", response_model=Dict[str, Any])
async def register(
    user_data: UserRegistration,
    auth_service: AuthService = Depends(get_auth_service)
):
    """Register a new user with email and password"""
    # Extract additional data from registration model
    additional_data = {
        "first_name": user_data.first_name,
        "last_name": user_data.last_name,
        "display_name": user_data.display_name,
        "auth_provider": "email"
    }
    
    result = await auth_service.register_user(
        user_data.email, 
        user_data.password,
        additional_data
    )
    
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    
    return result

@router.post("/login", response_model=Dict[str, Any])
async def login(
    credentials: UserLogin,
    auth_service: AuthService = Depends(get_auth_service)
):
    """Login with email and password"""
    result = await auth_service.login(credentials.email, credentials.password)
    
    if "error" in result:
        raise HTTPException(status_code=401, detail=result["error"])
    
    return result

@router.get("/google/login")
async def google_login(
    redirect_url: str,
    auth_service: AuthService = Depends(get_auth_service)
):
    """Generate Google OAuth URL for sign in"""
    result = await auth_service.google_sign_in(redirect_url)
    
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    
    # Return the URL that the client should redirect to
    return {"provider_url": result["provider_url"]}

@router.post("/telegram/login", response_model=Dict[str, Any])
async def telegram_login(
    telegram_data: TelegramAuth,
    auth_service: AuthService = Depends(get_auth_service)
):
    """Login or register with Telegram data"""
    # Convert pydantic model to dict
    telegram_dict = telegram_data.model_dump()
    
    result = await auth_service.telegram_sign_in(telegram_dict)
    
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    
    return result

@router.post("/reset-password")
async def reset_password(
    email_data: EmailRequest,
    auth_service: AuthService = Depends(get_auth_service)
):
    """Send password reset email"""
    result = await auth_service.send_reset_password_email(email_data.email)
    
    # Even if email doesn't exist, return success for security reasons
    return {"status": "email_sent"}

@router.get("/verify-email")
async def verify_email(
    token: str,
    auth_service: AuthService = Depends(get_auth_service)
):
    """Verify email with token"""
    result = await auth_service.verify_user_email(token)
    
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    
    # Redirect to frontend success page or return success response
    return result

@router.post("/magic-link")
async def send_magic_link(
    email_data: EmailRequest,
    auth_service: AuthService = Depends(get_auth_service)
):
    """Send magic link for passwordless login"""
    result = await auth_service.send_magic_link_email(email_data.email)
    
    # Even if email doesn't exist, return success for security reasons
    return {"status": "email_sent"}

@router.post("/refresh-token", response_model=Dict[str, Any])
async def refresh_token(
    refresh_token: str = Body(..., embed=True),
    auth_service: AuthService = Depends(get_auth_service)
):
    """Refresh authentication token"""
    result = await auth_service.refresh_auth_token(refresh_token)
    
    if "error" in result:
        raise HTTPException(status_code=401, detail=result["error"])
    
    return result

@router.post("/logout")
async def logout(
    access_token: str = Body(..., embed=True),
    auth_service: AuthService = Depends(get_auth_service)
):
    """Logout user"""
    result = await auth_service.logout_user(access_token)
    
    if "error" in result:
        raise HTTPException(status_code=401, detail=result["error"])
    
    if result:
        # If the logout was successful, clear the local storage
        # This is a placeholder; actual implementation would depend on client-side logic
        return JSONResponse(content={"status": "logged_out"}, status_code=200)
    
    return {"status": "logged_out"}

@router.get("/user/{user_id}", response_model=Dict[str, Any])
async def get_user_profile(
    user_id: str,
    auth_service: AuthService = Depends(get_auth_service)
):
    """Get user profile data"""
    result = await auth_service.get_user_profile(user_id)
    
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    
    return result

@router.put("/user/{user_id}", response_model=Dict[str, Any])
async def update_user_profile(
    user_id: str,
    profile_data: ProfileUpdate,
    auth_service: AuthService = Depends(get_auth_service)
):
    """Update user profile data"""
    # Convert pydantic model to dict and remove None values
    update_data = {k: v for k, v in profile_data.dict().items() if v is not None}
    
    result = await auth_service.update_user_profile(user_id, update_data)
    
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    
    return result


@router.get("/callback")
async def auth_callback():
    """Handle authentication callback from Supabase
    
    This endpoint doesn't actually process the token - that happens client-side.
    It simply serves an HTML page that will extract the token from the URL fragment
    and store it appropriately.
    """
    # Return HTML that will process the auth redirect
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Authentication</title>
        <script>
            // Extract token from URL hash
            const hashParams = new URLSearchParams(
                window.location.hash.substring(1)
            );
            
            const accessToken = hashParams.get('access_token');
            const refreshToken = hashParams.get('refresh_token');
            const expiresIn = hashParams.get('expires_in');
            const tokenType = hashParams.get('token_type');
            
            if (accessToken) {
                // Store tokens in localStorage
                localStorage.setItem('supabase.auth.token', JSON.stringify({
                    access_token: accessToken,
                    refresh_token: refreshToken,
                    expires_in: parseInt(expiresIn),
                    expires_at: Date.now() + parseInt(expiresIn) * 1000,
                    token_type: tokenType
                }));
                
                // Redirect to the app
                window.location.href = '/';
            } else {
                document.body.innerHTML = 'Authentication failed. Please try again.';
            }
        </script>
    </head>
    <body>
        Processing authentication...
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)