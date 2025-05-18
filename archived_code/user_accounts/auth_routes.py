# src/api/routes/auth_routes.py
from fastapi import APIRouter, Depends, HTTPException, Body, Query
from fastapi.responses import JSONResponse, RedirectResponse, HTMLResponse
from typing import Dict, Any, Optional
from pydantic import BaseModel, EmailStr
from archived_code.user_accounts.auth_service import AuthService
from infrastructure.database.supabase.crypto_repository import SupabaseCryptoRepository
from archived_code.user_accounts.dependencies import get_auth_service
from archived_code.user_accounts.account_auth_interface import (
    UserRegistration,
    UserLogin,
    TelegramAuth,
    EmailRequest,
    ProfileUpdate
)
import hashlib
import hmac
import os
import time
import firebase_admin
from firebase_admin import credentials, auth, db

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

# Initialize Firebase Admin SDK
cred_path = os.environ.get("FIREBASE_CREDENTIALS_PATH", "firebase/secrets/firebase-credentials.json")

cred = credentials.Certificate(cred_path)

firebase_admin.initialize_app(cred, {
    'databaseURL': os.environ.get("FIREBASE_DATABASE_URL")
})

# Your Telegram Bot Token - keep this secret!
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")

class TelegramAuthData(BaseModel):
    id: str
    email: str
    first_name: Optional[str] = ""
    last_name: Optional[str] = ""
    username: Optional[str] = ""
    photo_url: Optional[str] = ""
    auth_date: str
    hash: str

def verify_telegram_auth_data(auth_data: Dict[str, Any]) -> bool:
    """
    Verify that the auth data actually came from Telegram.
    https://core.telegram.org/widgets/login#checking-authorization
    """
    if not TELEGRAM_BOT_TOKEN:
        raise HTTPException(status_code=500, detail="Telegram bot token not configured")
    
    # Remove hash from the data before checking
    received_hash = auth_data.pop('hash', None)
    if not received_hash:
        return False
    
    # Check auth_date is not too old (within 1 day)
    auth_date = int(auth_data.get('auth_date', '0'))
    if time.time() - auth_date > 86400:  # 24 hours
        return False
    
    # Sort the remaining data alphabetically
    data_check_string = '\n'.join([f"{k}={v}" for k, v in sorted(auth_data.items())])
    
    # Create secret key from bot token
    secret_key = hashlib.sha256(TELEGRAM_BOT_TOKEN.encode()).digest()
    
    # Generate hash and compare
    computed_hash = hmac.new(secret_key, data_check_string.encode(), hashlib.sha256).hexdigest()
    
    # Return auth data for further use with added hash
    auth_data['hash'] = received_hash
    
    return computed_hash == received_hash

@router.post("/telegram")
async def telegram_auth(auth_data: Dict[str, str]):
    """
    Verify Telegram auth data and return a Firebase custom token
    """
    try:
        # Verify the Telegram auth data
        if not verify_telegram_auth_data(dict(auth_data)):
            return {
                "success": False, 
                "message": "Invalid or expired authentication data"
            }
        
        # Auth data is valid, get or create the user in Firebase
        telegram_id = auth_data.get('id')
        
        # Try to find existing user by custom claims
        try:
            existing_users = auth.list_users().iterate_all()
            firebase_user = None
            
            for user in existing_users:
                claims = user.custom_claims or {}
                if claims.get('telegramId') == telegram_id:
                    firebase_user = user
                    break
        except:
            firebase_user = None
            
        # Create or update the user
        if firebase_user:
            # User exists, update if needed
            user_record = firebase_user
        else:
            # Create new user
            # Before creating the custom token, check if the user has a verified email
            if auth_data.get('email'):
                # Check if this is a valid email or matches certain criteria
                user_email = auth_data.get('email')
                # Add your validation logic here
                
                # You could also store the real email instead of the generated one
                email = user_email
            else:
                # Fall back to the generated email
                email = f"telegram_{telegram_id}@telegram.login"

            password = auth.generate_password_hash(f"telegram_{telegram_id}_{int(time.time())}")
            
            try:
                user_record = auth.create_user(
                    email=email,
                    password=password,
                    display_name=f"{auth_data.get('first_name', '')} {auth_data.get('last_name', '')}".strip()
                )
                
                # Set custom claims for Telegram ID
                auth.set_custom_user_claims(user_record.uid, {'telegramId': telegram_id})
            except Exception as e:
                return {"success": False, "message": f"Failed to create Firebase user: {str(e)}"}
        
        # Create user data for the Realtime Database
        firstname = auth_data.get('first_name', '')
        lastname = auth_data.get('last_name', '')
        display_name = f"{firstname} {lastname}".strip()
        if not display_name and auth_data.get('username'):
            display_name = auth_data.get('username')
            
        user_data = {
            "uuid": user_record.uid,
            "email": user_record.email,
            "firstname": firstname,
            "secondname": lastname,
            "displayName": display_name,
            "avatarUrl": auth_data.get('photo_url', ''),
            "createdTime": str(int(time.time() * 1000)),
            "subscriptionType": "free",
            "isUsingTestDrive": False,
            "telegramId": telegram_id,
            "telegramUsername": auth_data.get('username', '')
        }
        
        # Save to Firebase Realtime Database
        try:
            ref = db.reference(f'/users/{user_record.uid}')
            ref.update(user_data)
        except Exception as e:
            return {"success": False, "message": f"Failed to save user data: {str(e)}"}
        
        # Create a custom token for the user
        try:
            custom_token = auth.create_custom_token(user_record.uid)
        except Exception as e:
            return {"success": False, "message": f"Failed to create custom token: {str(e)}"}
        
        # Return success with token and user data
        return {
            "success": True,
            "message": "Authentication successful",
            "firebaseToken": custom_token.decode('utf-8'),
            "user": user_data
        }
        
    except Exception as e:
        return {"success": False, "message": f"Authentication failed: {str(e)}"}

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
    
    raise HTTPException(status_code=400, detail="Logout failed")

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


@router.get("/supabase-callback")
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