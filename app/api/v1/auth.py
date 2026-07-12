from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.database import get_db
from app.api.dependencies import security
from app.services.auth_service import AuthService
from app.domain.schemas import Token, RefreshTokenRequest, PasswordResetRequest, PasswordResetConfirm
from pydantic import BaseModel

router = APIRouter(prefix="/auth", tags=["Authentication"])


class LoginRequest(BaseModel):
    username: str
    password: str


class LogoutRequest(BaseModel):
    refresh_token: Optional[str] = None


@router.post("/login", response_model=Token)
async def login(
    request_data: LoginRequest,
    db: AsyncSession = Depends(get_db)
):
    """Authenticate user credentials and return access and refresh tokens."""
    auth_service = AuthService(db)
    tokens = await auth_service.login(request_data.username, request_data.password)
    if not tokens:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password"
        )
    return tokens


@router.post("/logout")
async def logout(
    request_data: Optional[LogoutRequest] = None,
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: AsyncSession = Depends(get_db)
):
    """Log out user, adding active access token and optional refresh token to revocation blocklist."""
    auth_service = AuthService(db)
    await auth_service.logout(
        access_token=credentials.credentials,
        refresh_token=request_data.refresh_token if request_data else None
    )
    return {"detail": "Successfully logged out"}


@router.post("/refresh", response_model=Token)
async def refresh(
    request_data: RefreshTokenRequest,
    db: AsyncSession = Depends(get_db)
):
    """Refresh tokens using a valid refresh token. Implements token rotation and invalidates the old token."""
    auth_service = AuthService(db)
    tokens = await auth_service.refresh_tokens(request_data.refresh_token)
    if not tokens:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid, expired, or revoked refresh token"
        )
    return tokens


@router.post("/reset-password/request")
async def request_password_reset(
    request_data: PasswordResetRequest,
    db: AsyncSession = Depends(get_db)
):
    """Request a password reset link. Generates a short-lived token (returned for demo/testing)."""
    auth_service = AuthService(db)
    token = await auth_service.request_password_reset(request_data.email)
    if not token:
        # Avoid user enumeration by returning success status even if email doesn't exist
        return {"detail": "If the email is registered, a password reset token has been generated"}
        
    return {
        "detail": "Password reset token generated successfully",
        "token": token
    }


@router.post("/reset-password/confirm")
async def confirm_password_reset(
    request_data: PasswordResetConfirm,
    db: AsyncSession = Depends(get_db)
):
    """Reset the user's password using a valid reset token."""
    auth_service = AuthService(db)
    success = await auth_service.confirm_password_reset(request_data.token, request_data.new_password)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid, expired, or revoked password reset token"
        )
    return {"detail": "Password has been successfully updated"}
