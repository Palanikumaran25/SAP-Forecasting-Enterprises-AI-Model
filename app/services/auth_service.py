from datetime import datetime, timezone, timedelta
from typing import Optional, Union
import jwt
from sqlalchemy.ext.asyncio import AsyncSession
from app.repositories.user_repository import UserRepository
from app.repositories.token_repository import TokenBlocklistRepository
from app.core.security import verify_password, get_password_hash, create_access_token, create_refresh_token, decode_token, ALGORITHM
from app.core.config import settings
from app.domain.models import User
from app.domain.schemas import Token


class AuthService:
    def __init__(self, db: AsyncSession):
        self.user_repo = UserRepository(db)
        self.token_repo = TokenBlocklistRepository(db)

    async def authenticate(self, username: str, password: str) -> Optional[User]:
        """Authenticate a user against their username and plaintext password."""
        user = await self.user_repo.get_by_username(username)
        if not user or not user.is_active:
            return None
        if not verify_password(password, user.hashed_password):
            return None
        return user

    async def login(self, username: str, password: str) -> Optional[Token]:
        """Verify credentials and return a new Token containing access and refresh tokens."""
        user = await self.authenticate(username, password)
        if not user:
            return None
        
        access_token = create_access_token(user.id, user.role.value)
        refresh_token = create_refresh_token(user.id, user.role.value)
        return Token(access_token=access_token, refresh_token=refresh_token)

    async def logout(self, access_token: str, refresh_token: Optional[str] = None) -> None:
        """Invalidate the access token and optionally the refresh token by adding them to blocklist."""
        # Block access token
        acc_payload = decode_token(access_token)
        if acc_payload and "jti" in acc_payload:
            exp = datetime.fromtimestamp(acc_payload["exp"], tz=timezone.utc)
            await self.token_repo.block_jti(acc_payload["jti"], exp)

        # Block refresh token
        if refresh_token:
            ref_payload = decode_token(refresh_token)
            if ref_payload and "jti" in ref_payload:
                exp = datetime.fromtimestamp(ref_payload["exp"], tz=timezone.utc)
                await self.token_repo.block_jti(ref_payload["jti"], exp)

    async def refresh_tokens(self, refresh_token: str) -> Optional[Token]:
        """Validate the refresh token, add the old refresh token to blocklist, and return a new pair."""
        payload = decode_token(refresh_token)
        if not payload or payload.get("type") != "refresh":
            return None
        
        # Check blocklist to prevent replay attacks
        jti = payload.get("jti")
        if jti and await self.token_repo.is_jti_blocked(jti):
            return None
            
        user_id = payload.get("sub")
        if not user_id:
            return None
            
        user = await self.user_repo.get(int(user_id))
        if not user or not user.is_active:
            return None

        # Invalidate old refresh token
        if jti:
            exp = datetime.fromtimestamp(payload["exp"], tz=timezone.utc)
            await self.token_repo.block_jti(jti, exp)

        # Generate new pair
        new_access_token = create_access_token(user.id, user.role.value)
        new_refresh_token = create_refresh_token(user.id, user.role.value)
        return Token(access_token=new_access_token, refresh_token=new_refresh_token)

    async def request_password_reset(self, email: str) -> Optional[str]:
        """Generate a short-lived token for password reset."""
        user = await self.user_repo.get_by_email(email)
        if not user or not user.is_active:
            return None
        
        # Generate custom token with reset type
        expire = datetime.now(timezone.utc) + timedelta(minutes=15)
        to_encode = {
            "sub": str(user.id),
            "type": "reset",
            "exp": int(expire.timestamp())
        }
        reset_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=ALGORITHM)
        return reset_jwt

    async def confirm_password_reset(self, token: str, new_password: str) -> bool:
        """Validate reset token and update user password."""
        try:
            payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[ALGORITHM])
            if payload.get("type") != "reset":
                return False
            
            user_id = payload.get("sub")
            if not user_id:
                return False
                
            user = await self.user_repo.get(int(user_id))
            if not user or not user.is_active:
                return False
                
            user.hashed_password = get_password_hash(new_password)
            self.db_session_dirty = True  # Flag to indicate data state updated
            return True
        except jwt.PyJWTError:
            return False
