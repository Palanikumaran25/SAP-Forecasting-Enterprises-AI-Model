import bcrypt
import jwt
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional, Union, Any
from app.core.config import settings

ALGORITHM = "HS256"


def get_password_hash(password: str) -> str:
    """Hash a plaintext password using bcrypt."""
    pwd_bytes = password.encode("utf-8")
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(pwd_bytes, salt)
    return hashed.decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a plaintext password against a hashed password."""
    try:
        return bcrypt.checkpw(
            plain_password.encode("utf-8"),
            hashed_password.encode("utf-8")
        )
    except Exception:
        return False


def create_access_token(
    subject: Union[str, int, dict], role: Optional[str] = None, expires_delta: Optional[timedelta] = None
) -> str:
    """Generate a JWT access token containing subject ID, role, jti, and expiration."""
    if isinstance(subject, dict):
        sub_val = subject.get("sub")
        role_val = subject.get("role") or role or "Viewer"
    else:
        sub_val = subject
        role_val = role or "Viewer"

    if expires_delta:
        expire = expires_delta
    else:
        expire = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    
    expire_time = datetime.now(timezone.utc) + expire
    to_encode = {
        "sub": str(sub_val),
        "role": role_val,
        "type": "access",
        "jti": str(uuid.uuid4()),
        "exp": int(expire_time.timestamp()),
        "iat": int(datetime.now(timezone.utc).timestamp())
    }
    
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def create_refresh_token(
    subject: Union[str, int, dict], role: Optional[str] = None, expires_delta: Optional[timedelta] = None
) -> str:
    """Generate a JWT refresh token with longer duration for session renewal."""
    if isinstance(subject, dict):
        sub_val = subject.get("sub")
        role_val = subject.get("role") or role or "Viewer"
    else:
        sub_val = subject
        role_val = role or "Viewer"

    if expires_delta:
        expire = expires_delta
    else:
        expire = timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)
        
    expire_time = datetime.now(timezone.utc) + expire
    to_encode = {
        "sub": str(sub_val),
        "role": role_val,
        "type": "refresh",
        "jti": str(uuid.uuid4()),
        "exp": int(expire_time.timestamp()),
        "iat": int(datetime.now(timezone.utc).timestamp())
    }
    
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def decode_token(token: str) -> dict:
    """Decode and validate a JWT token, returning its claims dict, or raises PyJWTError if invalid."""
    return jwt.decode(token, settings.SECRET_KEY, algorithms=[ALGORITHM])


hash_password = get_password_hash
