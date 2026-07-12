from typing import Optional, Sequence
from fastapi import HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from app.repositories.user_repository import UserRepository
from app.domain.models import User
from app.domain.schemas import UserCreate, UserUpdate
from app.core.security import get_password_hash


class UserService:
    def __init__(self, db: AsyncSession):
        self.user_repo = UserRepository(db)

    async def get_user(self, user_id: int) -> Optional[User]:
        """Retrieve a user by ID."""
        return await self.user_repo.get(user_id)

    async def get_user_by_username(self, username: str) -> Optional[User]:
        """Retrieve a user by username."""
        return await self.user_repo.get_by_username(username)

    async def get_users(self, skip: int = 0, limit: int = 100) -> Sequence[User]:
        """Retrieve a list of users."""
        return await self.user_repo.get_multi(skip=skip, limit=limit)

    async def create_user(self, user_in: UserCreate) -> User:
        """Create a new user, validating that the username and email are unique."""
        # Check if username already exists
        existing_username = await self.user_repo.get_by_username(user_in.username)
        if existing_username:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Username '{user_in.username}' is already registered."
            )

        # Check if email already exists
        existing_email = await self.user_repo.get_by_email(user_in.email)
        if existing_email:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Email '{user_in.email}' is already registered."
            )

        return await self.user_repo.create_user(user_in)

    async def update_user(self, user_id: int, user_in: UserUpdate) -> Optional[User]:
        """Update user properties, checking unique constraints and hashing passwords if changed."""
        db_user = await self.user_repo.get(user_id)
        if not db_user:
            return None

        # Check username uniqueness if updating username
        if user_in.username and user_in.username != db_user.username:
            existing_username = await self.user_repo.get_by_username(user_in.username)
            if existing_username:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Username '{user_in.username}' is already in use."
                )
            db_user.username = user_in.username

        # Check email uniqueness if updating email
        if user_in.email and user_in.email != db_user.email:
            existing_email = await self.user_repo.get_by_email(user_in.email)
            if existing_email:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Email '{user_in.email}' is already in use."
                )
            db_user.email = user_in.email

        # Hash password if provided
        if user_in.password:
            db_user.hashed_password = get_password_hash(user_in.password)

        # Update other fields
        if user_in.role is not None:
            db_user.role = user_in.role
        if user_in.is_active is not None:
            db_user.is_active = user_in.is_active

        # Flush updates to session
        await self.user_repo.db.flush()
        return db_user

    async def delete_user(self, user_id: int) -> Optional[User]:
        """Delete a user from the system."""
        return await self.user_repo.delete(user_id)
