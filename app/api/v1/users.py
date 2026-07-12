from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.database import get_db
from app.api.dependencies import get_current_user, RoleChecker
from app.services.user_service import UserService
from app.domain.models import User
from app.domain.enums import UserRole
from app.domain.schemas import UserResponse, UserCreate, UserUpdate

router = APIRouter()

# Dependency filters
admin_only = Depends(RoleChecker([UserRole.ADMIN]))


@router.get("/me", response_model=UserResponse)
async def read_user_me(
    current_user: User = Depends(get_current_user)
):
    """Retrieve details of the currently logged-in user."""
    return current_user


@router.get("", response_model=List[UserResponse], dependencies=[admin_only])
async def list_users(
    skip: int = 0,
    limit: int = 100,
    db: AsyncSession = Depends(get_db)
):
    """Retrieve all users in the system (Admin only, with pagination)."""
    user_service = UserService(db)
    users = await user_service.get_users(skip=skip, limit=limit)
    return users


@router.post("", response_model=UserResponse, status_code=status.HTTP_201_CREATED, dependencies=[admin_only])
async def create_user(
    user_in: UserCreate,
    db: AsyncSession = Depends(get_db)
):
    """Create a new user in the system (Admin only)."""
    user_service = UserService(db)
    return await user_service.create_user(user_in)


@router.get("/{user_id}", response_model=UserResponse, dependencies=[admin_only])
async def get_user_by_id(
    user_id: int,
    db: AsyncSession = Depends(get_db)
):
    """Retrieve details of a user by ID (Admin only)."""
    user_service = UserService(db)
    user = await user_service.get_user(user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    return user


@router.put("/{user_id}", response_model=UserResponse, dependencies=[admin_only])
async def update_user(
    user_id: int,
    user_in: UserUpdate,
    db: AsyncSession = Depends(get_db)
):
    """Update details of a user by ID (Admin only)."""
    user_service = UserService(db)
    user = await user_service.update_user(user_id, user_in)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    return user


@router.delete("/{user_id}", response_model=UserResponse, dependencies=[admin_only])
async def delete_user(
    user_id: int,
    db: AsyncSession = Depends(get_db)
):
    """Delete a user by ID (Admin only)."""
    user_service = UserService(db)
    user = await user_service.delete_user(user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    return user
