from datetime import datetime
from typing import Optional
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.repositories.base import BaseRepository
from app.domain.models import TokenBlocklist


class TokenBlocklistRepository(BaseRepository[TokenBlocklist]):
    def __init__(self, db: AsyncSession):
        super().__init__(TokenBlocklist, db)

    async def is_jti_blocked(self, jti: str) -> bool:
        """Check if a specific token JTI is blocked."""
        query = select(TokenBlocklist).where(TokenBlocklist.jti == jti)
        result = await self.db.execute(query)
        return result.scalars().first() is not None

    async def block_jti(self, jti: str, expires_at: datetime) -> TokenBlocklist:
        """Block a token JTI until its expiration time."""
        blocked_token = TokenBlocklist(jti=jti, expires_at=expires_at)
        return await self.create(blocked_token)
