from typing import Generic, TypeVar, Type, Optional, List, Any, Sequence
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.database import Base

ModelType = TypeVar("ModelType", bound=Base)


class BaseRepository(Generic[ModelType]):
    def __init__(self, model: Type[ModelType], db: AsyncSession):
        self.model = model
        self.db = db

    async def get(self, id: Any) -> Optional[ModelType]:
        """Fetch a single record by primary key."""
        return await self.db.get(self.model, id)

    async def get_multi(self, skip: int = 0, limit: int = 100) -> Sequence[ModelType]:
        """Fetch multiple records with offset and limit."""
        query = select(self.model).offset(skip).limit(limit)
        result = await self.db.execute(query)
        return result.scalars().all()

    async def create(self, obj: ModelType) -> ModelType:
        """Add a new model instance to the session."""
        self.db.add(obj)
        await self.db.flush()  # Populates autoincremented primary keys
        return obj

    async def delete(self, id: Any) -> Optional[ModelType]:
        """Delete a record by primary key, returning the deleted object."""
        db_obj = await self.get(id)
        if db_obj:
            await self.db.delete(db_obj)
            await self.db.flush()
        return db_obj
