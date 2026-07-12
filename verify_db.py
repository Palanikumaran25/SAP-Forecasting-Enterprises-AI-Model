import asyncio
import sys
from sqlalchemy import text
from app.core.database import engine
from app.core.config import settings

async def verify_connection():
    print(f"Connecting to database at: {settings.DATABASE_URL.split('@')[-1]}")
    try:
        async with engine.connect() as conn:
            result = await conn.execute(text("SELECT 1"))
            val = result.scalar()
            if val == 1:
                print("SUCCESS: Database connection established successfully!")
                return True
            else:
                print(f"WARNING: Database returned unexpected query result: {val}")
                return False
    except Exception as e:
        print(f"ERROR: Failed to connect to database. Details:\n{e}", file=sys.stderr)
        print("\nPlease make sure that:", file=sys.stderr)
        print("1. Your PostgreSQL server is running.", file=sys.stderr)
        print("2. The database 'sap_fi_forecasting' exists.", file=sys.stderr)
        print("3. The credentials in your .env file are correct.", file=sys.stderr)
        return False

if __name__ == "__main__":
    success = asyncio.run(verify_connection())
    sys.exit(0 if success else 1)
