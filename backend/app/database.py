"""MongoDB async connection using Motor."""

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from app.config import settings

# Global database client and database references
client: AsyncIOMotorClient | None = None
db: AsyncIOMotorDatabase | None = None


async def connect_to_mongo():
    """Connect to MongoDB and initialize collections."""
    global client, db
    client = AsyncIOMotorClient(settings.MONGODB_URL)
    db = client[settings.DB_NAME]

    # Create indexes for faster queries
    await db.documents.create_index("filename")
    await db.documents.create_index("file_type")
    await db.documents.create_index("upload_time")
    await db.chunks.create_index("document_id")
    await db.chat_history.create_index("document_id")
    await db.chat_history.create_index("created_at")

    print(f"[+] Connected to MongoDB: {settings.MONGODB_URL}/{settings.DB_NAME}")


async def close_mongo_connection():
    """Close MongoDB connection."""
    global client
    if client:
        client.close()
        print("[-] MongoDB connection closed.")


def get_db() -> AsyncIOMotorDatabase:
    """Get the database instance."""
    if db is None:
        raise RuntimeError("Database not initialized. Call connect_to_mongo() first.")
    return db
