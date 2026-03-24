from datetime import datetime, UTC

from pydantic import BaseModel, Field


class MemoryData(BaseModel):
    """Core memory data without embedding."""
    id: int
    content: str = Field(min_length=1, description="Memory content")
    importance: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Importance score between 0 and 1"
    )
    category: str = Field(default="general", description="Memory category")
    topics: list[str] = Field(default_factory=list, description="Related topics")


class Memory(BaseModel):
    """Memory entry with embedding."""
    data: MemoryData
    embedding: list[float] = Field(description="Vector embedding")


class MemoryCollection(BaseModel):
    """Collection of memories for a user."""
    memories: list[Memory] = Field(default_factory=list)
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    last_deduplicated_at: datetime | None = None

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class UserProfile(BaseModel):
    """
    Structured user profile containing PII and personal details.

    Stored as key-value pairs where keys are descriptive field names
    (e.g., 'name', 'location', 'workplace') and values are the user's information.
    """
    info: dict[str, str] = Field(
        default_factory=dict,
        description="Key-value pairs of user information"
    )
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
