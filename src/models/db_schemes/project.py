from pydantic import BaseModel, Field, field_validator
from typing import Optional
from bson.objectid import ObjectId

class Project(BaseModel):
    """
    Represents a project model with attributes:
    - `id`: MongoDB ObjectId, used as the unique identifier.
    - `project_id`: Alphanumeric string that uniquely identifies the project.
    """
    
    # Define fields in the model
    id: Optional[ObjectId] = Field(None, alias="_id")  # Maps MongoDB `_id` to `id` for convenience
    project_id: str = Field(..., min_length=1)  # Ensures `project_id` is non-empty and a required field

    # Validator for the `project_id` field
    @field_validator('project_id')
    def validate_project_id(cls, value):
        """
        Validates that `project_id` contains only alphanumeric characters.
        Raises a ValueError if the validation fails.
        """
        if not value.isalnum():
            raise ValueError('project_id must be alphanumeric')  # Enforces alphanumeric constraint
        return value

    # Configuration for the Pydantic model
    class Config:
        arbitrary_types_allowed = True  # Allows custom types like ObjectId to be used

    # Class method to define MongoDB indexes
    @classmethod
    def get_indexes(cls):
        """
        Returns the indexes to be created in the MongoDB collection:
        - Creates a unique index on the `project_id` field to ensure no duplicates.
        """
        return [
            {
                "key": [("project_id", 1)],  # Specifies the field and sort order (ascending)
                "name": "project_id_index_1",  # Index name for reference
                "unique": True  # Enforces uniqueness of the `project_id`
            }
        ]
