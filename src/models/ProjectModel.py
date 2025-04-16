from .BaseDataModel import BaseDataModel
from .db_schemes import Project
from .enums.DataBaseEnum import DataBaseEnum
from pymongo.errors import PyMongoError
from typing import List, Tuple, Optional

class ProjectModel(BaseDataModel):
    """
    Model to handle project-related database operations.
    """

    def __init__(self, db_client: object):
        super().__init__(db_client=db_client)
        self.collection = self.db_client[DataBaseEnum.COLLECTION_PROJECT_NAME.value]

    @classmethod
    async def create_instance(cls, db_client: object) -> "ProjectModel":
        """
        Factory method to create and initialize an instance of ProjectModel.
        """
        instance = cls(db_client)
        await instance.init_collection()
        return instance

    async def init_collection(self) -> None:
        """
        Ensures the required collection exists and initializes indexes.
        """
        try:
            all_collections = await self.db_client.list_collection_names()
            collection_name = DataBaseEnum.COLLECTION_PROJECT_NAME.value

            if collection_name not in all_collections:
                self.collection = self.db_client[collection_name]
                indexes = Project.get_indexes()
                for index in indexes:
                    await self.collection.create_index(
                        index["key"],
                        name=index["name"],
                        unique=index["unique"]
                    )
        except PyMongoError as e:
            raise RuntimeError(f"Failed to initialize collection: {e}")

    async def create_project(self, project: Project) -> Project:
        """
        Inserts a new project into the database.

        Args:
            project (Project): The project object to insert.

        Returns:
            Project: The project object with the assigned ID.
        """
        try:
            result = await self.collection.insert_one(
                project.dict(by_alias=True, exclude_unset=True)
            )
            project.id = result.inserted_id
            return project
        except PyMongoError as e:
            raise RuntimeError(f"Error creating project: {e}")

    async def get_project_or_create_one(self, project_id: str) -> Project:
        """
        Retrieves an existing project by ID or creates a new one if not found.

        Args:
            project_id (str): The project ID.

        Returns:
            Project: The existing or newly created project.
        """
        try:
            record = await self.collection.find_one({"project_id": project_id})

            if not record:
                # Create a new project if none exists
                project = Project(project_id=project_id)
                return await self.create_project(project)

            return Project(**record)
        except PyMongoError as e:
            raise RuntimeError(f"Error retrieving or creating project: {e}")

    async def get_all_projects(self, page: int = 1, page_size: int = 10) -> Tuple[List[Project], int]:
        """
        Retrieves a paginated list of projects.

        Args:
            page (int): The current page number (default is 1).
            page_size (int): The number of items per page (default is 10).

        Returns:
            Tuple[List[Project], int]: A list of projects and the total number of pages.
        """
        try:
            total_documents = await self.collection.count_documents({})

            # Calculate total pages
            total_pages = (total_documents + page_size - 1) // page_size  # ceiling division

            cursor = self.collection.find().skip((page - 1) * page_size).limit(page_size)

            projects = [Project(**document) async for document in cursor]
            return projects, total_pages
        except PyMongoError as e:
            raise RuntimeError(f"Error retrieving projects: {e}")
