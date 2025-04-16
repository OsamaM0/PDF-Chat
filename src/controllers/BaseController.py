from helpers.config import get_settings, Settings
import os
import random
import string

class BaseController:
    
    def __init__(self):

        self.app_settings = get_settings()
        
        self.base_dir = os.path.dirname( os.path.dirname(__file__) )
        self.files_dir = os.path.join(
            self.base_dir,
            "assets/files"
        )

        self.database_dir = os.path.join(
            self.base_dir,
            "assets/database"
        )
        
    def generate_random_string(self, length: int=12):
        """Generate a random string of lowercase letters and digits to be used as a identifier for a document

        Args:
            length (int, optional): Number of characters in the string. Defaults to 12.

        Returns:
            _type_: Random string of lowercase letters and digits
        """
        return ''.join(random.choices(string.ascii_lowercase + string.digits, k=length))

    def get_database_path(self, db_name: str):
        """
        Get the path to a database file given the name of the database.

        Args:
            db_name (str): Name of the database.

        Returns:
            str: Path to the database file.
        """
        database_path = os.path.join(self.database_dir, db_name)

        # Ensure the directory exists, avoid error if it already exists
        try:
            os.makedirs(database_path, exist_ok=True)
            print(f"Directory ensured at: {database_path}")
        except Exception as e:  # Remove the raise statement
            print(f"Failed to ensure database directory at {database_path}: {e}")

        return database_path