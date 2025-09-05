import os
import shutil
from pathlib import Path

from dotenv import load_dotenv

from src.pipeline.deleteTFR import delete_old_files

load_dotenv()

log_path = os.environ.get('LOG_PATH')

delete_old_files(selected_type="train", parent_path=log_path, file_type="*.v2")