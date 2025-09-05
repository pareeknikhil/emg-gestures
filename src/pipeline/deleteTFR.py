import os
from pathlib import Path

tfrecord_path = os.environ.get('TFRECORD_PATH')

def delete_old_files(selected_type: list[str], parent_path=tfrecord_path, file_type="*.tfrecord") -> None:
    for type in selected_type:
        folder_path = Path(parent_path) / type
        if folder_path.is_dir():
            try:
                for file in folder_path.glob(pattern=file_type):
                    file.unlink()
                    print(f"Deleted: {file}")
            except OSError as e:
                print(f"Error deleting folder {folder_path}: {e}")
        else:
            print(f"Folder {folder_path} does not exists")
