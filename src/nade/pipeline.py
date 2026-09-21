# Step 1: User inputs folder location, automatically parse structe

# run main() from scan_folder.py
from nade.io.scan_folder import scan_folder_with_metadata

folder_name = "tests/reference_data"

files_metadata = scan_folder_with_metadata(folder_name)