import os
from pathlib import Path

# Define the root directory
data_root = Path('mdnf')

# Iterate through all task directories in the root directory
for task_dir in data_root.iterdir():
    if task_dir.is_dir():  # Ensure it's a directory
        # Define the screenshot directory path
        screenshot_dir = task_dir / 'screenshot'

        # Check if the screenshot directory exists and is a directory
        if screenshot_dir.exists() and screenshot_dir.is_dir():
            # Iterate through all files in the screenshot directory that match the pattern
            for file in screenshot_dir.glob('*highlight*.png'):
                try:
                    print(f"Deleting file: {file}")
                    file.unlink()  # Remove the file
                except Exception as e:
                    print(f"Failed to delete {file}: {e}")
