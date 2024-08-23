import os
import sys

def delete_ckpt_files(folder_path):
    # Walk through the directory structure
    for root, dirs, files in os.walk(folder_path):
        # Check if "WLASL" is in the name of the current directory
        if "WLASL" in os.path.basename(root):
            continue  # Skip this directory and its subdirectories

        for file in files:
            if file.endswith(".ckpt"):
                file_path = os.path.join(root, file)
                try:
                    os.remove(file_path)
                    print(f"Deleted: {file_path}")
                except Exception as e:
                    print(f"Error deleting {file_path}: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python delete_ckpt.py <folder_path>")
        sys.exit(1)

    folder_path = sys.argv[1]
    delete_ckpt_files(folder_path)
