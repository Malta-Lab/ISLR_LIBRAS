import os
import glob

# Path to the 'minds' folder
folder = '/mnt/G-SSD/BRACIS/BRACIS-2024/lightning_logs/vivit'  # Replace with the actual path to your 'minds' folder

# Use glob to find all .ckpt files recursively within the 'minds' folder
ckpt_files = glob.glob(os.path.join(folder, '**', '*.ckpt'), recursive=True)

# Loop through and remove all found .ckpt files
for ckpt_file in ckpt_files:
    try:
        os.remove(ckpt_file)
        print(f"Deleted: {ckpt_file}")
    except OSError as e:
        print(f"Error deleting {ckpt_file}: {e}")
