import csv
import os
import requests
from tqdm import tqdm
import unicodedata
import urllib3

# Disable SSL warnings (useful if certificate verification fails)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def sanitize_filename(name):
    """
    Convert the given name into a safe filename:
    - Remove control characters.
    - Normalize accented characters to ASCII.
    - Trim extra spaces, replace spaces with underscores, and convert to lowercase.
    """
    name = "".join(ch for ch in name if unicodedata.category(ch)[0] != "C")
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("utf-8")
    return name.strip().replace(" ", "_").lower()

def download_video(url, output_path):
    """
    Download the video from the given URL and save it to output_path.
    A tqdm progress bar shows the download progress.
    """
    response = requests.get(url, stream=True, verify=False)
    if response.status_code == 200:
        total_size = int(response.headers.get("Content-Length", 0))
        with open(output_path, "wb") as f, tqdm(
            total=total_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=os.path.basename(output_path)
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
    else:
        print(f"Failed to download {url} (HTTP {response.status_code})")

def main():
    # Hardcoded input CSV file and output folder
    input_csv = "links_videos_UFPE.csv"
    output_folder = "UFPE"
    os.makedirs(output_folder, exist_ok=True)
    
    # Read the CSV file (assumes UTF-8 encoding)
    with open(input_csv, newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        rows = list(reader)
    
    # Loop over each row with a progress bar
    for row in tqdm(rows, desc="Downloading videos"):
        palavra = row["Palavra"]
        video_url = row["Link"]
        
        # Generate a safe filename based on the "Palavra" column
        filename = f"{sanitize_filename(palavra)}.mp4"
        output_path = os.path.join(output_folder, filename)
        
        # Download the video
        download_video(video_url, output_path)

if __name__ == "__main__":
    main()
