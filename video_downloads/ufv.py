import requests
from bs4 import BeautifulSoup
import csv
from urllib.parse import urljoin

# Base URL for the dictionary
base_url = "https://sistemas.cead.ufv.br/capes/dicionario/"

# Fetch the main page that lists all sign links
response = requests.get(base_url)
response.encoding = 'utf-8'
main_page = response.text

# Parse the main page
soup_main = BeautifulSoup(main_page, 'html.parser')

# Find all links to sign pages (the URLs contain a query parameter like '?cadastros=')
sign_links = []
for a in soup_main.find_all('a', href=True):
    href = a['href']
    if '?cadastros=' in href:
        full_url = urljoin(base_url, href)
        if full_url not in sign_links:
            sign_links.append(full_url)

print(f"Found {len(sign_links)} sign pages.")

# Prepare a list to hold the rows for the CSV
data_rows = []

# Loop through each sign page to extract the sign name and video link
for link in sign_links:
    try:
        sign_resp = requests.get(link)
        sign_resp.encoding = 'utf-8'
        sign_page = sign_resp.text
        soup_sign = BeautifulSoup(sign_page, 'html.parser')
        
        # Extract the sign name.
        # In our sample 'biology_antebraco.html' page the title is like "Antebraço | Dicionário de Libras"
        # (see :contentReference[oaicite:0]{index=0})
        title_text = soup_sign.title.string if soup_sign.title else "Unknown"
        sign_name = title_text.split("|")[0].strip()
        
        # Extract the video download link.
        # First, we look for a <video> tag. If not found, we try for a <source> tag with type "video/mp4".
        video_link = None
        video_tag = soup_sign.find('video')
        if video_tag and video_tag.has_attr('src'):
            video_link = video_tag['src']
        else:
            source_tag = soup_sign.find('source', type="video/mp4")
            if source_tag and source_tag.has_attr('src'):
                video_link = source_tag['src']
        
        # If a video link is found, ensure it is a full URL and add to the list.
        if video_link:
            video_link = urljoin(link, video_link)
            data_rows.append([sign_name, video_link])
            print(f"Processed {sign_name}: {video_link}")
        else:
            print(f"Video not found for {sign_name} at {link}")
    except Exception as e:
        print(f"Error processing {link}: {e}")

# Write the collected data into a CSV file
with open('signs.csv', 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['sign', 'link'])
    writer.writerows(data_rows)

print("CSV file 'signs.csv' created successfully!")
