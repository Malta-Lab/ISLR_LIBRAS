import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import csv
import time

BASE_URL = 'https://libras.cin.ufpe.br/'
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
}
DELAY = 1  # Respectful scraping delay

def get_sign_links():
    """Extract sign names and their detail page URLs from the main page"""
    response = requests.get(BASE_URL, headers=HEADERS)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, 'html.parser')
    
    signs = []
    table = soup.find('table')
    for row in table.find_all('tr')[1:]:  # Skip header
        cols = row.find_all('td')
        if len(cols) < 1:
            continue
            
        link = cols[0].find('a')
        if link:
            sign_name = link.text.strip()
            sign_url = urljoin(BASE_URL, link['href'])
            signs.append((sign_name, sign_url))
    
    return signs

def get_video_links(sign_url):
    """Extract direct video links from a sign page"""
    time.sleep(DELAY)
    response = requests.get(sign_url, headers=HEADERS)
    if not response.ok:
        return []
    
    soup = BeautifulSoup(response.text, 'html.parser')
    videos = []
    
    for video in soup.find_all('video'):
        source = video.find('source')
        if source and source.get('src'):
            video_url = urljoin(BASE_URL, source['src'])
            videos.append(video_url)
    
    return videos[:3]  # Return max 3 videos

def create_csv():
    """Create CSV with sign, link, and dictionary columns"""
    with open('libras_dataset.csv', 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['sign', 'link', 'dictionary']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        signs = get_sign_links()
        for sign_name, sign_url in signs:
            print(f"Processing: {sign_name}")
            
            video_urls = get_video_links(sign_url)
            if not video_urls:
                continue
            
            # Create one row per articulador video
            for i, video_url in enumerate(video_urls, 1):
                writer.writerow({
                    'sign': sign_name,
                    'link': video_url,
                    'dictionary': 'UFPE'
                })

if __name__ == '__main__':
    create_csv()
    print("CSV file created successfully!")