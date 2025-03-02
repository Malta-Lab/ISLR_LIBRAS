import csv
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
# Optional: Uncomment these two lines to auto-manage ChromeDriver version.
# from webdriver_manager.chrome import ChromeDriverManager
# driver_service = ChromeService(ChromeDriverManager().install())

# --- Configuration ---
BASE_URL = 'https://libras.cin.ufpe.br'
CSV_FILE = 'libras_dataset.csv'
FINAL_PAGE = 69  # Total number of pages
DELAY = 1  # Delay (in seconds) after navigation actions

# --- Set up Selenium WebDriver ---
options = webdriver.ChromeOptions()
# Uncomment the next line to run in headless mode (without opening a browser window)
options.add_argument("--headless")
driver = webdriver.Chrome(options=options)  # Optionally add: service=driver_service

# --- Prepare CSV file for writing ---
with open(CSV_FILE, 'w', newline='', encoding='utf-8') as csvfile:
    fieldnames = ['sign', 'link', 'dictionary']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    # Process each paginated main page
    for page in range(1, FINAL_PAGE + 1):
        # Determine the URL for the current page.
        if page == 1:
            main_page_url = BASE_URL
        else:
            main_page_url = f"{BASE_URL}?page={page}"
        print(f"Processing main page: {main_page_url}")
        
        driver.get(main_page_url)
        # Wait for the table to be present on the page.
        try:
            table = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.XPATH, "//table[contains(@class, 'table')]"))
            )
        except Exception as e:
            print(f"Could not find the table on page {page}: {e}")
            continue

        # Extract table rows (skip header row)
        rows = table.find_elements(By.TAG_NAME, "tr")[1:]
        # Pre-scrape sign details from the current page:
        sign_details = []
        for row in rows:
            try:
                # Extract the sign name from the first cell (td)
                sign_name = row.find_element(By.XPATH, "./td[1]/a").text.strip()
                # Find all links with text "Exibir" in the row.
                exibir_links = row.find_elements(By.XPATH, ".//a[contains(., 'Exibir')]")
                if not exibir_links:
                    continue
                # Choose the first "Exibir" link.
                detail_page_url = exibir_links[0].get_attribute("href")
                sign_details.append((sign_name, detail_page_url))
            except Exception as e:
                print("Error extracting sign details from a row:", e)
                continue

        # Process each sign detail page
        for sign_name, detail_page_url in sign_details:
            print(f"  Processing sign: {sign_name} | Detail URL: {detail_page_url}")
            # Navigate to the detail page
            driver.get(detail_page_url)
            # Allow some time for the page to load
            time.sleep(DELAY)
            # Wait for at least one video element to appear (if available)
            try:
                WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.TAG_NAME, "video"))
                )
            except Exception as e:
                print(f"    No video found for {sign_name} on detail page: {e}")
                # Return to main page and proceed with the next sign
                driver.get(main_page_url)
                time.sleep(DELAY)
                continue

            # Scrape video URLs from the detail page:
            videos = driver.find_elements(By.TAG_NAME, "video")
            # For each video element, try to find the <source> child and extract the src attribute.
            for video in videos:
                try:
                    source = video.find_element(By.TAG_NAME, "source")
                    video_url = source.get_attribute("src")
                    if video_url:
                        writer.writerow({
                            'sign': sign_name,
                            'link': video_url,
                            'dictionary': 'UFPE'
                        })
                        print(f"    Found video: {video_url}")
                except Exception as e:
                    print(f"    Error extracting video URL for {sign_name}: {e}")
                    continue

            # After processing the detail page, return to the main page.
            driver.get(main_page_url)
            time.sleep(DELAY)

# Close the browser once done.
driver.quit()
print("CSV file created successfully!")
