import csv
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

def scrape_spreadthesign_diversos():
    """Scrapes the 'Diversos' category from SpreadTheSign and saves results to spread.csv."""

    # -- SETUP WEBDRIVER (Chrome in this example) --
    driver = webdriver.Chrome()
    driver.maximize_window()
    wait = WebDriverWait(driver, 10)

    try:
        # 1) GO TO THE DIVERSOS PAGE
        start_url = "https://www.spreadthesign.com/pt.br/search/by-category/1/diversos/"
        driver.get(start_url)

        # 2) OPTIONAL: ACCEPT COOKIES (if a cookie banner appears)
        #    Uncomment & adjust the selector if your local page needs it:
        # try:
        #     accept_button = wait.until(
        #         EC.element_to_be_clickable((By.ID, "accept-cookies-button"))
        #     )
        #     accept_button.click()
        # except:
        #     pass

        # Prepare CSV output
        csv_filename = "spread.csv"
        fieldnames = ["Palavra", "Link", "Instituicao"]

        with open(csv_filename, mode="w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            # We'll loop over all pages in the category
            while True:
                # 3) FIND ALL WORD LINKS IN LEFT COLUMN
                #    Each word is typically inside: div.search-result-title > a
                word_links = driver.find_elements(
                    By.CSS_SELECTOR,
                    "div.search-results div.search-result-title a"
                )
                print(f"Found {len(word_links)} words on this page.")

                # 4) ITERATE OVER EACH WORD ON THE CURRENT PAGE
                for link in word_links:
                    word_text = link.text.strip()  # e.g. "1" or "0"
                    print(f"  * Scraping word: {word_text}")

                    # Click it -> triggers JS that loads content into #show-result
                    link.click()

                    # Wait for the dynamic content in #show-result
                    # We'll wait for the container that has .search-result-content
                    try:
                        wait.until(
                            EC.presence_of_element_located(
                                (By.CSS_SELECTOR, "#show-result .search-result-content")
                            )
                        )
                    except:
                        print(f"    [!] Timed out waiting for video area for '{word_text}'.")
                        continue

                    # 5) Some words have multiple “Variant” tabs.
                    #    Let's find them: .nav.nav-tabs li a
                    variant_tabs = driver.find_elements(
                        By.CSS_SELECTOR, "#show-result .nav.nav-tabs li a"
                    )

                    if not variant_tabs:
                        # If no tabs, there's presumably just one variant (or no video).
                        # Attempt to get the single <video>.
                        extract_and_save_video(driver, writer, word_text)
                    else:
                        # Multiple variants. We click each tab, then get the video.
                        for idx, tab in enumerate(variant_tabs, start=1):
                            variant_label = f"{word_text}-variant{idx}"
                            try:
                                tab.click()
                                # Wait briefly for the video to refresh
                                time.sleep(1)
                                extract_and_save_video(driver, writer, variant_label)
                            except Exception as e:
                                print(f"    [!] Error clicking variant {idx} for '{word_text}': {e}")

                    # 6) (Optional) close the dynamic overlay or navigate back
                    #    In SpreadTheSign, the overlay is not an actual new page,
                    #    so we can just click the 'back' arrow or the close button.
                    try:
                        close_btn = driver.find_element(
                            By.CSS_SELECTOR, "#show-result button.result-close"
                        )
                        close_btn.click()
                        time.sleep(1)
                    except:
                        pass

                # 7) CHECK IF THERE IS A "PRÓXIMA PÁGINA" LINK
                try:
                    next_page_link = driver.find_element(
                        By.XPATH, "//a[contains(text(),'Próxima página')]"
                    )
                    # If found, click it
                    next_page_link.click()
                    # Wait a second for the new page to load
                    time.sleep(2)
                except:
                    print("No more pages found. Done scraping.")
                    break

    finally:
        driver.quit()

def extract_and_save_video(driver, writer, word_label):
    """
    Finds the <video> in #show-result, grabs its 'src',
    and writes a row to CSV (Palavra,Link,Instituicao).
    If no <video> is found, does nothing.
    """
    try:
        # We look for <video> inside #show-result
        video_elem = driver.find_element(By.CSS_SELECTOR, "#show-result video")
        src = video_elem.get_attribute("src")
        if src:
            writer.writerow({
                "Palavra": word_label,
                "Link": src,
                "Instituicao": "SpreadTheSign"
            })
            print(f"    -> Found video: {src}")
        else:
            # Sometimes <video> might have <source> inside
            source_elem = video_elem.find_element(By.TAG_NAME, "source")
            src = source_elem.get_attribute("src")
            if src:
                writer.writerow({
                    "Palavra": word_label,
                    "Link": src,
                    "Instituicao": "SpreadTheSign"
                })
                print(f"    -> Found video (source): {src}")
    except:
        print(f"    [!] No <video> found for '{word_label}'.")

if __name__ == "__main__":
    scrape_spreadthesign_diversos()
