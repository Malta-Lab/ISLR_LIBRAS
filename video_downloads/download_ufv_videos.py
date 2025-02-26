from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.options import Options as ChromeOptions
import time
import csv

def scrape_sinalario_page(driver, wait, page_url):
    """
    EXACT function you already have for single/multi-slide logic.
    Navigates to page_url, tries #carousel-example-generic-one, etc.
    Returns {word -> video_url}.
    """
    driver.get(page_url)
    time.sleep(2)

    # Attempt to find the carousel container (e.g. #carousel-example-generic-one)
    try:
        carousel_selector = "#carousel-example-generic-one"
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, carousel_selector)))
    except:
        # fallback
        print(f"  No carousel container found for {page_url}. Using fallback.")
        return scrape_single_slide(driver, wait)

    # Check for indicators
    indicators_sel = f"{carousel_selector} ol.carousel-indicators li"
    indicators = driver.find_elements(By.CSS_SELECTOR, indicators_sel)
    num_slides = len(indicators)
    print(f"  Found {num_slides} carousel slides for {page_url}.")

    if num_slides <= 1:
        return scrape_single_slide(driver, wait)
    else:
        return scrape_multi_slide(driver, wait, num_slides)

def scrape_multi_slide(driver, wait, num_slides):
    """
    EXACT function you already have. 
    Cycles slides with jQuery, collects links, visits each link for video.
    Returns {word -> video_url}.
    """
    word_links = {}

    for i in range(num_slides):
        js_cmd = f"$('#carousel-example-generic-one').carousel({i});"
        driver.execute_script(js_cmd)

        slide_selector = f"div.item.active[data-posicao='{i}']"
        try:
            active_slide = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, slide_selector)))
        except:
            print(f"  Slide {i} not found/active. Skipping.")
            continue

        links = active_slide.find_elements(By.TAG_NAME, "a")
        for link in links:
            text = link.text.strip()
            href = link.get_attribute("href")
            if text and href:
                word_links[text] = href

        time.sleep(1)

    # Now visit each link to extract videos
    video_links = {}
    for word, url in word_links.items():
        driver.get(url)
        try:
            video = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "div#videos video")))
            src = video.get_attribute("src")
            if not src:
                try:
                    source = video.find_element(By.TAG_NAME, "source")
                    src = source.get_attribute("src")
                except:
                    src = None
            video_links[word] = src
            print(f"    -> {word}: {src}")
        except Exception as e:
            print(f"    -> Video not found for '{word}' in {url}: {e}")
        time.sleep(1)

    return video_links

def scrape_single_slide(driver, wait):
    """
    EXACT function you already have.
    Fallback for single-slide or no .carousel-indicators.
    Returns {word -> video_url}.
    """
    try_selectors = [
        "div.item.active#results",
        "div.item#results"
    ]

    single_slide = None
    for sel in try_selectors:
        found = driver.find_elements(By.CSS_SELECTOR, sel)
        if found:
            single_slide = found[0]
            break

    if not single_slide:
        print("  Could not find any 'div.item#results' for single-slide fallback.")
        return {}

    # Gather <a> from this single slide
    links = single_slide.find_elements(By.TAG_NAME, "a")
    word_links = {}
    for link in links:
        text = link.text.strip()
        href = link.get_attribute("href")
        if text and href:
            word_links[text] = href

    # Extract videos
    video_links = {}
    for word, url in word_links.items():
        driver.get(url)
        try:
            video = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "div#videos video")))
            src = video.get_attribute("src")
            if not src:
                try:
                    source = video.find_element(By.TAG_NAME, "source")
                    src = source.get_attribute("src")
                except:
                    src = None
            video_links[word] = src
            print(f"    -> {word}: {src}")
        except Exception as e:
            print(f"    -> Video not found for '{word}' in {url}: {e}")
        time.sleep(1)

    return video_links

def main():
    chromedriver_path = "/usr/local/bin/chromedriver"
    service = ChromeService(executable_path=chromedriver_path)
    chrome_options = ChromeOptions()
    chrome_options.add_argument("--headless")
    driver = webdriver.Chrome(service=service, options=chrome_options)
    driver.implicitly_wait(10)
    wait = WebDriverWait(driver, 10)

    try:
        # Hard-coded subcategories for Sinalário
        # (the same logic you had before for Biologia, Letras, Matemática)
        sinalario_subcats = {
            "Biologia":    "https://sistemas.cead.ufv.br/capes/dicionario/?cadastros=antebraco&term=sinalario&value=biologia",
            "Letras":      "https://sistemas.cead.ufv.br/capes/dicionario/?cadastros=abstrato-1&term=sinalario&value=letras",
            "Matemática":  "https://sistemas.cead.ufv.br/capes/dicionario/?cadastros=cinco&term=sinalario&value=matematica"
        }

        # Hard-coded subcategories for Temas
        # Fill in as many as you like
        # E.g. "Alimentos", "Animais e Insetos", "Comemorações", etc.
        temas_subcats = {
            "Alimentos": "https://sistemas.cead.ufv.br/capes/dicionario/?cadastros=abacate&term=temas&value=alimentos",
            "Animais e Insetos": "https://sistemas.cead.ufv.br/capes/dicionario/?cadastros=aguia&term=temas&value=animais-e-insetos",
            "Comemorações": "https://sistemas.cead.ufv.br/capes/dicionario/?cadastros=aniversario&term=temas&value=comemoracoes",
            "Comunicação e Eletrônicos": "https://sistemas.cead.ufv.br/capes/dicionario/?cadastros=aplicativo&term=temas&value=comunicacao-e-eletronicos",
            "Construção": "https://sistemas.cead.ufv.br/capes/dicionario/?cadastros=aco&term=temas&value=construcao",
            "Cores": "https://sistemas.cead.ufv.br/capes/dicionario/?cadastros=amarelo&term=temas&value=cores",
            "Corpo Humano": "https://sistemas.cead.ufv.br/capes/dicionario/?cadastros=antebraco&term=temas&value=corpo-humano",
            "Cumprimentos": "https://sistemas.cead.ufv.br/capes/dicionario/?cadastros=adeus&term=temas&value=cumprimentos",
            "Dinheiro": "https://sistemas.cead.ufv.br/capes/dicionario/?cadastros=aluguel&term=temas&value=dinheiro",
            "Escola": "https://sistemas.cead.ufv.br/capes/dicionario/?cadastros=abstrato&term=temas&value=escola",
            "Disciplina": "https://sistemas.cead.ufv.br/capes/dicionario/?cadastros=agramatical&term=temas&value=disciplina",
            "Esporte e Diversão": "https://sistemas.cead.ufv.br/capes/dicionario/?cadastros=basquete&term=temas&value=esporte-e-diversao",
            "Instrumentos Musicais": "https://sistemas.cead.ufv.br/capes/dicionario/?cadastros=bateria&term=temas&value=instrumentos-musicais",
            "Lugares, Cidades e Países": "https://sistemas.cead.ufv.br/capes/dicionario/?cadastros=acre&term=temas&value=lugares-cidades-e-paises",
            "Meios de Transporte": "https://sistemas.cead.ufv.br/capes/dicionario/?cadastros=aviao&term=temas&value=meios-de-transporte",
            "Natureza": "https://sistemas.cead.ufv.br/capes/dicionario/?cadastros=acetona&term=temas&value=natureza",
            "Números": "https://sistemas.cead.ufv.br/capes/dicionario/?cadastros=algoritmo&term=temas&value=numeros",
            "Objetos": "https://sistemas.cead.ufv.br/capes/dicionario/?cadastros=abajur&term=temas&value=objetos",
            "Pessoas e Família": "https://sistemas.cead.ufv.br/capes/dicionario/?cadastros=acessibilidade&term=temas&value=pessoas-e-familia",
            "Profissões": "https://sistemas.cead.ufv.br/capes/dicionario/?cadastros=advogado&term=temas&value=profissoes",
            "Situações, Cotidiano e Eventos": "https://sistemas.cead.ufv.br/capes/dicionario/?cadastros=acidente-de-carro&term=temas&value=situacoes-cotidiano-e-eventos",
            "Tempo e Calendário": "https://sistemas.cead.ufv.br/capes/dicionario/?cadastros=abril&term=temas&value=tempo-e-calendario",
            "Verbos": "https://sistemas.cead.ufv.br/capes/dicionario/?cadastros=abencoar&term=temas&value=verbos",
            "Vestuário": "https://sistemas.cead.ufv.br/capes/dicionario/?cadastros=base&term=temas&value=vestuario",
        }

        # We'll store all results in one dictionary
        all_data = {}

        # 1) Sinalário
        print("\nScraping Sinalário subcategories...")
        for subcat_name, subcat_url in sinalario_subcats.items():
            print(f" Subcategory: {subcat_name} => {subcat_url}")
            result = scrape_sinalario_page(driver, wait, subcat_url)
            for w, vurl in result.items():
                all_data[w] = vurl

        # 2) Temas
        print("\nScraping Temas subcategories...")
        for subcat_name, subcat_url in temas_subcats.items():
            print(f" Subcategory: {subcat_name} => {subcat_url}")
            result = scrape_sinalario_page(driver, wait, subcat_url)
            for w, vurl in result.items():
                all_data[w] = vurl

        # (If you also want Configuração de Mão, you can do a 3rd dictionary here.)

        # 3) Write everything to CSV
        csv_filename = "words_ufv_test.csv"
        with open(csv_filename, mode='w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ["word", "video_url"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for word, video_url in all_data.items():
                writer.writerow({"word": word, "video_url": video_url})

        print(f"\nAll data saved to {csv_filename}.")

    finally:
        driver.quit()

if __name__ == "__main__":
    main()
