import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select
from bs4 import BeautifulSoup
from selenium.webdriver.chrome.options import Options
import pandas as pd
import unidecode
import csv


options = Options()
driver = webdriver.Chrome(options=options)
driver.get('http://www.acessibilidadebrasil.org.br/libras_3/')


time.sleep(5)

letras_menu = driver.find_elements(By.XPATH, '//div[@id="filter-letter"]/ul/li/a')
link_base = 'http://www.acessibilidadebrasil.org.br/libras_3/'
lista_option_letra = []
links_videos = []
palavras_site = []

for letra in letras_menu:
    letra.click()  
    html = driver.page_source 
    soup = BeautifulSoup(html, 'html.parser')

    options = soup.find_all('option')[1:]
    
    for option in options:
        option_text = option.text
        lista_option_letra.append(option_text)
        
    select_element = Select(driver.find_element(by=By.ID, value='input-palavras'))
    for palavra in lista_option_letra:
        select_element.select_by_visible_text(palavra)
        palavrinha = unidecode.unidecode(palavra)
        palavras_site.append(palavrinha.lower())
        html = driver.page_source
        soup = BeautifulSoup(html, 'html.parser')
        video_tag = soup.find("video", id="videojs")
        source_tag = video_tag.find("source")
        video_url = source_tag["src"]
        links_videos.append(link_base + video_url)
        
    lista_option_letra = []
    
driver.quit() 

#print(palavras_site)
#print(links_videos)

video_acessibilidade = []

for video, titulo in zip(links_videos, palavras_site):
    video_acessibilidade.append((titulo, video))

df_videos = pd.DataFrame(video_acessibilidade, columns=['Palavra', 'Link'])
df_videos['Instituicao'] = 'ACESSIBILIDADE_BRASIL' 

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
print(df_videos)


nome_arquivo = 'links_videos_acessibilidade_brasil.csv'
formato = {
    'Palavra': str,
    'Link': str,
    'Instituicao': str
}

with open(nome_arquivo, 'w', newline='', encoding='utf-8') as arquivo_csv:
    escritor = csv.DictWriter(arquivo_csv, fieldnames=formato.keys())
    escritor.writeheader()
    for _, linha in df_videos.iterrows():
        escritor.writerow({campo: formato[campo](valor) for campo, valor in linha.items()})


