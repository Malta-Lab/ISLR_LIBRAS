import requests
from bs4 import BeautifulSoup
import unidecode
import pandas as pd
import csv

url = 'https://videos.nals.cce.ufsc.br/SignBank/V%C3%ADdeos/'
response = requests.get(url)
content = response.content

soup = BeautifulSoup(content, 'html.parser')


links = soup.find_all('a')

srcs = []
palavras = []
lista_videos_ufsc = []

# Itera sobre os links encontrados e imprime o texto do link, excluindo a extensão '.mp4'
for link in links[3:]:
    href = link.get('href')
    srcs.append('https://videos.nals.cce.ufsc.br'+href)

    palavra = link.text
    palavra = palavra.replace('.mp4', '')
    palavra = unidecode.unidecode(palavra)
    palavras.append(palavra)

for src_link, titulo_video in zip(srcs, palavras):
    lista_videos_ufsc.append((titulo_video, src_link))

df_videos = pd.DataFrame(lista_videos_ufsc, columns=['Palavra', 'Link'])
df_videos['Instituicao'] = 'UFSC SignBank' 

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
print(df_videos)

nome_arquivo = 'links_videos_ufsc_signbank.csv'
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

print("Dados salvos em arquivo CSV:", nome_arquivo)
