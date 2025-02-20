import requests
from bs4 import BeautifulSoup
import pandas as pd
from unidecode import unidecode
import csv

def obter_videos():
    cabecalho = {'user-agent': 'Mozzila/5.0'}
    resposta = requests.get('https://edisciplinas.usp.br/mod/glossary/view.php?id=197645&mode&hook=ALL&sortkey&sortorder&fullsearch=0&page=-1',
                            headers=cabecalho)
    conteudo = resposta.text

    conteudo = BeautifulSoup(conteudo, 'html.parser')

    lista_videos = conteudo.find_all('source')
    lista_titulos = conteudo.find_all('h4')

    lista_videos_usp = []

    for video, titulo in zip(lista_videos, lista_titulos):
        src_link = video['src']
        titulo_video = unidecode(titulo.text)  # Normalização de caracteres

        lista_videos_usp.append((titulo_video, src_link))

    df_videos = pd.DataFrame(lista_videos_usp, columns=['Palavra', 'Link'])
    df_videos['Instituicao'] = 'USP' 

    return df_videos

df_videos = obter_videos()
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
print(df_videos)

# Salvar DataFrame em um arquivo CSV
nome_arquivo = 'links_videos_usp.csv'
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
