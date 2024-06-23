# TODOs:
- Conferir se a atualização do numero de intancias se manteve
- Listar exemplos errados
- Não pegar pré-treinado o modelo
- Inserir torchmetrics no script de avaliacao
- Ablations:
    - Conferir o efeito das transformações
    - Melhor modelo e ver quanto o tamanho do dataset influencia
        - Fixa a validação em 5 vídeos e vai aumentando a quantidade de vídeos de treino


- Testes para fazermos: 
    - Augmentations:
        
        ***(1) temporal displacement in the 10 frames
        obtained by summarizing. This is done by adding a random
        value between 1 and 4 to the position of each of these 10
        frames of each video; (2) horizontal mirroring; and (3)
        zoom from 5 to 15% on the original frames.***
        
        - Translacao
        - Espelhamento horizontal
        - Redimensionamento
        - Deslocamento Temporal
        - Mudanca de frames
        - Zoom (5% a 15% em cada quadro)
        
    - Aplicar as transformações de imagem dos modelos em si
        - Normalizar e ver o que mais eles fizeram ao passar os dados para o modelo
    - Fazer o CLIP funcionar


! Disclaimer
    - Arquivos faltantes MINDS:
        - Sinalizador 3:
            - Aluno (student)
            - América (America)
            - Cinco (Five)
        - Sinalizador 4:
            Filho (Son)
        - Sinalizador 9:
            - Amarelo (Yellow)
            - Banheiro (Bathroom)
            - Conhecer (To know)
            - Esquina (Corner)
            - Medo (Fear)
    - Arquivos extras:
        - Sinalizador 7:
            - Barulho (Noise) - 2 vídeos extra
            - Filho (Son) - 1 vídeo extra

! Dados sobre o dataset MINDS:
    - 12 atores
    - 20 sinais
    - Cada atores gravou 5x o mesmo sinal
    - 100 vídeos para cada sinal
    - 1200 vídeos no total

! Experimento MINDS:
    - split aleatório de 75/25 treino/teste, por sinal (paper)
        - ou pode-se fazer validação cruzada com 12-folds, por sinal (tese doutorado)
        - 1200 amostras, 5 quadros de vídeo cada amostra, divididas em 12-folds, 11 para treinamento e 1 para teste (=100 amostras para teste)
    - 5 quadros escala de cinza 224x224 (paper)
    - Treinamento com data augmentation (1100 amostras originais + 2200 geradas), 3200 para treino, 100 para teste 
    - CNN3D, LR 0,001, Adam, categorical_crossentropy, acurácia, batch_size=128, early_stop=50 epochs
    - acurácia média para cada fold
    -samples were randomized and
    partitioned into train (75%) and test (25%) sets, per sign.

! Experimento tese:
    - 1200 amostras divididas em 12-folds, 11 para treino, 1 para teste (100 amostras) *por sinal*
    - Conjunto de treinamento passou por DA (1100 amostras + 2200 sintéticas (depois do DA) = 3300) *por sinal*
    - Conjunto de validação = 100 amostras (que passaram por DA) *por sinal*
    - 
    