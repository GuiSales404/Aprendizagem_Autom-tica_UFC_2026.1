import pandas as pd
import os

COLUNAS = [
    "NU_ANO_CENSO", "CO_ENTIDADE", "CO_MUNICIPIO", 
    "NO_MUNICIPIO", "SG_UF", "TP_DEPENDENCIA", "TP_LOCALIZACAO",
    "QT_MAT_INF_CRE", "QT_MAT_INF_CRE_INT", "QT_MAT_BAS_0_3", 
    "QT_MAT_BAS_BRANCA", "QT_MAT_BAS_PRETA", "QT_MAT_BAS_PARDA", 
    "QT_MAT_BAS_AMARELA", "QT_MAT_BAS_INDIGENA"
]

NORDESTE = ['AL', 'BA', 'CE', 'MA', 'PB', 'PE', 'PI', 'RN', 'SE']

ANOS = [2018, 2019, 2020, 2021, 2022, 2023, 2024]

frames = []

for ano in ANOS:
    arquivo = f'microdados_ed_basica_{ano}.csv'
    
    if not os.path.exists(arquivo):
        print(f'{ano}: arquivo não encontrado, pulando...')
        continue
    
    df_ano = pd.read_csv(arquivo, sep=';', encoding='latin1',
                         usecols=COLUNAS, low_memory=False)
    
    df_ano = df_ano[df_ano['SG_UF'].isin(NORDESTE)]
    
    frames.append(df_ano)
    print(f'{ano}: {df_ano.shape[0]} escolas')

# Empilha 2018-2024
df_historico = pd.concat(frames, ignore_index=True)
print(f'\nHistórico 2018-2024: {df_historico.shape}')

# Carrega o 2025 que você já gerou
df_2025 = pd.read_csv('nordeste_2025.csv', sep=';', encoding='utf-8')

# Junta tudo
df_total = pd.concat([df_historico, df_2025], ignore_index=True)
print(f'Total geral: {df_total.shape}')

df_total.to_csv('nordeste_2018_2025.csv', sep=';', encoding='utf-8', index=False)