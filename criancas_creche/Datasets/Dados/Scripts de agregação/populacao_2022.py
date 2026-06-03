import pandas as pd
import re
import os

ESTADOS = ['AL', 'BA', 'CE', 'MA', 'PB', 'PE', 'PI', 'RN', 'SE']

frames = []

for uf in ESTADOS:
    arquivo = f'tabela9514_UF_MUN_{uf}.xlsx'
    
    if not os.path.exists(arquivo):
        print(f'{uf}: arquivo não encontrado, pulando...')
        continue
    
    # Lê a aba correta pulando as 6 linhas de cabeçalho
    df = pd.read_excel(
        arquivo,
        sheet_name='População residente (Pessoas)',
        header=None,
        skiprows=6
    )
    
    # Renomeia as colunas que interessam
    df.columns = ['MUNICIPIO_IBGE', 'FORMA', 'POP_TOTAL', 'POP_0_4'] + \
                 [f'COL_{i}' for i in range(len(df.columns) - 4)]
    
    # Filtra só linhas com dados reais — remove estado, fonte, vazias
    df = df[df['FORMA'] == 'Total'].copy()
    
    # Remove a linha do próprio estado (ex: "Alagoas")
    df = df[df['MUNICIPIO_IBGE'].str.contains(r'\(', na=False)].copy()
    
    # Extrai nome do município e UF do formato "Água Branca (AL)"
    df['NO_MUNICIPIO_IBGE'] = df['MUNICIPIO_IBGE'].str.extract(r'^(.+?)\s*\(')[0].str.strip()
    df['SG_UF'] = df['MUNICIPIO_IBGE'].str.extract(r'\((\w+)\)')[0]
    
    # Mantém só o que interessa
    df = df[['NO_MUNICIPIO_IBGE', 'SG_UF', 'POP_TOTAL', 'POP_0_4']].copy()
    
    frames.append(df)
    print(f'{uf}: {len(df)} municípios')

# Junta tudo
df_pop = pd.concat(frames, ignore_index=True)
print(f'\nTotal: {df_pop.shape}')
print(df_pop.head())

df_pop.to_csv('populacao_nordeste_2022.csv', sep=';', encoding='utf-8', index=False)