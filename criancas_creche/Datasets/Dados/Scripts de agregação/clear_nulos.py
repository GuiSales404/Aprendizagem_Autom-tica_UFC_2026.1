import pandas as pd
import numpy as np

# Carrega o dataset
dataset_2022_total = pd.read_csv('./dataset_final_2022.csv', sep=';')

# Diagnóstico antes
print(f'Antes: {dataset_2022_total.shape}')
print(f'Total de nulos por coluna:')
print(dataset_2022_total.isnull().sum()[dataset_2022_total.isnull().sum() > 0])

# Municípios com nulo na variável alvo
nulos_alvo = dataset_2022_total[dataset_2022_total['TAXA_CRE_INT'].isna()]
print(f'\nMunicípios com TAXA_CRE_INT nula ({len(nulos_alvo)}):')
print(nulos_alvo[['NO_MUNICIPIO', 'SG_UF']])

# Remove todas as linhas com qualquer nulo
dataset_2022_clean = dataset_2022_total.dropna()

print(f'\nDepois: {dataset_2022_clean.shape}')
print(f'Nulos restantes: {dataset_2022_clean.isnull().sum().sum()}')

# Salva
dataset_2022_clean.to_csv('./dataset_final_2022_clean.csv', sep=';', encoding='utf-8', index=False)
print('\nSalvo como dataset_final_2022_clean.csv')