import pandas as pd

# Carrega o dataset limpo
df = pd.read_csv('dataset_final_2022_clean.csv', sep=';', encoding='utf-8')

print(f'Antes: {df.shape}')

# Variáveis a remover
remover = [
    'PROP_DOM_SEM_CONJUGE',
    'PROP_DOM_SEM_CONJUGE_HOMEM',
    'PROP_MULHER_ENTRE_RESP_SEM_CONJUGE',
    'LOG_POP_0_4',
    'PROP_POP_0_4',
    'LOG_RENDA_MEDIANA_RESPONSAVEL',
    'IDH_PROP_POBREZA',
    'IDH_PROP_VULNER_POBREZA',
    'RAZAO_OUTRAS_FONTES_TRABALHO',
    'CRECHE_POR_1000_HAB'
]

df = df.drop(columns=remover)

print(f'Depois: {df.shape}')
print(f'\nColunas restantes ({len(df.columns)}):')
print(df.columns.tolist())

df.to_csv('dataset_modelo_2022.csv', sep=';', encoding='utf-8', index=False)
print('\nSalvo como dataset_modelo_2022.csv')