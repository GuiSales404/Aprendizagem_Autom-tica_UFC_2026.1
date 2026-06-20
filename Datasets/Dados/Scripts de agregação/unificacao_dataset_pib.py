import pandas as pd

df_2018_2025 = pd.read_csv('dataset_2018_2025.csv', sep=';', encoding='utf-8')
df_2022      = pd.read_csv('dataset_2022.csv', sep=';', encoding='utf-8')
df_pib       = pd.read_csv('pib_nordeste_2018_2025.csv', sep=';', encoding='utf-8')

# Merge por município + ano
df_2018_2025 = pd.merge(df_2018_2025, df_pib, on=['CO_MUNICIPIO', 'NU_ANO_CENSO'], how='left')

# No dataset 2022, filtra só o PIB de 2022
df_pib_2022 = df_pib[df_pib['NU_ANO_CENSO'] == 2022][['CO_MUNICIPIO', 'PIB']]
df_2022 = pd.merge(df_2022, df_pib_2022, on='CO_MUNICIPIO', how='left')

# Diagnóstico
print(f'2018-2025: {df_2018_2025.shape} | Nulos PIB: {df_2018_2025["PIB"].isna().sum()}')
print(f'2022: {df_2022.shape} | Nulos PIB: {df_2022["PIB"].isna().sum()}')

df_2018_2025.to_csv('dataset_2018_2025.csv', sep=';', encoding='utf-8', index=False)
df_2022.to_csv('dataset_2022.csv', sep=';', encoding='utf-8', index=False)

print('\nDatasets atualizados com PIB!')