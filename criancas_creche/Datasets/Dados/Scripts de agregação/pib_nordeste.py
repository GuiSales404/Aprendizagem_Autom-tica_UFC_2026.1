import pandas as pd

NORDESTE_CODIGOS = ['21', '22', '23', '24', '25', '26', '27', '28', '29']
ANOS = [2018, 2019, 2020, 2021, 2022, 2023]  # 2024 e 2025 vamos replicar do 2023

df_pib = pd.read_csv('br_ibge_pib_municipio.csv')

# Filtra Nordeste pelo prefixo do código do município
df_pib['id_municipio'] = df_pib['id_municipio'].astype(str)
df_pib = df_pib[df_pib['id_municipio'].str[:2].isin(NORDESTE_CODIGOS)].copy()
df_pib['id_municipio'] = df_pib['id_municipio'].astype(int)

# Filtra anos de interesse
df_pib = df_pib[df_pib['ano'].isin(ANOS)][['id_municipio', 'ano', 'pib']].copy()

# Renomeia
df_pib = df_pib.rename(columns={
    'id_municipio': 'CO_MUNICIPIO',
    'ano': 'NU_ANO_CENSO',
    'pib': 'PIB'
})

# Replica 2023 para 2024 e 2025
pib_2023 = df_pib[df_pib['NU_ANO_CENSO'] == 2023].copy()

pib_2024 = pib_2023.copy()
pib_2024['NU_ANO_CENSO'] = 2024

pib_2025 = pib_2023.copy()
pib_2025['NU_ANO_CENSO'] = 2025

df_pib = pd.concat([df_pib, pib_2024, pib_2025], ignore_index=True)
df_pib = df_pib.sort_values(['CO_MUNICIPIO', 'NU_ANO_CENSO']).reset_index(drop=True)

print(df_pib.shape)
print(df_pib.head(10))

df_pib.to_csv('pib_nordeste_2018_2025.csv', sep=';', encoding='utf-8', index=False)