import pandas as pd

# Lê o arquivo
df_pop_anual = pd.read_csv('br_ibge_populacao_municipio.csv')

# Filtra só Nordeste e anos de interesse
NORDESTE = ['AL', 'BA', 'CE', 'MA', 'PB', 'PE', 'PI', 'RN', 'SE']
ANOS = [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]

df_pop_anual = df_pop_anual[
    (df_pop_anual['sigla_uf'].isin(NORDESTE)) &
    (df_pop_anual['ano'].isin(ANOS))
].copy()

# Renomeia pra ficar consistente com o dataset do INEP
df_pop_anual = df_pop_anual.rename(columns={
    'ano': 'NU_ANO_CENSO',
    'id_municipio': 'CO_MUNICIPIO',
    'populacao': 'POP_TOTAL'
})[['NU_ANO_CENSO', 'CO_MUNICIPIO', 'POP_TOTAL']]

print(df_pop_anual.shape)
print(df_pop_anual.head())

df_pop_anual.to_csv('populacao_anual_nordeste.csv', sep=';', encoding='utf-8', index=False)