import pandas as pd

df_2018_2025  = pd.read_csv('dataset_2018_2025.csv', sep=';', encoding='utf-8')
df_2022       = pd.read_csv('dataset_2022.csv', sep=';', encoding='utf-8')
df_renda_comp = pd.read_csv('composicao_renda_nordeste_2022.csv', sep=';', encoding='utf-8')

colunas_renda_comp = ['PERC_RENDA_TRABALHO', 'PERC_RENDA_OUTRAS_FONTES']

df_2018_2025 = df_2018_2025.drop(columns=[c for c in colunas_renda_comp if c in df_2018_2025.columns])
df_2022      = df_2022.drop(columns=[c for c in colunas_renda_comp if c in df_2022.columns])

df_2018_2025 = pd.merge(df_2018_2025, df_renda_comp, on='CO_MUNICIPIO', how='left')
df_2022      = pd.merge(df_2022, df_renda_comp, on='CO_MUNICIPIO', how='left')

# Propaga para todos os anos
df_2018_2025 = df_2018_2025.sort_values(['CO_MUNICIPIO', 'NU_ANO_CENSO'])
for col in colunas_renda_comp:
    df_2018_2025[col] = df_2018_2025.groupby('CO_MUNICIPIO')[col].transform(
        lambda x: x.ffill().bfill()
    )

print(f'2018-2025: {df_2018_2025.shape} | Nulos: {df_2018_2025["PERC_RENDA_TRABALHO"].isna().sum()}')
print(f'2022:      {df_2022.shape}      | Nulos: {df_2022["PERC_RENDA_TRABALHO"].isna().sum()}')

df_2018_2025.to_csv('dataset_2018_2025.csv', sep=';', encoding='utf-8', index=False)
df_2022.to_csv('dataset_2022.csv', sep=';', encoding='utf-8', index=False)

print('\nDatasets atualizados com composição de renda!')