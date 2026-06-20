import pandas as pd

df_2018_2025 = pd.read_csv('dataset_2018_2025.csv', sep=';', encoding='utf-8')
df_2022      = pd.read_csv('dataset_2022.csv', sep=';', encoding='utf-8')
df_idh       = pd.read_csv('idh_nordeste_2010.csv', sep=';', encoding='utf-8')

colunas_idh = ['IDH_PROP_POBREZA', 'IDH_PROP_POBREZA_CRIANCAS', 'IDH_PROP_VULNER_POBREZA',
               'IDH_GINI', 'IDH_RENDA_PC', 'IDH_EXPECT_ANOS_ESTUDO',
               'IDH_TAXA_FREQ_0_3', 'IDH_RAZAO_DEPENDENCIA']

df_2018_2025 = df_2018_2025.drop(columns=[c for c in colunas_idh if c in df_2018_2025.columns])
df_2022      = df_2022.drop(columns=[c for c in colunas_idh if c in df_2022.columns])

df_2018_2025 = pd.merge(df_2018_2025, df_idh, on='CO_MUNICIPIO', how='left')
df_2022      = pd.merge(df_2022, df_idh, on='CO_MUNICIPIO', how='left')

# Propaga para todos os anos
df_2018_2025 = df_2018_2025.sort_values(['CO_MUNICIPIO', 'NU_ANO_CENSO'])
for col in colunas_idh:
    df_2018_2025[col] = df_2018_2025.groupby('CO_MUNICIPIO')[col].transform(
        lambda x: x.ffill().bfill()
    )

print(f'2018-2025: {df_2018_2025.shape} | Nulos IDH_GINI: {df_2018_2025["IDH_GINI"].isna().sum()}')
print(f'2022:      {df_2022.shape}      | Nulos IDH_GINI: {df_2022["IDH_GINI"].isna().sum()}')

df_2018_2025.to_csv('dataset_2018_2025.csv', sep=';', encoding='utf-8', index=False)
df_2022.to_csv('dataset_2022.csv', sep=';', encoding='utf-8', index=False)

print('\nDatasets atualizados com IDH!')