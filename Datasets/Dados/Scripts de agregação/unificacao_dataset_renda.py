import pandas as pd

# ── Carrega os datasets já processados ──────────────────────────
df_2018_2025 = pd.read_csv('dataset_2018_2025.csv', sep=';', encoding='utf-8')
df_2022      = pd.read_csv('dataset_2022.csv', sep=';', encoding='utf-8')
df_renda     = pd.read_csv('renda_nordeste_2022.csv', sep=';', encoding='utf-8')

# ── Merge renda nos dois datasets (por CO_MUNICIPIO) ────────────
df_2018_2025 = pd.merge(df_2018_2025, df_renda, on='CO_MUNICIPIO', how='left')
df_2022      = pd.merge(df_2022, df_renda, on='CO_MUNICIPIO', how='left')

# ── Diagnóstico ─────────────────────────────────────────────────
print(f'2018-2025: {df_2018_2025.shape}')
print(f'Nulos renda média:   {df_2018_2025["RENDA_MEDIA_RESPONSAVEL"].isna().sum()}')
print(f'Nulos renda mediana: {df_2018_2025["RENDA_MEDIANA_RESPONSAVEL"].isna().sum()}')
print(f'\n2022: {df_2022.shape}')
print(f'Nulos renda média:   {df_2022["RENDA_MEDIA_RESPONSAVEL"].isna().sum()}')
print(f'Nulos renda mediana: {df_2022["RENDA_MEDIANA_RESPONSAVEL"].isna().sum()}')

# ── Salva ────────────────────────────────────────────────────────
df_2018_2025.to_csv('dataset_2018_2025.csv', sep=';', encoding='utf-8', index=False)
df_2022.to_csv('dataset_2022.csv', sep=';', encoding='utf-8', index=False)

print('\nDatasets atualizados com renda!')