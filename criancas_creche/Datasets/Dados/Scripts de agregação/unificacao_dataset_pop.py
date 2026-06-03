import pandas as pd
import unicodedata

def normalizar(txt):
    if not isinstance(txt, str):
        return txt
    txt = txt.strip().lower()
    txt = ''.join(c for c in unicodedata.normalize('NFD', txt)
                  if unicodedata.category(c) != 'Mn')
    return txt

# ── Carrega os datasets ──────────────────────────────────────────
df_mun      = pd.read_csv('municipio_2018_2025.csv', sep=';', encoding='utf-8')
df_pop_anual = pd.read_csv('populacao_anual_nordeste.csv', sep=';', encoding='utf-8')
df_pop_2022  = pd.read_csv('populacao_nordeste_2022.csv', sep=';', encoding='utf-8')

# ── Merge 1: população anual (por CO_MUNICIPIO + ano) ───────────
df_merged = pd.merge(
    df_mun, df_pop_anual,
    on=['NU_ANO_CENSO', 'CO_MUNICIPIO'],
    how='left'
)

# ── Merge 2: população 0-4 anos do censo 2022 (por nome + UF) ───
df_pop_2022['NOME_KEY'] = df_pop_2022['NO_MUNICIPIO_IBGE'].apply(normalizar)
df_merged['NOME_KEY']   = df_merged['NO_MUNICIPIO'].apply(normalizar)

df_merged = pd.merge(
    df_merged,
    df_pop_2022[['NOME_KEY', 'SG_UF', 'POP_0_4']],
    on=['NOME_KEY', 'SG_UF'],
    how='left'
)

df_merged = df_merged.drop(columns=['NOME_KEY'])

# ── Diagnóstico ─────────────────────────────────────────────────
print(f'Shape final: {df_merged.shape}')
print(f'POP_TOTAL nulos: {df_merged["POP_TOTAL"].isna().sum()}')
print(f'POP_0_4 nulos: {df_merged["POP_0_4"].isna().sum()}')
print(df_merged.head())

# ── Salva dataset completo 2018-2025 ────────────────────────────
df_merged.to_csv('dataset_2018_2025.csv', sep=';', encoding='utf-8', index=False)

# ── Salva dataset corte 2022 ─────────────────────────────────────
df_2022 = df_merged[df_merged['NU_ANO_CENSO'] == 2022].copy()
df_2022.to_csv('dataset_2022.csv', sep=';', encoding='utf-8', index=False)

print(f'\nDataset 2018-2025: {df_merged.shape}')
print(f'Dataset 2022: {df_2022.shape}')