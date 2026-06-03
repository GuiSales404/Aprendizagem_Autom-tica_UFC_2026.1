import pandas as pd
import unicodedata

def normalizar(txt):
    if not isinstance(txt, str):
        return txt
    txt = txt.strip().lower()
    txt = ''.join(c for c in unicodedata.normalize('NFD', txt)
                  if unicodedata.category(c) != 'Mn')
    return txt

CORRECOES = {
    'graccho cardoso': 'gracho cardoso',
    'unas': 'una'
}

def normalizar_com_correcao(txt):
    resultado = normalizar(txt)
    return CORRECOES.get(resultado, resultado)

df_2018_2025 = pd.read_csv('dataset_2018_2025.csv', sep=';', encoding='utf-8')
df_2022      = pd.read_csv('dataset_2022.csv', sep=';', encoding='utf-8')
df_ocup      = pd.read_csv('ocupacao_nordeste_2022.csv', sep=';', encoding='utf-8')

# Normaliza nomes
df_ocup['NOME_KEY']      = df_ocup['NO_MUNICIPIO_IBGE'].apply(normalizar_com_correcao)
df_2018_2025['NOME_KEY'] = df_2018_2025['NO_MUNICIPIO'].apply(normalizar_com_correcao)
df_2022['NOME_KEY']      = df_2022['NO_MUNICIPIO'].apply(normalizar_com_correcao)

# Remove colunas de ocupação antigas se já existirem (evita duplicatas ao rerodar)
colunas_ocup = ['TOTAL_HOMENS', 'HOMENS_OCUPADOS', 'TOTAL_MULHERES', 'MULHERES_OCUPADAS']
df_2018_2025 = df_2018_2025.drop(columns=[c for c in colunas_ocup if c in df_2018_2025.columns])
df_2022      = df_2022.drop(columns=[c for c in colunas_ocup if c in df_2022.columns])

# Merge
df_2018_2025 = pd.merge(df_2018_2025,
                         df_ocup[['NOME_KEY', 'SG_UF'] + colunas_ocup],
                         on=['NOME_KEY', 'SG_UF'], how='left')

df_2022 = pd.merge(df_2022,
                   df_ocup[['NOME_KEY', 'SG_UF'] + colunas_ocup],
                   on=['NOME_KEY', 'SG_UF'], how='left')

# Remove coluna auxiliar
df_2018_2025 = df_2018_2025.drop(columns=['NOME_KEY'])
df_2022      = df_2022.drop(columns=['NOME_KEY'])

# Diagnóstico
print(f'2018-2025: {df_2018_2025.shape} | Nulos MULHERES_OCUPADAS: {df_2018_2025["MULHERES_OCUPADAS"].isna().sum()}')
print(f'2022:      {df_2022.shape}      | Nulos MULHERES_OCUPADAS: {df_2022["MULHERES_OCUPADAS"].isna().sum()}')

df_2018_2025.to_csv('dataset_2018_2025.csv', sep=';', encoding='utf-8', index=False)
df_2022.to_csv('dataset_2022.csv', sep=';', encoding='utf-8', index=False)

print('\nDatasets atualizados com ocupação!')