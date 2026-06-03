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
df_dom       = pd.read_csv('domicilios_nordeste_2022.csv', sep=';', encoding='utf-8')

df_dom['NOME_KEY']       = df_dom['NO_MUNICIPIO_IBGE'].apply(normalizar_com_correcao)
df_2018_2025['NOME_KEY'] = df_2018_2025['NO_MUNICIPIO'].apply(normalizar_com_correcao)
df_2022['NOME_KEY']      = df_2022['NO_MUNICIPIO'].apply(normalizar_com_correcao)

colunas_dom = ['DOMICILIOS_TOTAL', 'DOM_SEM_CONJUGE_TOTAL', 'DOM_SEM_CONJUGE_HOMEM', 'DOM_SEM_CONJUGE_MULHER']

df_2018_2025 = df_2018_2025.drop(columns=[c for c in colunas_dom if c in df_2018_2025.columns])
df_2022      = df_2022.drop(columns=[c for c in colunas_dom if c in df_2022.columns])

df_2018_2025 = pd.merge(df_2018_2025,
                         df_dom[['NOME_KEY', 'SG_UF'] + colunas_dom],
                         on=['NOME_KEY', 'SG_UF'], how='left')

df_2022 = pd.merge(df_2022,
                   df_dom[['NOME_KEY', 'SG_UF'] + colunas_dom],
                   on=['NOME_KEY', 'SG_UF'], how='left')

# Propaga valores para todos os anos
df_2018_2025 = df_2018_2025.sort_values(['CO_MUNICIPIO', 'NU_ANO_CENSO'])
for col in colunas_dom:
    df_2018_2025[col] = df_2018_2025.groupby('CO_MUNICIPIO')[col].transform(
        lambda x: x.ffill().bfill()
    )

df_2018_2025 = df_2018_2025.drop(columns=['NOME_KEY'])
df_2022      = df_2022.drop(columns=['NOME_KEY'])

print(f'2018-2025: {df_2018_2025.shape} | Nulos: {df_2018_2025["DOM_SEM_CONJUGE_MULHER"].isna().sum()}')
print(f'2022:      {df_2022.shape}      | Nulos: {df_2022["DOM_SEM_CONJUGE_MULHER"].isna().sum()}')

df_2018_2025.to_csv('dataset_2018_2025.csv', sep=';', encoding='utf-8', index=False)
df_2022.to_csv('dataset_2022.csv', sep=';', encoding='utf-8', index=False)

print('\nDatasets atualizados com domicílios!')