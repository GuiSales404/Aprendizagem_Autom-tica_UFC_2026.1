import pandas as pd

NORDESTE = ['AL', 'BA', 'CE', 'MA', 'PB', 'PE', 'PI', 'RN', 'SE']

df_renda = pd.read_csv(
    'Agregados_por_municipios_renda_responsavel_BR.csv',
    sep=';', encoding='latin1'
)

# Extrai a UF do código do município (primeiros 2 dígitos)
# e filtra só Nordeste
df_renda['CO_MUNICIPIO'] = df_renda['CD_MUN'].astype(str).str[:7].astype(int)
df_renda['CO_UF'] = df_renda['CD_MUN'].astype(str).str[:2].astype(int)

CODIGOS_NORDESTE = [21, 22, 23, 24, 25, 26, 27, 28, 29]  # MA,PI,CE,RN,PB,PE,AL,SE,BA

df_renda = df_renda[df_renda['CO_UF'].isin(CODIGOS_NORDESTE)].copy()

# Renomeia e seleciona colunas
df_renda = df_renda.rename(columns={
    'V06004': 'RENDA_MEDIA_RESPONSAVEL',
    'V06006': 'RENDA_MEDIANA_RESPONSAVEL'
})[['CO_MUNICIPIO', 'RENDA_MEDIA_RESPONSAVEL', 'RENDA_MEDIANA_RESPONSAVEL']]

# Corrige V06004 que veio com vírgula decimal
df_renda['RENDA_MEDIA_RESPONSAVEL'] = (
    df_renda['RENDA_MEDIA_RESPONSAVEL']
    .astype(str).str.replace(',', '.').astype(float)
)

print(df_renda.shape)
print(df_renda.head())

df_renda.to_csv('renda_nordeste_2022.csv', sep=';', encoding='utf-8', index=False)