import pandas as pd

NORDESTE_CODIGOS = ['21', '22', '23', '24', '25', '26', '27', '28', '29']

COLUNAS = [
    'id_municipio',
    'prop_pobreza',
    'prop_pobreza_criancas',
    'prop_vulner_pobreza',
    'indice_gini',
    'renda_pc',
    'expectativa_anos_estudo',
    'taxa_freq_0_3',
    'razao_dependencia'
]

df_idh = pd.read_csv('mundo_onu_adh_municipio.csv', usecols=['ano'] + COLUNAS)

# Filtra 2010 e Nordeste
df_idh = df_idh[df_idh['ano'] == 2010].copy()
df_idh['id_municipio'] = df_idh['id_municipio'].astype(str)
df_idh = df_idh[df_idh['id_municipio'].str[:2].isin(NORDESTE_CODIGOS)].copy()
df_idh['id_municipio'] = df_idh['id_municipio'].astype(int)

# Renomeia
df_idh = df_idh.rename(columns={
    'id_municipio': 'CO_MUNICIPIO',
    'prop_pobreza': 'IDH_PROP_POBREZA',
    'prop_pobreza_criancas': 'IDH_PROP_POBREZA_CRIANCAS',
    'prop_vulner_pobreza': 'IDH_PROP_VULNER_POBREZA',
    'indice_gini': 'IDH_GINI',
    'renda_pc': 'IDH_RENDA_PC',
    'expectativa_anos_estudo': 'IDH_EXPECT_ANOS_ESTUDO',
    'taxa_freq_0_3': 'IDH_TAXA_FREQ_0_3',
    'razao_dependencia': 'IDH_RAZAO_DEPENDENCIA'
})[['CO_MUNICIPIO', 'IDH_PROP_POBREZA', 'IDH_PROP_POBREZA_CRIANCAS',
    'IDH_PROP_VULNER_POBREZA', 'IDH_GINI', 'IDH_RENDA_PC',
    'IDH_EXPECT_ANOS_ESTUDO', 'IDH_TAXA_FREQ_0_3', 'IDH_RAZAO_DEPENDENCIA']]

print(df_idh.shape)
print(df_idh.head())

df_idh.to_csv('idh_nordeste_2010.csv', sep=';', encoding='utf-8', index=False)