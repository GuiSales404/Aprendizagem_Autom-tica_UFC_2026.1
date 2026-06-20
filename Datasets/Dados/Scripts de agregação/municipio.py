import pandas as pd

df_total = pd.read_csv('nordeste_2018_2025.csv', sep=';', encoding='utf-8', low_memory=False)

# Criando colunas binárias para dependência
df_total['ESC_FEDERAL']   = (df_total['TP_DEPENDENCIA'] == 1).astype(int)
df_total['ESC_ESTADUAL']  = (df_total['TP_DEPENDENCIA'] == 2).astype(int)
df_total['ESC_MUNICIPAL'] = (df_total['TP_DEPENDENCIA'] == 3).astype(int)
df_total['ESC_PRIVADA']   = (df_total['TP_DEPENDENCIA'] == 4).astype(int)
df_total['ESC_PUBLICA']   = (df_total['TP_DEPENDENCIA'].isin([1, 2, 3])).astype(int)

# Criando colunas binárias para localização
df_total['ESC_URBANA'] = (df_total['TP_LOCALIZACAO'] == 1).astype(int)
df_total['ESC_RURAL']  = (df_total['TP_LOCALIZACAO'] == 2).astype(int)

# Agrega por município-ano somando tudo
df_municipio = df_total.groupby(
    ['NU_ANO_CENSO', 'CO_MUNICIPIO', 'NO_MUNICIPIO', 'SG_UF']
).agg(
    QT_MAT_INF_CRE      = ('QT_MAT_INF_CRE', 'sum'),
    QT_MAT_INF_CRE_INT  = ('QT_MAT_INF_CRE_INT', 'sum'),
    QT_MAT_BAS_0_3      = ('QT_MAT_BAS_0_3', 'sum'),
    QT_MAT_BAS_BRANCA   = ('QT_MAT_BAS_BRANCA', 'sum'),
    QT_MAT_BAS_PRETA    = ('QT_MAT_BAS_PRETA', 'sum'),
    QT_MAT_BAS_PARDA    = ('QT_MAT_BAS_PARDA', 'sum'),
    QT_MAT_BAS_AMARELA  = ('QT_MAT_BAS_AMARELA', 'sum'),
    QT_MAT_BAS_INDIGENA = ('QT_MAT_BAS_INDIGENA', 'sum'),
    QT_ESCOLAS          = ('CO_ENTIDADE', 'count'),
    QT_ESC_FEDERAL      = ('ESC_FEDERAL', 'sum'),
    QT_ESC_ESTADUAL     = ('ESC_ESTADUAL', 'sum'),
    QT_ESC_MUNICIPAL    = ('ESC_MUNICIPAL', 'sum'),
    QT_ESC_PRIVADA      = ('ESC_PRIVADA', 'sum'),
    QT_ESC_PUBLICA      = ('ESC_PUBLICA', 'sum'),
    QT_ESC_URBANA       = ('ESC_URBANA', 'sum'),
    QT_ESC_RURAL        = ('ESC_RURAL', 'sum')
).reset_index()

print(df_municipio.shape)
print(df_municipio.head())

df_municipio.to_csv('municipio_2018_2025.csv', sep=';', encoding='utf-8', index=False)