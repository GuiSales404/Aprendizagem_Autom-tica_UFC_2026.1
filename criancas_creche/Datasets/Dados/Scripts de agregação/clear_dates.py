import pandas as pd

# Colunas de interesse
COLUNAS_ESCOLA = [
    "NU_ANO_CENSO", "CO_ENTIDADE", "CO_MUNICIPIO", 
    "NO_MUNICIPIO", "SG_UF", "TP_DEPENDENCIA", "TP_LOCALIZACAO"
]

COLUNAS_MATRICULA = [
    "NU_ANO_CENSO", "CO_ENTIDADE", "QT_MAT_INF_CRE", 
    "QT_MAT_INF_CRE_INT", "QT_MAT_BAS_0_3", "QT_MAT_BAS_BRANCA", 
    "QT_MAT_BAS_PRETA", "QT_MAT_BAS_PARDA", "QT_MAT_BAS_AMARELA", 
    "QT_MAT_BAS_INDIGENA"
]

NORDESTE = ['AL', 'BA', 'CE', 'MA', 'PB', 'PE', 'PI', 'RN', 'SE']

# Lendo os arquivos — ajuste o caminho pro seu
escola = pd.read_csv('Tabela_Escola_2025.csv', sep=';', encoding='latin1', 
                     usecols=COLUNAS_ESCOLA, low_memory=False)

matricula = pd.read_csv('Tabela_Matricula_2025.csv', sep=';', encoding='latin1', 
                        usecols=COLUNAS_MATRICULA, low_memory=False)

# Filtrando só Nordeste na tabela escola
escola_nordeste = escola[escola['SG_UF'].isin(NORDESTE)]

# Juntando as duas tabelas pelo CO_ENTIDADE + NU_ANO_CENSO
df = pd.merge(escola_nordeste, matricula, 
              on=['CO_ENTIDADE', 'NU_ANO_CENSO'], 
              how='inner')

print(df.shape)
print(df.head())
df.to_csv('nordeste_2025.csv', sep=';', encoding='utf-8', index=False)