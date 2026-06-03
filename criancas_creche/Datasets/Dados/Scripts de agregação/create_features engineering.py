import pandas as pd
import numpy as np

df_2022      = pd.read_csv('dataset_2022.csv', sep=';', encoding='utf-8')
df_2018_2025 = pd.read_csv('dataset_2018_2025.csv', sep=';', encoding='utf-8')

def criar_features(df):
    novo = pd.DataFrame()

    # Identificação
    novo['NU_ANO_CENSO'] = df['NU_ANO_CENSO'] if 'NU_ANO_CENSO' in df.columns else None
    novo['CO_MUNICIPIO'] = df['CO_MUNICIPIO']
    novo['NO_MUNICIPIO'] = df['NO_MUNICIPIO']
    novo['SG_UF']        = df['SG_UF']

    # 1. Target
    novo['TAXA_CRE_INT'] = (df['QT_MAT_INF_CRE_INT'] / df['QT_MAT_INF_CRE']).round(6)

    # 2. Estrutura escolar
    novo['PROP_ESC_PUBLICA'] = (df['QT_ESC_PUBLICA'] / df['QT_ESCOLAS']).round(6)
    novo['PROP_ESC_URBANA']  = (df['QT_ESC_URBANA']  / df['QT_ESCOLAS']).round(6)

    # 3. Composição racial — denominador corrigido
    TOTAL_COR = (df['QT_MAT_BAS_BRANCA'] + df['QT_MAT_BAS_PRETA'] +
                 df['QT_MAT_BAS_PARDA']  + df['QT_MAT_BAS_AMARELA'] +
                 df['QT_MAT_BAS_INDIGENA'])

    novo['PROP_MAT_BRANCA']   = (df['QT_MAT_BAS_BRANCA']   / TOTAL_COR).round(6)
    novo['PROP_MAT_PRETA']    = (df['QT_MAT_BAS_PRETA']    / TOTAL_COR).round(6)
    novo['PROP_MAT_PARDA']    = (df['QT_MAT_BAS_PARDA']    / TOTAL_COR).round(6)
    novo['PROP_MAT_INDIGENA'] = (df['QT_MAT_BAS_INDIGENA'] / TOTAL_COR).round(6)

    # 4. Cobertura
    novo['COBERTURA_CRECHE_0_3']          = (df['QT_MAT_BAS_0_3']  / df['POP_0_4']).round(6)
    novo['CRECHE_POR_1000_HAB']           = ((df['QT_MAT_INF_CRE'] / df['POP_TOTAL']) * 1000).round(6)
    novo['ESCOLAS_CRECHE_POR_1000_CRIAN'] = ((df['QT_ESCOLAS']     / df['POP_0_4'])   * 1000).round(6)
    novo['MEDIA_MAT_0_3_POR_ESCOLA']      = (df['QT_MAT_BAS_0_3']  / df['QT_ESCOLAS']).round(0).astype('Int64')

    # 5. População
    novo['PROP_POP_0_4']  = (df['POP_0_4'] / df['POP_TOTAL']).round(6)
    novo['LOG_POP_TOTAL'] = np.log1p(df['POP_TOTAL']).round(6)
    novo['LOG_POP_0_4']   = np.log1p(df['POP_0_4']).round(6)

    # 6. PIB
    novo['PIB_PC']     = (df['PIB'] / df['POP_TOTAL']).round(2)
    novo['LOG_PIB_PC'] = np.log1p(df['PIB'] / df['POP_TOTAL']).round(6)

    # 7. Ocupação
    novo['TAXA_OCUP_FEM']  = (df['MULHERES_OCUPADAS'] / df['TOTAL_MULHERES']).round(6)
    novo['TAXA_OCUP_MASC'] = (df['HOMENS_OCUPADOS']   / df['TOTAL_HOMENS']).round(6)

    # 8. Mães jovens
    novo['PROP_MAES_JOVENS'] = (df['MULHERES_FILHOS_12_19'] / df['MULHERES_FILHOS_TOTAL']).round(6)

    # 9. Composição familiar
    novo['PROP_DOM_SEM_CONJUGE']               = (df['DOM_SEM_CONJUGE_TOTAL']  / df['DOMICILIOS_TOTAL']).round(6)
    novo['PROP_DOM_SEM_CONJUGE_MULHER']        = (df['DOM_SEM_CONJUGE_MULHER'] / df['DOMICILIOS_TOTAL']).round(6)
    novo['PROP_DOM_SEM_CONJUGE_HOMEM']         = (df['DOM_SEM_CONJUGE_HOMEM']  / df['DOMICILIOS_TOTAL']).round(6)
    novo['PROP_MULHER_ENTRE_RESP_SEM_CONJUGE'] = (df['DOM_SEM_CONJUGE_MULHER'] / df['DOM_SEM_CONJUGE_TOTAL']).round(6)

    # 10. Composição da renda
    novo['PERC_RENDA_OUTRAS_FONTES']     = df['PERC_RENDA_OUTRAS_FONTES'].round(6)
    novo['RAZAO_OUTRAS_FONTES_TRABALHO'] = (df['PERC_RENDA_OUTRAS_FONTES'] / df['PERC_RENDA_TRABALHO']).round(6)

    # 11. Renda
    novo['RENDA_MEDIANA_RESPONSAVEL']     = df['RENDA_MEDIANA_RESPONSAVEL'].round(2)
    novo['LOG_RENDA_MEDIANA_RESPONSAVEL'] = np.log1p(df['RENDA_MEDIANA_RESPONSAVEL']).round(6)

    # 12. IDH
    novo['IDH_PROP_POBREZA']          = df['IDH_PROP_POBREZA'].round(6)
    novo['IDH_PROP_POBREZA_CRIANCAS'] = df['IDH_PROP_POBREZA_CRIANCAS'].round(6)
    novo['IDH_PROP_VULNER_POBREZA']   = df['IDH_PROP_VULNER_POBREZA'].round(6)
    novo['IDH_GINI']                  = df['IDH_GINI'].round(6)
    novo['IDH_EXPECT_ANOS_ESTUDO']    = df['IDH_EXPECT_ANOS_ESTUDO'].round(6)
    novo['IDH_RAZAO_DEPENDENCIA']     = df['IDH_RAZAO_DEPENDENCIA'].round(6)

    return novo

df_final_2022      = criar_features(df_2022)
df_final_2018_2025 = criar_features(df_2018_2025)

print(f'Final 2022:      {df_final_2022.shape}')
print(f'Final 2018-2025: {df_final_2018_2025.shape}')

# Verifica se proporções de cor ficaram entre 0 e 1
for col in ['PROP_MAT_BRANCA', 'PROP_MAT_PRETA', 'PROP_MAT_PARDA', 'PROP_MAT_INDIGENA']:
    acima = (df_final_2022[col] > 1).sum()
    print(f'{col}: {acima} municípios com valor > 1')

df_final_2022.to_csv('dataset_final_2022.csv', sep=';', encoding='utf-8', index=False)
df_final_2018_2025.to_csv('dataset_final_2018_2025.csv', sep=';', encoding='utf-8', index=False)

print('\nDatasets finais criados com sucesso!')