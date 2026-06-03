# Análise de Multicolinearidade (VIF) — Projeto ML
## Creche Integral e Vulnerabilidade Social no Nordeste Brasileiro

---

## Objetivo

Identificar e tratar multicolinearidade entre as features do dataset, complementando a análise de correlação realizada na EDA. O VIF (Variance Inflation Factor) quantifica o quanto a variância de um coeficiente é inflada pela correlação com as demais variáveis.

**Referência de corte:** VIF > 10 indica multicolinearidade relevante. VIF > 100 indica multicolinearidade severa.

---

## Dataset Analisado

`dataset_final_2022_clean.csv` — 1.781 municípios × 33 features (após remoção de nulos)

---

## Resultado Completo do VIF

| Variável | VIF | Situação |
|----------|-----|----------|
| `PROP_DOM_SEM_CONJUGE` | 1.22e+11 | ❌ Removida |
| `PROP_DOM_SEM_CONJUGE_MULHER` | 9.21e+10 | ✅ Mantida |
| `PROP_DOM_SEM_CONJUGE_HOMEM` | 2.16e+09 | ❌ Removida |
| `LOG_POP_TOTAL` | 3.42e+05 | ✅ Mantida |
| `LOG_POP_0_4` | 1.79e+05 | ❌ Removida |
| `LOG_RENDA_MEDIANA_RESPONSAVEL` | 9.86e+04 | ❌ Removida |
| `PROP_MULHER_ENTRE_RESP_SEM_CONJUGE` | 3.16e+04 | ❌ Removida |
| `PROP_POP_0_4` | 3.99e+03 | ❌ Removida |
| `PROP_MAT_PARDA` | 2.57e+03 | ✅ Mantida |
| `RENDA_MEDIANA_RESPONSAVEL` | 2.44e+03 | ✅ Mantida |
| `LOG_PIB_PC` | 1.44e+03 | ✅ Mantida |
| `IDH_PROP_VULNER_POBREZA` | 7.78e+02 | ❌ Removida |
| `IDH_PROP_POBREZA` | 6.86e+02 | ❌ Removida |
| `IDH_PROP_POBREZA_CRIANCAS` | 6.20e+02 | ✅ Mantida |
| `PERC_RENDA_OUTRAS_FONTES` | 5.96e+02 | ✅ Mantida |
| `IDH_RAZAO_DEPENDENCIA` | 2.84e+02 | ✅ Mantida |
| `PROP_MAT_BRANCA` | 2.66e+02 | ✅ Mantida |
| `IDH_GINI` | 2.38e+02 | ✅ Mantida |
| `PROP_ESC_PUBLICA` | 2.22e+02 | ✅ Mantida |
| `IDH_EXPECT_ANOS_ESTUDO` | 1.59e+02 | ✅ Mantida |
| `TAXA_OCUP_MASC` | 1.25e+02 | ✅ Mantida |
| `RAZAO_OUTRAS_FONTES_TRABALHO` | 1.20e+02 | ❌ Removida |
| `COBERTURA_CRECHE_0_3` | 9.67e+01 | ✅ Mantida |
| `CRECHE_POR_1000_HAB` | 9.12e+01 | ❌ Removida |
| `TAXA_OCUP_FEM` | 8.01e+01 | ✅ Mantida |
| `PROP_MAT_PRETA` | 2.71e+01 | ✅ Mantida |
| `PROP_ESC_URBANA` | 1.82e+01 | ✅ Mantida |
| `MEDIA_MAT_0_3_POR_ESCOLA` | 1.73e+01 | ✅ Mantida |
| `PROP_MAT_INDIGENA` | 1.53e+01 | ✅ Mantida |
| `ESCOLAS_CRECHE_POR_1000_CRIAN` | 1.49e+01 | ✅ Mantida |
| `PROP_MAES_JOVENS` | 1.07e+01 | ✅ Mantida |
| `PIB_PC` | 4.45e+00 | ✅ Mantida |

---

## Variáveis Removidas — Justificativas

| Variável | VIF | Grupo | Justificativa |
|----------|-----|-------|---------------|
| `PROP_DOM_SEM_CONJUGE` | 1.22e+11 | Domicílios | VIF praticamente infinito — combinação linear quase perfeita das demais variáveis de domicílio. Mantida `PROP_DOM_SEM_CONJUGE_MULHER` por maior relevância conceitual |
| `PROP_DOM_SEM_CONJUGE_HOMEM` | 2.16e+09 | Domicílios | VIF extremamente alto — redundante dentro do grupo de domicílios sem cônjuge |
| `PROP_MULHER_ENTRE_RESP_SEM_CONJUGE` | 3.16e+04 | Domicílios | Alta colinearidade com `PROP_DOM_SEM_CONJUGE_MULHER` |
| `LOG_POP_0_4` | 1.79e+05 | População | Quase idêntica a `LOG_POP_TOTAL` (correlação 0.99) — população total é mais geral |
| `PROP_POP_0_4` | 3.99e+03 | População | Redundante com `LOG_POP_TOTAL` e features de cobertura |
| `LOG_RENDA_MEDIANA_RESPONSAVEL` | 9.86e+04 | Renda | VIF muito superior à variável original — manter `RENDA_MEDIANA_RESPONSAVEL` mais interpretável |
| `IDH_PROP_POBREZA` | 6.86e+02 | IDH Pobreza | Correlação >0.96 com `IDH_PROP_POBREZA_CRIANCAS` — todas medem vulnerabilidade socioeconômica |
| `IDH_PROP_VULNER_POBREZA` | 7.78e+02 | IDH Pobreza | Correlação >0.92 com `IDH_PROP_POBREZA_CRIANCAS` — mantida a variável mais alinhada ao tema da creche |
| `RAZAO_OUTRAS_FONTES_TRABALHO` | 1.20e+02 | Renda | Correlação 0.96 com `PERC_RENDA_OUTRAS_FONTES` — manter a mais interpretável |
| `CRECHE_POR_1000_HAB` | 9.12e+01 | Cobertura | Correlação 0.91 com `COBERTURA_CRECHE_0_3` — manter a diretamente ligada ao problema |

---

## Dataset Resultante

| Dataset | Linhas | Colunas |
|---------|--------|---------|
| Entrada (`dataset_final_2022_clean.csv`) | 1.781 | 37 |
| Saída (`dataset_modelo_2022.csv`) | 1.781 | 27 |

**Features removidas:** 10  
**Features mantidas:** 23 features + 4 colunas de identificação

---

## Observação sobre Modelos de Árvore

Embora variáveis com VIF elevado ainda permaneçam no dataset (ex: `PROP_MAT_PARDA` com 2.6e+03, `LOG_PIB_PC` com 1.4e+03), optou-se por não remover todas as variáveis com VIF acima do limiar padrão. Os modelos baseados em árvores (Random Forest, XGBoost) são inerentemente robustos à multicolinearidade, pois selecionam features por ganho de informação e não por coeficientes lineares. O refinamento final será realizado com base na importância das variáveis via análise SHAP.

---

## Próximos Passos

- [x] EDA concluída
- [x] VIF concluído
- [ ] **Treinamento dos modelos** — Linear Regression (baseline), Random Forest Regressor, XGBoost Regressor
- [ ] **Análise SHAP** — interpretação e refinamento final do conjunto de features

---

*Última atualização: Junho 2026*
