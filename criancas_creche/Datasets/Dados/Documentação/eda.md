# Análise Exploratória de Dados (EDA) — Projeto ML
## Creche Integral e Vulnerabilidade Social no Nordeste Brasileiro

---

## Objetivo

Verificar a qualidade e estrutura dos datasets finais nos seguintes aspectos:

- Distribuição das variáveis
- Outliers e valores extremos
- Valores nulos e divisões inválidas
- Variáveis constantes ou de baixa granularidade
- Colinearidade e redundância entre features

---

## Datasets Analisados

| Dataset | Linhas | Colunas |
|---------|--------|---------|
| `dataset_final_2022.csv` | 1.794 | 37 |
| `dataset_final_2018_2025.csv` | 14.352 | 37 |

---

## 1. Valores Nulos

### Dataset 2022

| Variável | Nulos | Observação |
|----------|-------|-----------|
| `TAXA_CRE_INT` | 12 | Municípios sem nenhuma matrícula em creche — denominador zero |
| `COBERTURA_CRECHE_0_3` | 1 | Município com `POP_0_4 = 0` |
| `ESCOLAS_CRECHE_POR_1000_CRIAN` | 1 | Município com `POP_0_4 = 0` |
| `PROP_POP_0_4` | 1 | Município com `POP_0_4 = 0` |
| `LOG_POP_0_4` | 1 | Município com `POP_0_4 = 0` |

**Tratamento aplicado:** Remoção das linhas com nulos. Os 12 municípios com `TAXA_CRE_INT` nula representam 0,67% da base — percentual reduzido que não compromete a representatividade da amostra. O município com `POP_0_4 = 0` foi removido junto.

**Dataset após limpeza:** `dataset_final_2022_clean.csv`

---

## 2. Divisões Inválidas

Nenhum valor infinito foi encontrado em qualquer coluna numérica dos dois datasets. As divisões realizadas no feature engineering não geraram inconsistências.

---

## 3. Variáveis Constantes e Baixa Granularidade

| Variável | Valores únicos | Observação |
|----------|---------------|-----------|
| `IDH_GINI` | 37 | Baixa granularidade — quase categórica. Feature de baixo poder discriminativo individual |

Nenhuma variável completamente constante foi encontrada.

---

## 4. Distribuição da Variável Alvo

A variável alvo `TAXA_CRE_INT` apresentou distribuição fortemente assimétrica:

| Faixa | Municípios | Proporção |
|-------|-----------|-----------|
| `TAXA_CRE_INT = 0` | 590 | 32,9% |
| `TAXA_CRE_INT = 1` | 83 | 4,6% |
| Valores intermediários | 1.121 | 62,5% |

Apesar da assimetria, a presença de 62,5% dos municípios com valores intermediários indica variabilidade suficiente para o treinamento dos modelos preditivos. A assimetria será considerada na escolha das métricas de avaliação.

---

## 5. Análise dos Extremos

**Municípios com `TAXA_CRE_INT = 1` (100% integral):**
Apresentaram perfis distintos quanto à renda, ocupação feminina e cobertura de creches. Alguns com `TAXA_OCUP_FEM` muito baixa (0.13–0.15), sugerindo que em municípios muito pequenos e pobres a integralidade da creche pode ser estrutural — independente da inserção formal da mãe no mercado de trabalho.

**Municípios com `TAXA_CRE_INT = 0` (sem creche integral):**
Apresentaram predominantemente alta proporção de escolas públicas, baixa urbanização e altos índices de pobreza pelo IDH — padrão coerente com municípios rurais sem oferta de período integral.

A análise exploratória sugere que a oferta de matrículas em tempo integral não parece ser explicada por um único fator socioeconômico ou estrutural. Municípios com taxa máxima de atendimento integral apresentaram perfis distintos quanto à renda, ocupação feminina e cobertura de creches, indicando a provável influência conjunta de múltiplas características locais.

---

## 6. Correlação com a Variável Alvo

Correlações de Pearson entre cada feature e `TAXA_CRE_INT` (dataset 2022):

| Variável | Correlação | Direção |
|----------|-----------|---------|
| `PROP_MAT_PRETA` | 0.227 | ↑ positiva |
| `TAXA_OCUP_MASC` | 0.150 | ↑ positiva |
| `TAXA_OCUP_FEM` | 0.130 | ↑ positiva |
| `IDH_EXPECT_ANOS_ESTUDO` | -0.146 | ↓ negativa |
| `IDH_PROP_POBREZA` | -0.141 | ↓ negativa |
| `IDH_RAZAO_DEPENDENCIA` | -0.134 | ↓ negativa |
| `IDH_PROP_VULNER_POBREZA` | -0.127 | ↓ negativa |
| `RENDA_MEDIANA_RESPONSAVEL` | 0.107 | ↑ positiva |
| `PROP_ESC_URBANA` | 0.103 | ↑ positiva |
| `COBERTURA_CRECHE_0_3` | 0.109 | ↑ positiva |

> As correlações individuais são modestas, o que é esperado em dados socioeconômicos municipais com fenômenos multidimensionais e relações não lineares. Os modelos baseados em árvores (Random Forest, XGBoost) são capazes de capturar combinações de variáveis que a correlação linear não detecta.

---

## 7. Colinearidade entre Features

Foram identificados grupos de atributos altamente correlacionados entre si. Optou-se por **mapear as sugestões de remoção sem aplicá-las agora**, pois os modelos baseados em árvores são menos sensíveis à multicolinearidade. A importância das variáveis (SHAP) será utilizada para refinar o conjunto de atributos posteriormente.

### Grupos identificados

**Grupo 1 — Escala populacional**

| Par | Correlação | Sugestão |
|-----|-----------|----------|
| `LOG_POP_TOTAL` ↔ `LOG_POP_0_4` | 0.99 | Remover `LOG_POP_0_4` — `LOG_POP_TOTAL` é mais geral |

**Grupo 2 — Transformação de renda**

| Par | Correlação | Sugestão |
|-----|-----------|----------|
| `RENDA_MEDIANA_RESPONSAVEL` ↔ `LOG_RENDA_MEDIANA_RESPONSAVEL` | 0.98 | Remover `LOG_RENDA_MEDIANA_RESPONSAVEL` — manter a variável original |

**Grupo 3 — Domicílios sem cônjuge**

| Par | Correlação | Sugestão |
|-----|-----------|----------|
| `PROP_DOM_SEM_CONJUGE` ↔ `PROP_DOM_SEM_CONJUGE_MULHER` | 0.98 | Remover `PROP_DOM_SEM_CONJUGE` — `PROP_DOM_SEM_CONJUGE_MULHER` conversa mais com a literatura sobre demanda por creche |

**Grupo 4 — Índices de pobreza IDH**

| Par | Correlação | Sugestão |
|-----|-----------|----------|
| `IDH_PROP_POBREZA` ↔ `IDH_PROP_POBREZA_CRIANCAS` | 0.97 | Remover `IDH_PROP_POBREZA` e `IDH_PROP_VULNER_POBREZA` |
| `IDH_PROP_POBREZA` ↔ `IDH_PROP_VULNER_POBREZA` | 0.95 | Manter `IDH_PROP_POBREZA_CRIANCAS` — conversa diretamente com o tema da creche |
| `IDH_PROP_POBREZA_CRIANCAS` ↔ `IDH_PROP_VULNER_POBREZA` | 0.92 | |

**Grupo 5 — Composição de renda**

| Par | Correlação | Sugestão |
|-----|-----------|----------|
| `PERC_RENDA_OUTRAS_FONTES` ↔ `RAZAO_OUTRAS_FONTES_TRABALHO` | 0.96 | Remover `RAZAO_OUTRAS_FONTES_TRABALHO` — `PERC_RENDA_OUTRAS_FONTES` é mais interpretável |

**Grupo 6 — Cobertura de creche**

| Par | Correlação | Sugestão |
|-----|-----------|----------|
| `COBERTURA_CRECHE_0_3` ↔ `CRECHE_POR_1000_HAB` | 0.91 | Remover `CRECHE_POR_1000_HAB` — `COBERTURA_CRECHE_0_3` é mais diretamente ligada ao problema |

**Grupo 7 — Ocupação por sexo** *(não remover)*

| Par | Correlação | Observação |
|-----|-----------|-----------|
| `TAXA_OCUP_FEM` ↔ `TAXA_OCUP_MASC` | 0.84 | Manter ambas — são conceitos distintos e cada uma pode ter importância independente no modelo |

---

## 8. Conclusão da EDA

- ✅ Nenhum valor infinito encontrado
- ✅ Poucos valores ausentes (<1%) — removidos sem impacto na representatividade
- ✅ Nenhuma variável completamente constante
- ✅ Variável alvo com ampla variabilidade apesar da assimetria
- ⚠️ `IDH_GINI` com baixa granularidade — avaliar contribuição no modelo
- ⚠️ Grupos de atributos altamente correlacionados mapeados — decisão de remoção adiada para após análise de importância (SHAP)

---

## Próximos Passos

- [x] EDA concluída
- [ ] **VIF** — análise formal de multicolinearidade para confirmar sugestões de remoção
- [ ] **Treinamento dos modelos** — Linear Regression (baseline), Random Forest Regressor, XGBoost Regressor
- [ ] **Análise SHAP** — interpretação e refinamento final do conjunto de features

---

*Última atualização: Maio 2026*
