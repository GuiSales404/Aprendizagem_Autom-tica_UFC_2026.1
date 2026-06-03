# Feature Engineering — Projeto ML
## Creche Integral e Vulnerabilidade Social no Nordeste Brasileiro

---

## Datasets de Entrada

| Dataset | Linhas | Colunas | Descrição |
|---------|--------|---------|-----------|
| `dataset_2022_raw.csv` | 1.794 | 45 | Corte transversal — dados brutos agregados por município |
| `dataset_2018_2025_raw.csv` | 14.352 | 45 | Série temporal — dados brutos agregados por município-ano |

---

## Datasets de Saída

| Dataset | Linhas | Colunas | Descrição |
|---------|--------|---------|-----------|
| `dataset_final_2022.csv` | 1.794 | 37 | Corte transversal — features derivadas |
| `dataset_final_2018_2025.csv` | 14.352 | 37 | Série temporal — features derivadas |

---

## Convenções de Arredondamento

| Tipo de variável | Casas decimais | Exemplo |
|-----------------|----------------|---------|
| Proporções e taxas | 6 casas | `0.823451` |
| Variáveis monetárias | 2 casas | `1842.50` |
| Contagens | Sem decimais (inteiro) | `42` |

---

## Variáveis de Identificação

Mantidas diretamente dos datasets base sem transformação.

| Variável | Descrição |
|----------|-----------|
| `NU_ANO_CENSO` | Ano do censo escolar |
| `CO_MUNICIPIO` | Código IBGE do município |
| `NO_MUNICIPIO` | Nome do município |
| `SG_UF` | Sigla do estado |

---

## 1. Variável Alvo

| Variável | Fórmula | Tipo |
|----------|---------|------|
| `TAXA_CRE_INT` | `QT_MAT_INF_CRE_INT / QT_MAT_INF_CRE` | Proporção |

> Proporção de matrículas em creche de tempo integral sobre o total de matrículas em creche. Utilizada como indicador da utilização observada da educação infantil em tempo integral nos municípios analisados.

---

## 2. Estrutura Escolar

| Variável | Fórmula | Tipo |
|----------|---------|------|
| `PROP_ESC_PUBLICA` | `QT_ESC_PUBLICA / QT_ESCOLAS` | Proporção |
| `PROP_ESC_URBANA` | `QT_ESC_URBANA / QT_ESCOLAS` | Proporção |

**Variáveis excluídas:**

| Variável | Motivo |
|----------|--------|
| `PROP_ESC_PRIVADA` | Multicolinearidade perfeita com `PROP_ESC_PUBLICA` — soma sempre 1 |
| `PROP_ESC_RURAL` | Multicolinearidade perfeita com `PROP_ESC_URBANA` — soma sempre 1 |

---

## 3. Composição Racial das Matrículas (0–3 anos)

| Variável | Fórmula | Tipo |
|----------|---------|------|
| `PROP_MAT_BRANCA` | `QT_MAT_BAS_BRANCA / QT_MAT_BAS_0_3` | Proporção |
| `PROP_MAT_PRETA` | `QT_MAT_BAS_PRETA / QT_MAT_BAS_0_3` | Proporção |
| `PROP_MAT_PARDA` | `QT_MAT_BAS_PARDA / QT_MAT_BAS_0_3` | Proporção |
| `PROP_MAT_INDIGENA` | `QT_MAT_BAS_INDIGENA / QT_MAT_BAS_0_3` | Proporção |

**Variável excluída:**

| Variável | Motivo |
|----------|--------|
| `PROP_MAT_AMARELA` | Frequência muito baixa no Nordeste — altamente correlacionada com ruído estatístico |

---

## 4. Cobertura e Intensidade da Educação Infantil

| Variável | Fórmula | Tipo |
|----------|---------|------|
| `COBERTURA_CRECHE_0_3` | `QT_MAT_BAS_0_3 / POP_0_4` | Proporção |
| `CRECHE_POR_1000_HAB` | `(QT_MAT_INF_CRE / POP_TOTAL) * 1000` | Proporção |
| `ESCOLAS_CRECHE_POR_1000_CRIAN` | `(QT_ESCOLAS / POP_0_4) * 1000` | Proporção |
| `MEDIA_MAT_0_3_POR_ESCOLA` | `QT_MAT_BAS_0_3 / QT_ESCOLAS` | Contagem |

---

## 5. População

| Variável | Fórmula | Tipo |
|----------|---------|------|
| `PROP_POP_0_4` | `POP_0_4 / POP_TOTAL` | Proporção |
| `LOG_POP_TOTAL` | `log1p(POP_TOTAL)` | Proporção |
| `LOG_POP_0_4` | `log1p(POP_0_4)` | Proporção |

> Transformação logarítmica aplicada para reduzir assimetria em variáveis de escala populacionais.

---

## 6. PIB

| Variável | Fórmula | Tipo |
|----------|---------|------|
| `PIB_PC` | `PIB / POP_TOTAL` | Monetário |
| `LOG_PIB_PC` | `log1p(PIB / POP_TOTAL)` | Proporção |

**Variável excluída:**

| Variável | Motivo |
|----------|--------|
| `LOG_PIB` | Altamente correlacionada com `LOG_PIB_PC` e `LOG_POP_TOTAL` — redundância |

---

## 7. Ocupação por Sexo

| Variável | Fórmula | Tipo |
|----------|---------|------|
| `TAXA_OCUP_FEM` | `MULHERES_OCUPADAS / TOTAL_MULHERES` | Proporção |
| `TAXA_OCUP_MASC` | `HOMENS_OCUPADOS / TOTAL_HOMENS` | Proporção |

**Variáveis excluídas:**

| Variável | Motivo |
|----------|--------|
| `DIF_OCUP_MASC_FEM` | Derivada de `TAXA_OCUP_FEM` e `TAXA_OCUP_MASC` — redundância |
| `RAZAO_OCUP_FEM_MASC` | Derivada de `TAXA_OCUP_FEM` e `TAXA_OCUP_MASC` — redundância |

> `TAXA_OCUP_FEM` é variável de interesse central para a hipótese do trabalho — municípios com maior ocupação feminina tendem a apresentar maior demanda por creche integral.

---

## 8. Mães Jovens

| Variável | Fórmula | Tipo |
|----------|---------|------|
| `PROP_MAES_JOVENS` | `MULHERES_FILHOS_12_19 / MULHERES_FILHOS_TOTAL` | Proporção |

---

## 9. Composição Familiar

| Variável | Fórmula | Tipo |
|----------|---------|------|
| `PROP_DOM_SEM_CONJUGE` | `DOM_SEM_CONJUGE_TOTAL / DOMICILIOS_TOTAL` | Proporção |
| `PROP_DOM_SEM_CONJUGE_MULHER` | `DOM_SEM_CONJUGE_MULHER / DOMICILIOS_TOTAL` | Proporção |
| `PROP_DOM_SEM_CONJUGE_HOMEM` | `DOM_SEM_CONJUGE_HOMEM / DOMICILIOS_TOTAL` | Proporção |
| `PROP_MULHER_ENTRE_RESP_SEM_CONJUGE` | `DOM_SEM_CONJUGE_MULHER / DOM_SEM_CONJUGE_TOTAL` | Proporção |

---

## 10. Composição do Rendimento Domiciliar

| Variável | Origem / Fórmula | Tipo |
|----------|-----------------|------|
| `PERC_RENDA_OUTRAS_FONTES` | Direto do dataset base (%) | Proporção |
| `RAZAO_OUTRAS_FONTES_TRABALHO` | `PERC_RENDA_OUTRAS_FONTES / PERC_RENDA_TRABALHO` | Proporção |

**Variável excluída:**

| Variável | Motivo |
|----------|--------|
| `PERC_RENDA_TRABALHO` | Multicolinearidade perfeita com `PERC_RENDA_OUTRAS_FONTES` — soma sempre 100% |

---

## 11. Renda dos Responsáveis

| Variável | Origem / Fórmula | Tipo |
|----------|-----------------|------|
| `RENDA_MEDIANA_RESPONSAVEL` | Direto do dataset base | Monetário |
| `LOG_RENDA_MEDIANA_RESPONSAVEL` | `log1p(RENDA_MEDIANA_RESPONSAVEL)` | Proporção |

**Variáveis excluídas:**

| Variável | Motivo |
|----------|--------|
| `RENDA_MEDIA_RESPONSAVEL` | A mediana é mais robusta a outliers em análises de vulnerabilidade — média descartada |
| `LOG_RENDA_MEDIA_RESPONSAVEL` | Derivada da média descartada |

---

## 12. Indicadores IDH / Atlas

| Variável | Origem | Tipo |
|----------|--------|------|
| `IDH_PROP_POBREZA` | Direto do dataset base | Proporção |
| `IDH_PROP_POBREZA_CRIANCAS` | Direto do dataset base | Proporção |
| `IDH_PROP_VULNER_POBREZA` | Direto do dataset base | Proporção |
| `IDH_GINI` | Direto do dataset base | Proporção |
| `IDH_EXPECT_ANOS_ESTUDO` | Direto do dataset base | Proporção |
| `IDH_RAZAO_DEPENDENCIA` | Direto do dataset base | Proporção |

**Variáveis excluídas:**

| Variável | Motivo |
|----------|--------|
| `IDH_TAXA_FREQ_0_3` | Risco de leakage conceitual com a variável alvo `TAXA_CRE_INT` |
| `IDH_RENDA_PC` | Altamente correlacionada com `RENDA_MEDIANA_RESPONSAVEL` e `PIB_PC` — redundância |

---

## Resumo Geral das Exclusões

| Variável | Grupo | Motivo |
|----------|-------|--------|
| `PROP_ESC_PRIVADA` | Estrutura escolar | Multicolinearidade perfeita com `PROP_ESC_PUBLICA` |
| `PROP_ESC_RURAL` | Estrutura escolar | Multicolinearidade perfeita com `PROP_ESC_URBANA` |
| `PROP_MAT_AMARELA` | Composição racial | Frequência muito baixa — correlacionada com ruído |
| `LOG_PIB` | PIB | Redundante com `LOG_PIB_PC` e `LOG_POP_TOTAL` |
| `DIF_OCUP_MASC_FEM` | Ocupação | Derivada de `TAXA_OCUP_FEM` e `TAXA_OCUP_MASC` |
| `RAZAO_OCUP_FEM_MASC` | Ocupação | Derivada de `TAXA_OCUP_FEM` e `TAXA_OCUP_MASC` |
| `PERC_RENDA_TRABALHO` | Composição de renda | Multicolinearidade perfeita com `PERC_RENDA_OUTRAS_FONTES` |
| `RENDA_MEDIA_RESPONSAVEL` | Renda | Mediana mais robusta — média descartada |
| `LOG_RENDA_MEDIA_RESPONSAVEL` | Renda | Derivada da média descartada |
| `IDH_RENDA_PC` | IDH | Altamente correlacionada com `RENDA_MEDIANA_RESPONSAVEL` e `PIB_PC` |
| `IDH_TAXA_FREQ_0_3` | IDH | Risco de leakage conceitual com a variável alvo |

---

## Estrutura Final dos Datasets

```
4  colunas de identificação
1  variável alvo
32 features candidatas
──────────────
37 colunas totais
```

---

## Próximos Passos

- [ ] **EDA** — distribuições, outliers, valores nulos e infinitos gerados por divisões
- [ ] **Correlação + VIF** — identificar redundâncias e multicolinearidade entre features derivadas; decidir com dados quais variáveis ficam no modelo final
- [ ] **Treinamento dos modelos** — Linear Regression (baseline), Random Forest Regressor, XGBoost Regressor
- [ ] **Análise SHAP** — interpretação dos fatores associados à variável alvo

---

*Última atualização: Maio 2026*
