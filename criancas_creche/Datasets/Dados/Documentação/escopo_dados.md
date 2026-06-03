# Documentação de Dados — Projeto ML
## Creche Integral e Vulnerabilidade Social no Nordeste Brasileiro

## Objetivo

> Dadas características socioeconômicas e estruturais de um município, prever a taxa de crianças de 0–3 anos em creche integral e investigar quais fatores apresentam maior poder explicativo sobre esse fenômeno.

---

## Pergunta de Pesquisa

> Em que medida fatores socioeconômicos e estruturais estão associados à presença de crianças de 0–3 anos em creche integral nos municípios do Nordeste brasileiro?

---

## Datasets Gerados

### 1. `dataset_2022.csv` — Corte transversal
- **Unidade de análise:** Município (1.794 municípios do Nordeste)
- **Uso previsto:** Modelo principal — Regressão supervisionada: Linear Regression (baseline), Random Forest Regressor e XGBoost Regressor
- **Linhas:** 1.794 × 45 colunas

### 2. `dataset_2018_2025.csv` — Série temporal
- **Unidade de análise:** Município-ano
- **Uso previsto:** Análise de tendência temporal, clustering evolutivo
- **Linhas:** 14.352 × 45 colunas

---

## Fontes e Variáveis

### Censo Escolar — INEP
**Fonte:** https://www.gov.br/inep/pt-br/acesso-a-informacao/dados-abertos/microdados/censo-escolar  
**Cobertura temporal:** 2018–2025  
**Disponibilidade:** Todos os anos do intervalo ✅

| Variável | Descrição | Disponibilidade |
|----------|-----------|-----------------|
| `NU_ANO_CENSO` | Ano do censo escolar | 2018–2025 ✅ |
| `CO_MUNICIPIO` | Código IBGE do município | 2018–2025 ✅ |
| `NO_MUNICIPIO` | Nome do município | 2018–2025 ✅ |
| `SG_UF` | Sigla do estado | 2018–2025 ✅ |
| `QT_MAT_INF_CRE` | Total de matrículas na creche | 2018–2025 ✅ |
| `QT_MAT_INF_CRE_INT` | Matrículas em creche de tempo integral | 2018–2025 ✅ |
| `QT_MAT_BAS_0_3` | Matrículas de crianças de 0 a 3 anos | 2018–2025 ✅ |
| `QT_MAT_BAS_BRANCA` | Matrículas declaradas brancas | 2018–2025 ✅ |
| `QT_MAT_BAS_PRETA` | Matrículas declaradas pretas | 2018–2025 ✅ |
| `QT_MAT_BAS_PARDA` | Matrículas declaradas pardas | 2018–2025 ✅ |
| `QT_MAT_BAS_AMARELA` | Matrículas declaradas amarelas | 2018–2025 ✅ |
| `QT_MAT_BAS_INDIGENA` | Matrículas declaradas indígenas | 2018–2025 ✅ |
| `QT_ESCOLAS` | Número de escolas com creche no município | 2018–2025 ✅ |
| `QT_ESC_FEDERAL` | Escolas federais | 2018–2025 ✅ |
| `QT_ESC_ESTADUAL` | Escolas estaduais | 2018–2025 ✅ |
| `QT_ESC_MUNICIPAL` | Escolas municipais | 2018–2025 ✅ |
| `QT_ESC_PRIVADA` | Escolas privadas | 2018–2025 ✅ |
| `QT_ESC_PUBLICA` | Escolas públicas (federal + estadual + municipal) | 2018–2025 ✅ |
| `QT_ESC_URBANA` | Escolas em zona urbana | 2018–2025 ✅ |
| `QT_ESC_RURAL` | Escolas em zona rural | 2018–2025 ✅ |

---

### População Municipal — IBGE (Estimativas Anuais)
**Fonte:** Base dos Dados / IBGE — Estimativas populacionais municipais (`br_ibge_populacao_municipio.csv`)  
**Cobertura temporal:** 2018–2025  
**Disponibilidade:** Todos os anos do intervalo ✅

| Variável | Descrição | Disponibilidade |
|----------|-----------|-----------------|
| `POP_TOTAL` | População total estimada do município | 2018–2025 ✅ |

---

### Censo Demográfico — IBGE 2022
**Fonte:** IBGE — Tabela 9514 — População residente por sexo, idade e forma de declaração (`tabela9514_UF_MUN_XX.xlsx`, 9 arquivos por estado do Nordeste)  
**Cobertura temporal:** 2022 apenas  
**Disponibilidade:** Somente 2022 ⚠️ — replicado para todos os anos no dataset temporal

| Variável | Descrição | Disponibilidade |
|----------|-----------|-----------------|
| `POP_0_4` | População residente de 0 a 4 anos | Apenas 2022 ⚠️ replicado |

> ⚠️ **Nota sobre POP_0_4:** O Censo Demográfico do IBGE tem periodicidade decenal. O dado mais recente consolidado é de 2022. Este valor foi replicado para todos os anos do dataset temporal (2018–2025), pois a variância nessa faixa etária entre anos é baixa. Esta limitação deve ser mencionada na seção de metodologia do artigo.

---

### Renda dos Responsáveis — IBGE Censo Demográfico 2022
**Fonte:** IBGE — Agregados por municípios (`Agregados_por_municipios_renda_responsavel_BR.csv`)  
**Cobertura temporal:** 2022 apenas  
**Disponibilidade:** Somente 2022 ⚠️ — replicado para todos os anos no dataset temporal

| Variável | Descrição | Disponibilidade |
|----------|-----------|-----------------|
| `RENDA_MEDIA_RESPONSAVEL` | Rendimento nominal médio mensal dos responsáveis com rendimento (R$) | Apenas 2022 ⚠️ replicado |
| `RENDA_MEDIANA_RESPONSAVEL` | Rendimento nominal mediano mensal dos responsáveis com rendimento (R$) | Apenas 2022 ⚠️ replicado |

> ⚠️ **Nota sobre renda:** Dado do Censo Demográfico 2022 (IBGE). Replicado para todos os anos do dataset temporal. A mediana é preferível à média para análises de vulnerabilidade, pois é menos sensível a valores extremos.

---

### PIB Municipal — IBGE
**Fonte:** Base dos Dados / IBGE — PIB dos municípios (`br_ibge_pib_municipio.csv`)  
**Cobertura temporal:** 2002–2023 (usado 2018–2023; 2024 e 2025 replicados de 2023)  
**Disponibilidade:** 2018–2023 ✅ | 2024–2025 ⚠️ replicado do último ano disponível

| Variável | Descrição | Disponibilidade |
|----------|-----------|-----------------|
| `PIB` | Produto Interno Bruto municipal (R$ mil) | 2018–2023 ✅ / 2024–2025 ⚠️ replicado |

> ⚠️ **Nota sobre PIB:** O IBGE divulga o PIB municipal com defasagem de aproximadamente 2 anos. Os anos 2024 e 2025 foram preenchidos com o valor de 2023. Esta limitação deve ser mencionada na seção de metodologia do artigo.

---

### Ocupação por Sexo — IBGE Censo Demográfico 2022
**Fonte:** IBGE — Tabela 10253 — Pessoas de 10 anos ou mais por situação de ocupação, sexo, cor/raça e grupos de idade (`Tabela_5_Situacao_de_ocupacao.xlsx`, aba `Tabela base do SIDRA 10253`)  
**Cobertura temporal:** 2022 apenas  
**Disponibilidade:** Somente 2022 ⚠️ — replicado para todos os anos no dataset temporal

| Variável | Descrição | Disponibilidade |
|----------|-----------|-----------------|
| `TOTAL_HOMENS` | Total de homens de 14 anos ou mais | Apenas 2022 ⚠️ replicado |
| `HOMENS_OCUPADOS` | Homens de 14 anos ou mais ocupados na semana de referência | Apenas 2022 ⚠️ replicado |
| `TOTAL_MULHERES` | Total de mulheres de 14 anos ou mais | Apenas 2022 ⚠️ replicado |
| `MULHERES_OCUPADAS` | Mulheres de 14 anos ou mais ocupadas na semana de referência | Apenas 2022 ⚠️ replicado |

> ⚠️ **Nota sobre ocupação:** Dado do Censo Demográfico 2022. Replicado para todos os anos do dataset temporal. A taxa de ocupação feminina é variável de interesse central para a hipótese do trabalho — municípios com maior ocupação feminina tendem a ter maior demanda por creche integral.

---

### Mães Jovens — IBGE Censo Demográfico 2022
**Fonte:** IBGE — Tabela 10077 — Mulheres de 12 anos ou mais que tiveram filhos nascidos vivos, por grupos de idade (`Tabela_10_Mulheres_com_filhos_nascidos_vivos.xlsx`, aba `Tabela SIDRA 10077 e 10078`)  
**Cobertura temporal:** 2022 apenas  
**Disponibilidade:** Somente 2022 ⚠️ — replicado para todos os anos no dataset temporal

| Variável | Descrição | Disponibilidade |
|----------|-----------|-----------------|
| `MULHERES_FILHOS_12_19` | Mulheres de 12 a 19 anos que tiveram filhos nascidos vivos (soma 12–14 + 15–19 anos) | Apenas 2022 ⚠️ replicado |
| `MULHERES_FILHOS_TOTAL` | Total de mulheres de 12 anos ou mais que tiveram filhos nascidos vivos | Apenas 2022 ⚠️ replicado |

> ⚠️ **Nota sobre mães jovens:** Permite calcular a proporção de mães jovens no município (`MULHERES_FILHOS_12_19 / MULHERES_FILHOS_TOTAL`), indicador relevante de vulnerabilidade social e contexto familiar associado à demanda por creche.

---

### Domicílios por Composição Familiar — IBGE Censo Demográfico 2022
**Fonte:** IBGE — Tabela 9882 — Domicílios particulares por presença de cônjuge e filhos, segundo o sexo do responsável (`tabela_MUN_XX.xlsx`, 9 arquivos por estado do Nordeste)  
**Cobertura temporal:** 2022 apenas  
**Disponibilidade:** Somente 2022 ⚠️ — replicado para todos os anos no dataset temporal

| Variável | Descrição | Disponibilidade |
|----------|-----------|-----------------|
| `DOMICILIOS_TOTAL` | Total de domicílios particulares no município | Apenas 2022 ⚠️ replicado |
| `DOM_SEM_CONJUGE_TOTAL` | Domicílios com responsável sem cônjuge com filho(s) e/ou enteado(s) — Total | Apenas 2022 ⚠️ replicado |
| `DOM_SEM_CONJUGE_HOMEM` | Domicílios com responsável sem cônjuge com filho(s) e/ou enteado(s) — Homens | Apenas 2022 ⚠️ replicado |
| `DOM_SEM_CONJUGE_MULHER` | Domicílios com responsável sem cônjuge com filho(s) e/ou enteado(s) — Mulheres | Apenas 2022 ⚠️ replicado |

> ⚠️ **Nota sobre domicílios:** Permite calcular a proporção de domicílios chefiados por mulher sozinha com filhos — indicador estrutural de vulnerabilidade familiar e demanda por creche integral.

---

### Composição do Rendimento Domiciliar — IBGE Censo Demográfico 2022
**Fonte:** IBGE — Tabela 10297 — Participação percentual na composição do rendimento nominal mensal domiciliar (`tabela10297__1_.xlsx`)  
**Cobertura temporal:** 2022 apenas  
**Disponibilidade:** Somente 2022 ⚠️ — replicado para todos os anos no dataset temporal

| Variável | Descrição | Disponibilidade |
|----------|-----------|-----------------|
| `PERC_RENDA_TRABALHO` | Participação percentual do rendimento do trabalho no rendimento domiciliar total | Apenas 2022 ⚠️ replicado |
| `PERC_RENDA_OUTRAS_FONTES` | Participação percentual de outras fontes (transferências, aposentadoria, etc.) no rendimento domiciliar total | Apenas 2022 ⚠️ replicado |

> ⚠️ **Nota sobre composição de renda:** Municípios com maior proporção de renda proveniente de outras fontes (transferências governamentais, aposentadorias) tendem a apresentar maior vulnerabilidade econômica ativa da população em idade de trabalho — indicador relevante para a hipótese de necessidade de creche integral.

---

### Indicadores IDH Municipal — Atlas do Desenvolvimento Humano (PNUD/IPEA)
**Fonte:** Programa das Nações Unidas para o Desenvolvimento — Atlas do Desenvolvimento Humano no Brasil (`mundo_onu_adh_municipio.csv`)  
**Cobertura temporal:** 1991, 2000, 2010 — utilizado apenas 2010  
**Disponibilidade:** Apenas 2010 ⚠️ — replicado para todos os anos no dataset temporal

| Variável | Descrição | Disponibilidade |
|----------|-----------|-----------------|
| `IDH_PROP_POBREZA` | Proporção de pessoas em situação de pobreza | Apenas 2010 ⚠️ replicado |
| `IDH_PROP_POBREZA_CRIANCAS` | Proporção de crianças em situação de pobreza | Apenas 2010 ⚠️ replicado |
| `IDH_PROP_VULNER_POBREZA` | Proporção de pessoas em vulnerabilidade à pobreza | Apenas 2010 ⚠️ replicado |
| `IDH_GINI` | Índice de Gini — desigualdade de renda | Apenas 2010 ⚠️ replicado |
| `IDH_RENDA_PC` | Renda per capita municipal | Apenas 2010 ⚠️ replicado |
| `IDH_EXPECT_ANOS_ESTUDO` | Expectativa de anos de estudo | Apenas 2010 ⚠️ replicado |
| `IDH_TAXA_FREQ_0_3` | Taxa de frequência escolar de crianças de 0 a 3 anos ⚠️ **variável suspeita de leakage conceitual — será validada posteriormente** | Apenas 2010 ⚠️ replicado |
| `IDH_RAZAO_DEPENDENCIA` | Razão de dependência demográfica | Apenas 2010 ⚠️ replicado |

> ⚠️ **Nota sobre dados IDH:** Os indicadores do Atlas do Desenvolvimento Humano são baseados nos Censos Demográficos do IBGE. O dado mais recente consolidado em nível municipal é de 2010, pois os resultados do Censo 2022 ainda não foram incorporados ao Atlas. Estas variáveis são utilizadas como **indicadores estruturais municipais**, associados aos diferentes anos do Censo Escolar devido à ausência de atualização censitária municipal equivalente para os períodos analisados. Sua interpretação deve considerar que refletem condições estruturais de longo prazo, não variações anuais.

> ⚠️ **Nota sobre `IDH_TAXA_FREQ_0_3`:** Esta variável representa a taxa de frequência escolar de crianças de 0 a 3 anos em 2010 e é **suspeita de leakage conceitual** em relação à variável alvo do modelo (`TAXA_CRE_INT`). Será validada e possivelmente excluída do conjunto de features na etapa de modelagem.

---

## Decisões Metodológicas

| Decisão | Justificativa |
|---------|---------------|
| Recorte regional: Nordeste | Foco em contexto de maior vulnerabilidade social e relevância para políticas públicas regionais |
| Unidade de análise: município-ano | Políticas públicas operam territorialmente; o município é a unidade de gestão da educação infantil |
| Variável alvo: `TAXA_CRE_INT` | `QT_MAT_INF_CRE_INT / QT_MAT_INF_CRE` — Proporção de crianças em creche integral como indicador de utilização da educação infantil em tempo integral |
| Replicação de `POP_0_4` | Dado decenal do IBGE — variância interanual baixa justifica replicação |
| Replicação de `RENDA_MEDIA/MEDIANA` | Dado decenal do IBGE — proxy estrutural de renda do município |
| Replicação de `PIB` 2024–2025 | IBGE divulga PIB com defasagem de ~2 anos — último disponível é 2023 |
| Replicação de `OCUPACAO` | Dado do Censo 2022 — proxy estrutural de mercado de trabalho por município |
| Replicação de `MAES_JOVENS` | Dado do Censo 2022 — indicador de vulnerabilidade social e contexto familiar |
| Replicação de `DOMICILIOS` | Dado do Censo 2022 — proxy de composição familiar e chefia monoparental |
| Replicação de `COMPOSICAO_RENDA` | Dado do Censo 2022 — indicador da dependência de transferências governamentais |
| Replicação de `IDH 2010` | Último dado censitário consolidado disponível — usado como indicador estrutural de longo prazo |
| Validação pendente de `IDH_TAXA_FREQ_0_3` | Suspeita de leakage conceitual com a variável alvo — será avaliada antes da modelagem |
| Cor/raça disponível 2018–2025 | Coletada via Censo Escolar INEP — disponível em todo o intervalo |

---

## Próximos Passos

- [ ] **Feature engineering**  Criar variáveis derivadas (taxas e proporções)
- [ ] **EDA** — distribuições, outliers, valores nulos e infinitos gerados por divisões
- [ ] **Correlação + VIF** — identificar redundâncias e multicolinearidade entre features derivadas; decidir com dados quais variáveis ficam no modelo final
- [ ] **Treinamento dos modelos** — Linear Regression (baseline), Random Forest Regressor, XGBoost Regressor
- [ ] **Análise SHAP** — interpretação dos fatores associados à variável alvo

---

*Última atualização: Maio 2026*
