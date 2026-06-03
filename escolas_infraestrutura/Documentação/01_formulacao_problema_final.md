# Algoritmos de Machine Learning para Classificação do Nível de Infraestrutura de Escolas Públicas de Ensino Fundamental no Nordeste Brasileiro

## 1. Descrição do Problema

Este trabalho tem como objetivo **prever o nível de infraestrutura de escolas públicas de Ensino Fundamental da região Nordeste do Brasil** a partir de características institucionais e contextuais, utilizando algoritmos de classificação supervisionada treinados com dados do Censo Escolar 2025.

A variável-alvo é o nível de infraestrutura escolar, definido segundo a Escala de Infraestrutura Geral da UNESCO (2019), que organiza as escolas em sete níveis ordinais com base na presença ou ausência de itens físicos e de equipamentos. As features do modelo são exclusivamente variáveis de perfil institucional: localização, porte, oferta educacional, composição do quadro de profissionais e perfil do gestor escolar. Nenhuma variável de infraestrutura física integra o vetor de entrada do modelo.

A decisão metodológica central é a separação entre o que compõe a variável-alvo e o que compõe as features: o modelo aprende a classificar o nível de infraestrutura de uma escola a partir de informações que não descrevem diretamente essa infraestrutura. Dado o perfil institucional de qualquer escola pública de Ensino Fundamental do Nordeste com dados disponíveis no Censo Escolar, o modelo produz uma previsão do nível de infraestrutura em que ela se enquadra.

---

## 2. Referencial Teórico

### 2.1 Escala de Infraestrutura Escolar

A variável-alvo deste trabalho é baseada na **Escala de Infraestrutura Geral** proposta pela UNESCO (2019) no documento *Qualidade da infraestrutura das escolas públicas do ensino fundamental no Brasil*. A escala foi construída a partir de 61 itens do Censo Escolar, com parâmetros estimados por meio da **Teoria de Resposta ao Item (TRI)**, e organiza as escolas em **sete níveis** (I a VII), do mais precário ao mais completo.

| Nível | Pontuação | Características principais |
|-------|-----------|---------------------------|
| I | ≤ 2 | Sem banheiro, sem água, sem energia elétrica |
| II | +2 a 4 | Água de poço, banheiro, energia, TV e DVD |
| III | +4 a 5 | Água de rede pública, coleta de lixo, secretaria, sala de professores, internet sem banda larga |
| IV | +5 a 6 | Laboratório de informática, banda larga, biblioteca, quadra descoberta |
| V | +6 a 7 | Laboratório de ciências, quadra coberta, área verde, refeitório, banheiro PNE |
| VI | +7 a 8 | Auditório, pátio coberto e descoberto, 20+ computadores, acessibilidade adequada |
| VII | ≥ 8 | Recursos de inclusão (Braille, comunicação alternativa), infraestrutura completa |

Para fins deste estudo, os critérios qualitativos do Quadro 1 da UNESCO (2019, p. 57) foram operacionalizados a partir das variáveis binárias disponíveis no Censo Escolar 2025, gerando uma **pontuação aproximada** para cada escola. Esta é uma limitação metodológica reconhecida: a escala original utiliza parâmetros TRI não disponíveis publicamente, de modo que a classificação aqui produzida é uma aproximação baseada nos critérios descritivos publicados pela UNESCO. Tal abordagem é amplamente utilizada na literatura quando os parâmetros originais de escalonamento não estão disponíveis.

---

## 3. Definição Formal do Problema

Seja $\mathcal{E}$ o conjunto de $n = 35.017$ escolas públicas de Ensino Fundamental da região Nordeste do Brasil, observadas no Censo Escolar 2025.

Para cada escola $e_i \in \mathcal{E}$, define-se:

- $\mathbf{x}_i \in \mathbb{R}^p$: vetor de $p$ features institucionais e contextuais da escola $i$, incluindo características de localização, porte, oferta educacional, corpo de profissionais e perfil do gestor escolar.

- $y_i \in \{1, 2, 3, 4, 5, 6, 7\}$: nível de infraestrutura da escola $i$, calculado a partir dos critérios da Escala de Infraestrutura Geral da UNESCO (2019), com base em variáveis físicas e de equipamentos do Censo Escolar **não incluídas em $\mathbf{x}_i$**.

O problema de pesquisa consiste em encontrar uma função $f: \mathbb{R}^p \rightarrow \{1, 2, 3, 4, 5, 6, 7\}$ tal que:

$$\hat{y}_i = f(\mathbf{x}_i)$$

onde $\hat{y}_i$ é a previsão do nível de infraestrutura da escola $i$, e $f$ é aprendida a partir de um conjunto de treinamento $\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^{n}$.

A separação entre as variáveis que compõem $y_i$ (infraestrutura física) e as variáveis em $\mathbf{x}_i$ (perfil institucional e contextual) é a decisão metodológica central deste trabalho: o modelo aprende a inferir o nível de infraestrutura a partir de informações que **não descrevem diretamente a infraestrutura**.

### 3.1 Objetivo geral

Prever o nível de infraestrutura física de escolas públicas de Ensino Fundamental do Nordeste brasileiro utilizando algoritmos de classificação supervisionada aplicados a características institucionais e contextuais extraídas do Censo Escolar 2025.

### 3.2 Objetivos específicos

- Comparar o desempenho de dois algoritmos de classificação no problema proposto.
- Identificar quais features institucionais apresentam maior poder preditivo sobre o nível de infraestrutura.
- Avaliar a capacidade de generalização do modelo para escolas não vistas durante o treinamento.

---

## 4. Natureza do Problema de ML

O problema é de **classificação multiclasse supervisionada**, com as seguintes características:

- **Classes:** 7 níveis ordinais (I a VII), com distribuição desbalanceada
- **Instâncias:** 35.017 escolas
- **Features:** variáveis numéricas e categóricas de perfil institucional
- **Desbalanceamento:** o nível I representa 2.1% das instâncias e o nível VII representa 28.2%, o que exige estratégias específicas de tratamento

| Nível | N | % |
|-------|------|------|
| I | 725 | 2.1% |
| II | 5.590 | 16.0% |
| III | 4.169 | 11.9% |
| IV | 4.429 | 12.6% |
| V | 4.990 | 14.3% |
| VI | 5.242 | 15.0% |
| VII | 9.872 | 28.2% |

A natureza ordinal dos níveis (I < II < ... < VII) é uma propriedade relevante: um erro de previsão que classifica uma escola de nível I como nível II é menos grave do que classificá-la como nível VII. Modelos que exploram essa ordenação tendem a produzir resultados mais interpretáveis.

---

## 5. Métricas de Avaliação

Dado o desbalanceamento das classes, a acurácia simples é uma métrica insuficiente. As métricas utilizadas serão:

**Acurácia:** proporção de previsões corretas sobre o total:

$$\text{Acurácia} = \frac{\sum_{c=1}^{7} TP_c}{n}$$

**F1-score macro:** média não ponderada do F1 por classe, dando peso igual a todas as classes independentemente do tamanho:

$$\text{F1-macro} = \frac{1}{7} \sum_{c=1}^{7} \frac{2 \cdot \text{Precision}_c \cdot \text{Recall}_c}{\text{Precision}_c + \text{Recall}_c}$$

**F1-score weighted:** média ponderada pelo número de instâncias de cada classe.

**Matriz de confusão:** análise qualitativa dos erros por nível.

---

## 6. Estratégia de Modelagem

Serão treinados e comparados dois algoritmos:

1. **Algoritmo baseline** (visto na disciplina): a definir entre Regressão Logística Multinomial, SVM multiclasse ou Random Forest.
2. **Algoritmo novo** (não visto na disciplina): a definir.

Ambos serão avaliados com **validação cruzada k-fold** (k=10) e testados em um conjunto de teste holdout (20% dos dados), estratificado por nível para preservar a distribuição das classes.

---

## Referências

- UNESCO. *Qualidade da infraestrutura das escolas públicas do ensino fundamental no Brasil*. Brasília: UNESCO, 2019.
- INEP. *Censo Escolar da Educação Básica 2025 — Microdados*. Brasília: INEP, 2025.
