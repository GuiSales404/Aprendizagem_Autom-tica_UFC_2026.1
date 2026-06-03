# Construção do Dataset

## 1. Fontes de Dados

Os dados utilizados neste estudo são provenientes do **Censo Escolar da Educação Básica 2025**, publicado pelo Instituto Nacional de Estudos e Pesquisas Educacionais Anísio Teixeira (INEP). O Censo Escolar é o principal levantamento estatístico da educação básica brasileira, realizado anualmente em regime de colaboração entre o MEC e as secretarias estaduais e municipais de educação, com cobertura de todas as escolas do país.

Os microdados do Censo Escolar 2025 são distribuídos em tabelas separadas, cada uma contendo informações de uma dimensão específica do sistema educacional. As tabelas utilizadas neste estudo foram:

| Tabela | Conteúdo | Linhas (Brasil) |
|--------|----------|-----------------|
| Tabela\_Escola\_2025 | Dados institucionais, infraestrutura física e equipamentos | 214.192 |
| Tabela\_Matricula\_2025 | Quantitativo de matrículas por etapa e modalidade | 178.766 |
| Tabela\_Docente\_2025 | Quantitativo de docentes por etapa e modalidade | 178.772 |
| Tabela\_Turma\_2025 | Quantitativo de turmas por etapa e modalidade | 178.772 |
| Tabela\_Gestor\_Escolar\_2025 | Perfil dos gestores escolares por formação e forma de acesso ao cargo | 180.540 |

Todas as tabelas compartilham o identificador `CO_ENTIDADE` (código da escola), que foi utilizado como chave de junção entre elas.

---

## 2. Recorte da Amostra

A amostra foi construída a partir de três critérios sucessivos de filtragem aplicados sobre a Tabela\_Escola:

**Critério 1: Região geográfica.** Foram mantidas apenas as escolas localizadas nos nove estados da região Nordeste, identificados pela variável `SG_UF` com os valores: MA, PI, CE, RN, PB, PE, AL, SE e BA. O recorte regional é justificado pela própria escala de infraestrutura da UNESCO (2019), que identifica o Nordeste como a região com maior concentração de escolas nos níveis mais críticos de infraestrutura, tornando-o o contexto mais relevante para o problema de classificação proposto.

**Critério 2: Dependência administrativa.** Foram mantidas apenas as escolas públicas, identificadas pelos valores 1 (federal), 2 (estadual) e 3 (municipal) na variável `TP_DEPENDENCIA`. Escolas privadas foram excluídas por apresentarem lógica de financiamento e manutenção distinta, o que tornaria a comparação de infraestrutura metodologicamente inadequada.

**Critério 3: Etapa de ensino.** Foram mantidas apenas as escolas com matrículas no Ensino Fundamental, identificadas pelas variáveis `IN_COMUM_FUND_AI` (Anos Iniciais) ou `IN_COMUM_FUND_AF` (Anos Finais) com valor igual a 1. O foco no Ensino Fundamental é coerente com a escala de infraestrutura da UNESCO (2019), que foi construída especificamente para essa etapa.

Após a aplicação dos três critérios, a amostra resultante contém **35.017 escolas**.

| Etapa de filtragem | N |
|-------------------|------|
| Total de escolas no Brasil | 214.192 |
| Após filtro Nordeste | 74.335 |
| Após filtro público + Ensino Fundamental | 35.017 |

---

## 3. Construção da Variável-Alvo

A variável-alvo `NIVEL_UNESCO` foi construída a partir dos critérios qualitativos do Quadro 1 da UNESCO (2019, p. 57), que descreve sete níveis de infraestrutura escolar com base na presença ou ausência de itens físicos e de equipamentos.

A escala original foi calculada com parâmetros estimados por Teoria de Resposta ao Item (TRI) a partir de 61 itens do Censo Escolar, com pontos de corte definidos por julgamento substantivo de especialistas. Como os parâmetros TRI não estão disponíveis publicamente, foi adotada uma **operacionalização aproximada**: para cada escola, calculou-se uma pontuação de infraestrutura com base nas variáveis binárias do Censo Escolar 2025 que correspondem aos itens descritos em cada nível da escala. As escolas foram então classificadas nos sete níveis de acordo com os intervalos de pontuação definidos pela UNESCO.

As variáveis utilizadas para calcular a pontuação de infraestrutura e seus respectivos pesos foram organizadas em seis grupos:

**Serviços básicos:** abastecimento de água por rede pública (+1,5), por poço artesiano ou cisterna (+0,5), fornecimento de energia elétrica pela rede pública (+1,0), esgoto por rede pública (+1,0) ou fossa (+0,5), e coleta de lixo (+0,5).

**Banheiro:** existência de banheiro (+0,5) e banheiro acessível para pessoas com deficiência (+0,5).

**Espaços pedagógicos:** biblioteca ou sala de leitura (+0,5), biblioteca (+0,5), laboratório de informática (+0,5), laboratório de ciências (+0,5), quadra de esportes descoberta (+0,5), quadra coberta (+0,5), refeitório (+0,5), auditório (+0,5), área verde (+0,25), parque infantil (+0,25), pátio coberto (+0,25) e pátio descoberto (+0,25).

**Equipamentos:** televisão (+0,25), DVD (+0,25), computador administrativo (+0,25), computador para alunos (+0,25), impressora (+0,25) e projetor multimídia (+0,25).

**Internet:** acesso à internet (+0,25), banda larga (+0,25) e internet para uso dos alunos (+0,25).

**Acessibilidade:** rampas (+0,25), corrimão (+0,25), ausência de qualquer recurso de acessibilidade (-0,25), e sala de atendimento educacional especializado (+0,5).

A pontuação total de cada escola foi então convertida nos sete níveis da escala conforme os intervalos originais da UNESCO: nível I (≤ 2), nível II (+2 a 4), nível III (+4 a 5), nível IV (+5 a 6), nível V (+6 a 7), nível VI (+7 a 8) e nível VII (≥ 8).

Esta é uma limitação metodológica reconhecida do estudo. A pontuação obtida é uma aproximação dos parâmetros TRI originais, e não uma replicação exata da escala. No entanto, a operacionalização foi construída de forma a respeitar a hierarquia dos itens conforme descrita pela UNESCO, e os pesos atribuídos refletem a progressão cumulativa de qualidade entre os níveis.

---

## 4. Construção das Features

As features do modelo foram selecionadas a partir das cinco tabelas do Censo Escolar, com o critério de **não incluir nenhuma variável de infraestrutura física ou de equipamentos** — ou seja, nenhuma variável utilizada no cálculo da variável-alvo.

As tabelas foram integradas por meio de junção à esquerda (*left join*) utilizando `CO_ENTIDADE` como chave, tendo a Tabela\_Escola como base. Todas as escolas da amostra foram preservadas após a junção.

As features foram organizadas em seis grupos temáticos:

**Localização e dependência administrativa:** tipo de localização (urbana ou rural), localização diferenciada (assentamento, terra indígena, comunidade quilombola) e dependência administrativa (federal, estadual ou municipal).

**Local de funcionamento:** indicadores sobre se a escola funciona em prédio escolar próprio, em galpão ou barracão, em salas de outra escola, e se o prédio é compartilhado com outro estabelecimento.

**Porte e capacidade:** número de salas utilizadas, número de salas climatizadas, total de matrículas no Ensino Fundamental (anos iniciais e finais), total de docentes no Ensino Fundamental e total de turmas no Ensino Fundamental.

**Oferta educacional:** indicadores de oferta de EJA, de anos iniciais e finais do Ensino Fundamental em turmas comuns, de mediação presencial e a distância, e de alimentação escolar.

**Profissionais de apoio:** quantitativo de psicólogos, pedagogos, nutricionistas, assistentes sociais, coordenadores, bibliotecários, seguranças e auxiliares administrativos em atuação na escola.

**Perfil do gestor escolar:** quantitativo de gestores com graduação em licenciatura, com mestrado e com doutorado; e quantitativo de gestores por forma de acesso ao cargo (eleição, concurso ou indicação).

A tabela a seguir lista todas as features incluídas no modelo:

| Feature | Descrição | Fonte |
|---------|-----------|-------|
| `TP_LOCALIZACAO` | Urbana (1) ou rural (2) | Escola |
| `TP_LOCALIZACAO_DIFERENCIADA` | Assentamento, terra indígena, quilombola | Escola |
| `TP_DEPENDENCIA` | Federal, estadual ou municipal | Escola |
| `TP_OCUPACAO_PREDIO_ESCOLAR` | Próprio, alugado ou cedido | Escola |
| `QT_SALAS_UTILIZADAS` | Total de salas utilizadas | Escola |
| `QT_SALAS_UTILIZA_CLIMATIZADAS` | Salas climatizadas | Escola |
| `QT_MAT_FUND` | Matrículas no Ensino Fundamental | Matrícula |
| `QT_MAT_FUND_AI` | Matrículas nos Anos Iniciais | Matrícula |
| `QT_MAT_FUND_AF` | Matrículas nos Anos Finais | Matrícula |
| `IN_EJA` | Oferta de EJA | Escola |
| `IN_COMUM_FUND_AI` | Turmas comuns nos Anos Iniciais | Escola |
| `IN_COMUM_FUND_AF` | Turmas comuns nos Anos Finais | Escola |
| `QT_PROF_PSICOLOGO` | Psicólogos na escola | Escola |
| `QT_PROF_PEDAGOGIA` | Pedagogos na escola | Escola |
| `QT_PROF_NUTRICIONISTA` | Nutricionistas na escola | Escola |
| `QT_PROF_ASSIST_SOCIAL` | Assistentes sociais na escola | Escola |
| `QT_PROF_COORDENADOR` | Coordenadores de turno/disciplina | Escola |
| `QT_PROF_BIBLIOTECARIO` | Bibliotecários na escola | Escola |
| `QT_PROF_SEGURANCA` | Seguranças na escola | Escola |
| `QT_PROF_ADMINISTRATIVOS` | Auxiliares administrativos | Escola |
| `TP_PROPOSTA_PEDAGOGICA` | PPP atualizado nos últimos 12 meses | Escola |
| `QT_GEST_BAS_ESCO_SUP_GRAD_LICEN` | Gestores com licenciatura | Gestor |
| `QT_GEST_BAS_ACESSO_CARGO_ELEIC` | Gestores eleitos | Gestor |
| `QT_GEST_BAS_ACESSO_CARGO_INDIC` | Gestores indicados | Gestor |

---

## 5. Análise Exploratória e Limpeza das Features

Antes da modelagem, foi realizada uma análise exploratória para identificar e tratar problemas de qualidade nas features selecionadas. Foram investigados três aspectos: variância baixa, colinearidade entre features e distribuição das variáveis numéricas.

### 5.1 Remoção de features com variância baixa

Features com variância muito baixa carregam pouca informação discriminante e tendem a prejudicar a generalização dos modelos. O critério adotado foi a remoção de features em que 95% ou mais das observações apresentam o mesmo valor.

A análise identificou 13 features nessa condição, todas removidas do conjunto final:

| Feature removida | Valor dominante | Frequência |
|-----------------|----------------|------------|
| `IN_REGULAR` | 1 (sim) | 100,0% |
| `IN_MEDIACAO_PRESENCIAL` | 1 (sim) | 100,0% |
| `IN_MEDIACAO_EAD` | 0 (não) | 100,0% |
| `IN_ALIMENTACAO` | 1 (sim) | 100,0% |
| `QT_GEST_BAS_ESCO_SUP_POS_MESTRA` | 0 | 97,5% |
| `QT_GEST_BAS_ESCO_SUP_POS_DOUTO` | 0 | 99,6% |
| `QT_GEST_BAS_ACESSO_CARGO_CONC` | 0 | 99,0% |
| `IN_LOCAL_FUNC_PREDIO_ESCOLAR` | 1 (sim) | 98,9% |
| `IN_LOCAL_FUNC_GALPAO` | 0 (não) | 98,6% |
| `IN_LOCAL_FUNC_SALAS_OUTRA_ESC` | 0 (não) | 95,6% |
| `IN_PREDIO_COMPARTILHADO` | 0 (não) | 95,7% |
| `IN_ESP_EXCLUSIVA_FUND_AI` | 0 (não) | 99,7% |
| `IN_ESP_EXCLUSIVA_FUND_AF` | 0 (não) | 99,9% |

O alto percentual de alimentação escolar (100%) e mediação presencial (100%) reflete a cobertura universal dessas políticas nas escolas públicas de Ensino Fundamental do Nordeste, o que, embora relevante do ponto de vista educacional, torna essas variáveis inúteis para fins de classificação.

### 5.2 Remoção de features colineares

A correlação de Spearman entre todas as features numéricas foi calculada para identificar pares com alta redundância. Foram encontrados três pares com correlação superior a 0,85:

| Par | Correlação |
|-----|-----------|
| `QT_TUR_FUND` e `QT_MAT_FUND` | 0,96 |
| `QT_TUR_FUND` e `QT_DOC_FUND` | 0,91 |
| `QT_DOC_FUND` e `QT_MAT_FUND` | 0,90 |

As três variáveis (`QT_TUR_FUND`, `QT_MAT_FUND` e `QT_DOC_FUND`) medem dimensões distintas do mesmo construto subjacente: o porte da escola. Em situações de alta colinearidade, manter todas as variáveis não acrescenta informação ao modelo e pode dificultar a interpretação dos coeficientes em algoritmos lineares.

Optou-se por manter `QT_MAT_FUND` e remover `QT_TUR_FUND` e `QT_DOC_FUND`. A justificativa é dupla: `QT_MAT_FUND` apresentou a maior correlação com a variável-alvo (Spearman = 0,767) entre as três, e o número de matrículas é o indicador de porte mais diretamente associado à demanda por infraestrutura.

### 5.3 Tratamento de valores ausentes e extremos

Valores ausentes foram tratados de acordo com o tipo de variável: para variáveis binárias (`IN_`), os valores ausentes foram substituídos por 0 (ausência); para variáveis quantitativas (`QT_`), pela mediana da distribuição; e para variáveis categóricas (`TP_`), pela moda.

Valores marcados com o código 88888 no Censo Escolar, que indicam registros com valores extremos segundo critério do INEP, foram também substituídos pela mediana da respectiva variável.

---

## 6. Dataset Final

Após todas as etapas de construção e limpeza, o dataset final apresenta as seguintes características:

| Característica | Valor |
|---------------|-------|
| Número de instâncias | 35.017 |
| Número de features | 24 |
| Variável-alvo | `NIVEL_UNESCO` (1 a 7) |
| Valores ausentes | 0 |
| Fontes integradas | 5 tabelas do Censo Escolar 2025 |

A distribuição da variável-alvo no dataset final é:

| Nível | N | % |
|-------|------|------|
| I | 725 | 2,1% |
| II | 5.590 | 16,0% |
| III | 4.169 | 11,9% |
| IV | 4.429 | 12,6% |
| V | 4.990 | 14,3% |
| VI | 5.242 | 15,0% |
| VII | 9.872 | 28,2% |

O dataset apresenta desbalanceamento entre as classes, com o nível I representando apenas 2,1% das instâncias e o nível VII representando 28,2%. Esse aspecto será considerado na escolha e avaliação dos algoritmos de classificação.

---

## Referências

- UNESCO. *Qualidade da infraestrutura das escolas públicas do ensino fundamental no Brasil*. Brasília: UNESCO, 2019.
- INEP. *Censo Escolar da Educação Básica 2025 — Microdados*. Brasília: INEP, 2025.
