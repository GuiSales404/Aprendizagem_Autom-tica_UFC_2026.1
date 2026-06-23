# Counterfactual Actionability Analysis

## Configuration

- Non-actionable features: `['CD_MUN', 'COD_MUNICIPIO', 'COD_UF', 'ESTADO', 'ID', 'MUNICIPIO', 'NM_MUN', 'SG_UF', 'UF']`
- Actionable features: all changed features not listed as non-actionable.

## Summary

- Total counterfactuals analyzed: **30**
- Counterfactuals using only actionable features: **30** (100.0%)
- Counterfactuals using at least one non-actionable feature: **0** (0.0%)
- Mean number of changed features per counterfactual: **1.73**
- Mean actionable changes per counterfactual: **1.73**
- Mean non-actionable changes per counterfactual: **0.00**
- Most frequently changed feature: **IDH_EXPECT_ANOS_ESTUDO** (10 changes)

## Feature change frequency

| feature                       | feature_type   |   n_changes |
|:------------------------------|:---------------|------------:|
| IDH_EXPECT_ANOS_ESTUDO        | actionable     |          10 |
| PROP_MAT_BRANCA               | actionable     |           9 |
| IDH_RAZAO_DEPENDENCIA         | actionable     |           8 |
| PROP_MAT_INDIGENA             | actionable     |           3 |
| PROP_DOM_SEM_CONJUGE_MULHER   | actionable     |           3 |
| PERC_RENDA_OUTRAS_FONTES      | actionable     |           3 |
| COBERTURA_CRECHE_0_3          | actionable     |           3 |
| PROP_MAT_PARDA                | actionable     |           2 |
| RENDA_MEDIANA_RESPONSAVEL     | actionable     |           2 |
| LOG_POP_TOTAL                 | actionable     |           2 |
| MEDIA_MAT_0_3_POR_ESCOLA      | actionable     |           2 |
| TAXA_OCUP_MASC                | actionable     |           2 |
| ESCOLAS_CRECHE_POR_1000_CRIAN | actionable     |           1 |
| IDH_PROP_POBREZA_CRIANCAS     | actionable     |           1 |
| PROP_MAT_PRETA                | actionable     |           1 |


## Change frequency by actionability type

| feature_type   |   n_changes |
|:---------------|------------:|
| actionable     |          52 |


## Interpretation guide

- If many counterfactuals rely on non-actionable features, the model may be using useful predictive proxies that are weak for intervention-oriented explanations.
- If actionable features frequently appear with small sparse changes, the explanations are more useful for practical or policy interpretation.
- A strong dependence on geographic or identity-like variables should be treated as a potential proxy/bias issue, even when predictive performance is good.


## Generated files

- `counterfactual_actionability_profile.png`
- `counterfactual_change_counts_by_actionability.csv`
- `counterfactual_feature_change_counts_by_type.csv`
- `counterfactual_feature_changes_labeled.csv`
- `counterfactual_level_actionability_summary.csv`
- `counterfactual_numeric_delta_summary.csv`
- `counterfactual_sparsity_histogram.png`
- `top_changed_features_by_actionability.png`
- `top_numeric_mean_abs_delta.png`