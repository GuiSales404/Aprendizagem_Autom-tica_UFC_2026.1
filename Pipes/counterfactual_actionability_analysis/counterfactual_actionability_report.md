# Counterfactual Actionability Analysis

## Configuration

- Non-actionable features: `['CD_MUN', 'COD_MUNICIPIO', 'COD_UF', 'ESTADO', 'ID', 'MUNICIPIO', 'NM_MUN', 'SG_UF', 'UF']`
- Actionable features: all changed features not listed as non-actionable.

## Summary

- Total counterfactuals analyzed: **30**
- Counterfactuals using only actionable features: **10** (33.3%)
- Counterfactuals using at least one non-actionable feature: **20** (66.7%)
- Mean number of changed features per counterfactual: **1.77**
- Mean actionable changes per counterfactual: **1.10**
- Mean non-actionable changes per counterfactual: **0.67**
- Most frequently changed feature: **SG_UF** (20 changes)

## Feature change frequency

| feature                     | feature_type   |   n_changes |
|:----------------------------|:---------------|------------:|
| SG_UF                       | non_actionable |          20 |
| COBERTURA_CRECHE_0_3        | actionable     |          10 |
| IDH_EXPECT_ANOS_ESTUDO      | actionable     |           4 |
| LOG_PIB_PC                  | actionable     |           3 |
| LOG_POP_TOTAL               | actionable     |           2 |
| MEDIA_MAT_0_3_POR_ESCOLA    | actionable     |           2 |
| PIB_PC                      | actionable     |           2 |
| PROP_ESC_PUBLICA            | actionable     |           2 |
| PROP_MAT_PRETA              | actionable     |           2 |
| IDH_RAZAO_DEPENDENCIA       | actionable     |           1 |
| PERC_RENDA_OUTRAS_FONTES    | actionable     |           1 |
| PROP_DOM_SEM_CONJUGE_MULHER | actionable     |           1 |
| PROP_ESC_URBANA             | actionable     |           1 |
| PROP_MAT_BRANCA             | actionable     |           1 |
| PROP_MAT_PARDA              | actionable     |           1 |


## Change frequency by actionability type

| feature_type   |   n_changes |
|:---------------|------------:|
| actionable     |          33 |
| non_actionable |          20 |


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