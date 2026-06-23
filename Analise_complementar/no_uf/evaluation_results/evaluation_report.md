# Model Evaluation Report

## Overall ranking

| model_name     |     rmse |   rmse_ci_low |   rmse_ci_high |      mae |   mae_ci_low |   mae_ci_high |       r2 |
|:---------------|---------:|--------------:|---------------:|---------:|-------------:|--------------:|---------:|
| stacking       | 0.328539 |      0.307753 |       0.348854 | 0.274823 |     0.256447 |      0.293647 | 0.162871 |
| simple_average | 0.329268 |      0.308651 |       0.34924  | 0.277164 |     0.259412 |      0.295873 | 0.159148 |
| lightgbm       | 0.330375 |      0.308784 |       0.351057 | 0.276272 |     0.257331 |      0.295683 | 0.153488 |
| xgboost        | 0.33217  |      0.311048 |       0.35262  | 0.27893  |     0.260401 |      0.298151 | 0.144264 |
| random_forest  | 0.332868 |      0.312029 |       0.353218 | 0.281791 |     0.263643 |      0.300623 | 0.140663 |
| catboost       | 0.332931 |      0.31213  |       0.353267 | 0.277716 |     0.259306 |      0.296893 | 0.140339 |
| mlp            | 0.33331  |      0.312478 |       0.354234 | 0.277287 |     0.259002 |      0.296418 | 0.138378 |


## Main findings

- Best overall model by RMSE: **stacking** (RMSE = 0.32854, MAE = 0.27482, R² = 0.16287).
- Best individual/base model: **lightgbm** (RMSE = 0.33037).
- Stacking changed RMSE by **0.56%** relative to the best individual model.
- Stacking had lower absolute error than the best individual model in **51.8%** of test samples.


## Prediction correlation

|                     |   stacking_pred |   simple_average_pred |   lightgbm_pred |   catboost_pred |   random_forest_pred |   xgboost_pred |   mlp_pred |
|:--------------------|----------------:|----------------------:|----------------:|----------------:|---------------------:|---------------:|-----------:|
| stacking_pred       |        1        |              0.990037 |        0.922333 |        0.955779 |             0.925556 |       0.935114 |   0.907578 |
| simple_average_pred |        0.990037 |              1        |        0.96348  |        0.962442 |             0.962171 |       0.971838 |   0.84792  |
| lightgbm_pred       |        0.922333 |              0.96348  |        1        |        0.918285 |             0.95223  |       0.968383 |   0.716753 |
| catboost_pred       |        0.955779 |              0.962442 |        0.918285 |        1        |             0.919099 |       0.92782  |   0.768358 |
| random_forest_pred  |        0.925556 |              0.962171 |        0.95223  |        0.919099 |             1        |       0.964915 |   0.718046 |
| xgboost_pred        |        0.935114 |              0.971838 |        0.968383 |        0.92782  |             0.964915 |       1        |   0.733899 |
| mlp_pred            |        0.907578 |              0.84792  |        0.716753 |        0.768358 |             0.718046 |       0.733899 |   1        |


## Generated figures

- `bland_altman_lightgbm.png`
- `bland_altman_simple_average.png`
- `bland_altman_stacking.png`
- `model_ranking_rmse.png`
- `residual_histogram_lightgbm.png`
- `residual_histogram_simple_average.png`
- `residual_histogram_stacking.png`
- `residuals_vs_pred_lightgbm.png`
- `residuals_vs_pred_simple_average.png`
- `residuals_vs_pred_stacking.png`
- `stacking_error_improvement_histogram.png`
- `test_prediction_correlation.png`
- `true_vs_pred_lightgbm.png`
- `true_vs_pred_simple_average.png`
- `true_vs_pred_stacking.png`