# Model Evaluation Report

## Overall ranking

| model_name     |     rmse |   rmse_ci_low |   rmse_ci_high |      mae |   mae_ci_low |   mae_ci_high |       r2 |
|:---------------|---------:|--------------:|---------------:|---------:|-------------:|--------------:|---------:|
| stacking       | 0.304877 |      0.28063  |       0.327574 | 0.242612 |     0.224281 |      0.26232  | 0.279111 |
| simple_average | 0.305599 |      0.282315 |       0.327924 | 0.245673 |     0.227402 |      0.264898 | 0.275691 |
| xgboost        | 0.306604 |      0.283216 |       0.32888  | 0.247992 |     0.230204 |      0.267078 | 0.27092  |
| random_forest  | 0.30839  |      0.28433  |       0.330625 | 0.249495 |     0.231688 |      0.268628 | 0.262403 |
| lightgbm       | 0.308613 |      0.28541  |       0.330849 | 0.249725 |     0.23152  |      0.268806 | 0.261333 |
| catboost       | 0.308865 |      0.285331 |       0.331657 | 0.24964  |     0.230846 |      0.269362 | 0.260128 |
| mlp            | 0.312466 |      0.288516 |       0.335565 | 0.244054 |     0.224934 |      0.264394 | 0.242775 |


## Main findings

- Best overall model by RMSE: **stacking** (RMSE = 0.30488, MAE = 0.24261, R² = 0.27911).
- Best individual/base model: **xgboost** (RMSE = 0.30660).
- Stacking changed RMSE by **0.56%** relative to the best individual model.
- Stacking had lower absolute error than the best individual model in **55.5%** of test samples.


## Prediction correlation

|                     |   stacking_pred |   simple_average_pred |   lightgbm_pred |   catboost_pred |   random_forest_pred |   xgboost_pred |   mlp_pred |
|:--------------------|----------------:|----------------------:|----------------:|----------------:|---------------------:|---------------:|-----------:|
| stacking_pred       |        1        |              0.999857 |        0.977062 |        0.974437 |             0.974785 |       0.985455 |   0.908348 |
| simple_average_pred |        0.999857 |              1        |        0.974857 |        0.972616 |             0.976797 |       0.985874 |   0.91041  |
| lightgbm_pred       |        0.977062 |              0.974857 |        1        |        0.943339 |             0.946931 |       0.972931 |   0.842184 |
| catboost_pred       |        0.974437 |              0.972616 |        0.943339 |        1        |             0.952812 |       0.966337 |   0.834338 |
| random_forest_pred  |        0.974785 |              0.976797 |        0.946931 |        0.952812 |             1        |       0.977162 |   0.840464 |
| xgboost_pred        |        0.985455 |              0.985874 |        0.972931 |        0.966337 |             0.977162 |       1        |   0.844552 |
| mlp_pred            |        0.908348 |              0.91041  |        0.842184 |        0.834338 |             0.840464 |       0.844552 |   1        |


## Generated figures

- `bland_altman_simple_average.png`
- `bland_altman_stacking.png`
- `bland_altman_xgboost.png`
- `model_ranking_rmse.png`
- `residual_histogram_simple_average.png`
- `residual_histogram_stacking.png`
- `residual_histogram_xgboost.png`
- `residuals_vs_pred_simple_average.png`
- `residuals_vs_pred_stacking.png`
- `residuals_vs_pred_xgboost.png`
- `stacking_error_improvement_histogram.png`
- `test_prediction_correlation.png`
- `true_vs_pred_simple_average.png`
- `true_vs_pred_stacking.png`
- `true_vs_pred_xgboost.png`