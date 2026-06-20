Dados brutos
↓
Split treino/teste (sem leakage)
↓
AutoML (descoberta de arquiteturas promissoras)
↓
Seleção das top arquiteturas diversas
↓
Feature selection + Optuna para cada modelo
↓
Treinamento individual
↓
Simple Voting
↓
Stacking
↓
Avaliação completa
    • RMSE, MAE, R²
    • Bootstrap IC95%
    • Resíduos
    • y_true × y_pred
    • Bland–Altman
    • Correlação entre modelos
↓
Explicabilidade do melhor modelo
    • SHAP global
    • SHAP local
    • Contrafactuais (DiCE)
↓
Análise crítica dos contrafactuais
    • Features acionáveis
    • Features não acionáveis
    • Sparsity
    • Possíveis vieses/proxies