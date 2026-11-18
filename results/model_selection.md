# Model selection note

Selection metric: mean trip-grouped CV Macro F1, with CV Top-1 accuracy as the tie-breaker. Training accuracy was not used.

**Selected model:** `random_forest`

| Model | Family | CV Macro F1 | CV Top-1 | CV Log Loss |
| :--- | :--- | ---: | ---: | ---: |
| random_forest | ensemble_forest | 0.8894 | 0.8182 | 0.1784 |
| extra_trees | ensemble | 0.8858 | 0.8137 | 0.1773 |
| mlp | neural | 0.8850 | 0.8144 | 0.1603 |
| knn | instance_based | 0.8775 | 0.8062 | 0.4908 |
| hist_gradient_boosting | boosting_hist | 0.8679 | 0.8186 | 0.2054 |
| lightgbm | boosting | 0.8647 | 0.8206 | 0.2119 |
| adaboost | boosting_adaptive | 0.8626 | 0.8070 | 0.3472 |
| decision_tree | tree | 0.8409 | 0.8050 | 0.5968 |
| gaussian_nb | probabilistic | 0.8271 | 0.8020 | 0.4753 |
| logistic_regression | linear | 0.8023 | 0.8109 | 0.2878 |
| sgd_logistic | linear_sgd | 0.8021 | 0.8115 | 0.2886 |
