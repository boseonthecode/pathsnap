# Model selection note

Selection metric: mean trip-grouped CV Macro F1, with CV Top-1 accuracy as the tie-breaker. Training accuracy was not used.

**Selected model:** `extra_trees`

| Model | Family | CV Macro F1 | CV Top-1 | CV Log Loss |
| :--- | :--- | ---: | ---: | ---: |
| extra_trees | ensemble | 0.8858 | 0.8137 | 0.1773 |
| mlp | neural | 0.8850 | 0.8144 | 0.1603 |
| knn | instance_based | 0.8775 | 0.8062 | 0.4908 |
| lightgbm | boosting | 0.8647 | 0.8206 | 0.2119 |
| logistic_regression | linear | 0.8023 | 0.8109 | 0.2878 |
