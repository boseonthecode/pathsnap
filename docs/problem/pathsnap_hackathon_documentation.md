# PathSnap: Distributed Map-Matching & Route Reconstruction
## Complete Hackathon Documentation

This document is a comprehensive guide consolidating all the requirements, guidelines, and policies for the PathSnap hackathon. It is compiled from the following reference files:
*   "Main Guidelines.pdf"
*   "Dataset Exploration.pdf"
*   "Models and Training Guidelines.pdf"
*   "Deliverables.pdf"
*   "Assessment and Plagiarism Policy.pdf"

---

## 1. Main Guidelines
*(Reference: "Main Guidelines.pdf")*

### 1.1 Challenge Overview
GPS tracking systems often produce inaccurate location measurements due to urban buildings, signal interference, multipath effects, GPS dropouts, and low sampling frequencies. Consequently, a vehicle travelling along a road may appear to move onto a parallel street, jump across buildings, or deviate from its actual route. 

Your task is to develop **PathSnap**, a distributed machine learning system that reconstructs realistic vehicle trajectories by matching noisy GPS observations to a road network. The challenge involves developing an end-to-end solution, from understanding and processing raw GPS data to training machine learning models, reconstructing routes, and deploying the final system.

### 1.2 Two-Laptop Deployment Architecture
| Laptop | Component | Responsibilities |
| :--- | :--- | :--- |
| **Laptop 1** | Model and API | Hosts the trained classifier, HMM/Viterbi implementation, road-network information, and server-side evaluation process. |
| **Laptop 2** | User Interface | Displays the ten evaluation-trip options, sends the selected trip to the API, and displays the returned route and results. |

### 1.3 Expected Work Process
The overall challenge should be approached in the following stages:
1. Understand the problem.
2. Explore and understand the raw dataset.
3. Develop an independent data-processing and feature-engineering approach.
4. Train and compare multiple machine learning models.
5. Select the best-performing model based on experimental evidence.
6. Perform Bayesian hyperparameter optimisation.
7. Implement HMM and Viterbi-based route reconstruction.
8. Export the trained model and required artifacts.
9. Develop the FastAPI backend.
10. Develop the evaluator-facing UI.
11. Integrate and test the complete system.
12. Prepare the final technical report.

---

## 2. Dataset Description and Exploration
*(Reference: "Dataset Exploration.pdf")*

### 2.1 Dataset Overview
The dataset contains GPS trajectory information and associated road-network information. A trajectory records the observed movement of a vehicle over time, while the road network represents the geographical structure against which GPS observations must be matched. The raw data package is organised by trip. 

### 2.2 Raw File Types
| Raw file | General meaning | What these files have |
| :--- | :--- | :--- |
| `.track` | GPS observations associated with a trajectory. | 3 whitespace-separated columns without headers: longitude, latitude, and timestamp in seconds. Nominally 1 Hz: about 97% of steps are exactly 1 s apart. |
| `.nodes` | Geographical nodes in the road network. | 2 whitespace-separated columns without headers: longitude and latitude (0-based row index). |
| `.arcs` | Connections between road-network nodes representing road segments. | 2 integer columns without headers: `from_node_id` and `to_node_id` (0-based row index serves as `arc_id`). |
| `.route` | Reference route information, where supplied. | 1 integer column without headers: chronological sequence of ground-truth `arc_id` values traversed by the vehicle. |

### 2.3 Initial Dataset Exploration & Geographical Considerations
Before implementing a solution, teams must conduct a structured exploratory analysis identifying trip counts, GPS observation counts, file availability, data types, missing observations, timestamp ordering, and road-network connectivity.

**Geographical Considerations:**
*   Longitude and latitude are angular coordinates. Ordinary Euclidean distance in degrees is generally not an appropriate measure of physical distance.
*   Choose and justify an appropriate projected coordinate system or geodesic distance calculation.
*   Check coordinate order carefully and inspect coordinate plausibility.

### 2.4 Dataset Partitions and Leakage Prevention
| Partition | Number of trips | Use |
| :--- | :--- | :--- |
| **Training** | 80 | Training, development, cross-validation, and model optimisation. |
| **Student testing** | 10 | Development-stage evaluation and model comparison. |
| **Evaluator testing**| 10 | Final evaluation only. |

**Important Rules:**
*   No observations from the same trip may be distributed across different partitions.
*   Evaluator-testing data/labels must not be used for model training, feature selection, or choosing algorithms/hyperparameters.
*   Do not manually alter predictions or hardcode expected answers for evaluation trips.

---

## 3. Models and Training Guidelines
*(Reference: "Models and Training Guidelines.pdf")*

### 3.1 Independent Model Selection & Algorithm Families
Teams must independently explore different machine learning algorithms and train/evaluate at least **FIVE** different models before selecting their final model. 

| Algorithm family | Possible approaches | What to consider |
| :--- | :--- | :--- |
| **Linear and probabilistic** | Logistic Regression, Ridge Classifier, SGD Classifier, Gaussian Naive Bayes | Decision boundaries, assumptions, and probability estimates. |
| **Instance-based** | KNN, Distance-Weighted KNN, Radius Neighbours | Similarity, distance metrics, and scaling. |
| **Tree-based** | Decision Tree, Extra Trees | Nonlinear decision rules and overfitting. |
| **Ensemble** | Random Forest, Extra Trees Classifier, Bagging | Stability and generalisation. |
| **Boosting** | Gradient Boosting, HistGradientBoosting, XGBoost, LightGBM, CatBoost, AdaBoost | Sequential error correction and computational efficiency. |
| **Kernel-based** | Linear SVM, RBF SVM, Polynomial SVM | Kernel suitability and scaling feature. |
| **Neural networks** | Shallow MLP, Deep MLP | Nonlinear representations, regularisation, and training stability. |
| **Unsupervised/clustering**| K-Means, Mini-Batch K-Means, Agglomerative Clustering, DBSCAN | Cluster interpretation and conversion into meaningful predictions. |

### 3.2 Evaluation Metrics
Report the following where applicable:
*   Accuracy, Precision, Recall, F1-score, Confusion Matrix.
*   Log Loss, Candidate Miss Rate, Route Continuity.
*   Observation-level Top-1 Accuracy (select the candidate with the highest predicted score vs. reference segment).
*   Training Time, Inference Time.

### 3.3 Bayesian Hyperparameter Optimisation & HMM Decoding
*   **Bayesian Optimisation:** After initial comparison, select a promising model and perform hyperparameter optimisation using a Bayesian search strategy. Define the search space, specify the objective, use group-aware CV on the 80 training trips, and compare the tuned model with the untuned model.
*   **HMM and Viterbi:** Route reconstruction must use sequential information (HMM) and road-network connectivity to identify a coherent route. Compare classifier-only predictions with classifier-plus-HMM predictions. Handle missing candidates and GPS gaps appropriately.

---

## 4. Technical Report & Submission Guidelines
*(Reference: "Deliverables.pdf")*

### 4.1 Deliverable 1: GitHub Repository
The team's complete project source code must be made public on GitHub. It must contain:
*   Data processing & feature engineering source code.
*   Model training notebooks/scripts for every model.
*   Model evaluation and visualisations.
*   Bayesian hyperparameter optimisation code and results.
*   HMM and Viterbi source code.
*   API (backend) and UI (frontend) implementation.
*   Deployment instructions, required artifacts (weights), and a README. 

### 4.2 Deliverable 2: Technical Report (PDF)
A consolidated PDF report detailing the methodology and results. It must include:
1.  **Dataset Understanding and EDA:** Explanation of preprocessing, extracted features, missing-value analysis, and at least one Folium map visualisation.
2.  **Model Experimentation:** A consolidated comparison table of at least 5 models (Macro F1, Top-1 Accuracy, Route Continuity, Log Loss, Inference Time).
3.  **Model Details:** For *every* model trained, answer specific questions regarding its suitability, strengths, limitations, hyperparameters, loss function, and metrics.
4.  **Model Visualisations:** CV F1-score comparisons, Confusion matrices, and training/validation loss plots (generated from actual experimental results). 
5.  **Bayesian Hyperparameter Optimisation:** Documentation of the search space, trials, and a comparison showing if the tuned model outperformed the untuned one.
6.  **HMM and Viterbi Route Reconstruction:** Explanation of hidden states, emission/transition probabilities, handling missing data, and performance comparisons.
7.  **Geographical Visualisation (Folium):** Mandatory interactive maps showing original GPS observations, road-network geometry, reference route, and predicted route.
8.  **API, UI and Deployment:** Screenshots of the working UI and API responses.

---

## 5. Assessment and Academic Integrity Policy
*(Reference: "Assessment and Plagiarism Policy.pdf")*

### 5.1 Purpose and Assessment Focus
Assessment considers correctness, completeness, technical understanding, experimental rigour, reproducibility, and the team's ability to explain its work. A high score on one metric does not compensate for missing or invalid parts of the system.

### 5.2 Independent Work & Similarity Checks
Each team must independently implement, train, evaluate, and select its own models. Sharing code, weights, datasets, or reports between teams is strictly prohibited. Organisers will compare code, repository history, architectures, configurations, reports, and visualisations across teams. Similarity may trigger an investigation.

### 5.3 AI Tool Use and Report Authenticity
*   Submitting AI-generated analysis for unperformed experiments, fabricating metrics/graphs, or misrepresenting AI-generated work as independent experimentation is prohibited.
*   Every team member must be prepared to defend and explain all technical claims, tables, graphs, and design choices.

### 5.4 Viva, Demonstration & Data Leakage
During evaluation, organisers may ask team members to explain data preprocessing, model selection, Bayesian optimization, HMM/Viterbi implementation, API/UI structure, and experimental results.
*   **Data Leakage:** Evaluator-testing data must remain untouched during development. Hardcoded predictions are strictly prohibited and repositories will be examined for them.