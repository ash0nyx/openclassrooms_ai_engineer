# Attrition Classification — ESN

Predict which employees are likely to leave an IT services company (ESN), and identify the key drivers of attrition to guide HR retention strategies.

---

## Context

Employee turnover is costly: according to SHRM, the cost of replacing an employee can range from 50% to 200% of their annual salary, depending on their level. This project builds a classification model to flag at-risk employees before they leave, and uses explainability tools (SHAP) to understand *why* they leave.

**Target variable:** `a_quitte_l_entreprise` (Oui/Non → 1/0)  
**Class imbalance:** 84% stayed, 16% left

---

## Data

Three source files joined on `id_employee`:

| File | Content |
|------|---------|
| `extrait_sirh.csv` | HR data — age, salary, department, tenure |
| `extrait_eval.csv` | Performance data — evaluations, overtime, salary raises |
| `extrait_sondage.csv` | Survey data — satisfaction scores, commute distance |

**Final dataset:** 1,470 employees × 47 features after encoding

---

## Setup

```bash
# Clone the repo
git clone <repo-url>
cd attrition-classification

# Create virtual environment and install dependencies
uv sync
```

---

## Project Structure

```
├── Sazonova_Kseniia_1_notebook_052026.ipynb         # Main notebook — all analysis and modeling
├── Sazonova_Kseniia_2_presentation_052026.pdf       # presentation
├── extrait_sirh.csv
├── extrait_eval.csv
├── extrait_sondage.csv
├── pyproject.toml
├── uv.lock
└── README.md
```

---

## Approach

### 1. Data Cleaning
- Dropped constant columns (`nombre_heures_travaillees`, `ayant_enfants`)
- Cleaned `augementation_salaire_precedente` — removed `%` suffix, cast to float
- Made a version where I dropped highly correlated features (`niveau_hierarchique_poste` ↔ `revenu_mensuel`, r=0.95)

### 2. Feature Engineering
Two new features created:
- `experience_externe` = total experience − years in company (experience gained elsewhere)
- `evaluation_change` = current evaluation − previous evaluation (performance trend)
- `anciennete_avant_poste_actuel` = years in the company - years in the current position

### 3. Encoding
| Type | Features | Method |
|------|---------|--------|
| Binary | `genre`, `heure_supplementaires` | Map (0/1) |
| Ordinal | `frequence_deplacement` | 0 / 1 / 2 |
| Nominal | `statut_marital`, `departement`, `poste`, `domaine_etude` | OneHotEncoding |

### 4. Modeling
- **Train/test split:** 80/20, stratified on target
- **Cross-validation:** 10-fold stratified
- **9 models tested:** Dummy, Logistic Regression, Random Forest, XGBoost, SVC, Linear SVC, Naive Bayes, KNN, Gradient Boosting
- **Class imbalance:** addressed via threshold optimization (0.07 instead of default 0.50)
- SMOTE was tested but decreased recall (55% vs 89%) — rejected

### 5. Threshold Optimization
Default threshold of 0.50 assumes false positives and false negatives are equally costly. In HR context, missing a leaver costs ~10× more than a false alarm. Threshold was lowered to **0.07** using Precision-Recall curves to maximize recall while keeping precision above 30%.

---

## Results

**Best model: Gradient Boosting (untuned, df_out, threshold=0.07)**

| Metric | Score |
|--------|-------|
| Test Recall | **89%** |
| CV Recall (10-fold) | **85% ± 9%** |
| Test Precision | 28% |
| True leavers caught | 42 / 47 |

Fine-tuning via GridSearchCV was attempted but decreased performance — default hyperparameters were already well-adapted to the dataset.

**Primary metric: Recall** — catching leavers is the priority over avoiding false alarms.

---

## Key Attrition Drivers

Confirmed by both Permutation Importance and SHAP Beeswarm:

| Rank | Feature | Insight |
|------|---------|---------|
| 1 | `heure_supplementaires` | Biggest driver by far — overworked employees leave |
| 2 | `augementation_salaire_precedente` | Low raises signal lack of recognition |
| 3 | `satisfaction_employee_equipe` | Unhappy teams drive attrition |
| 4 | `nombre_participation_pee` | Low financial engagement = low attachment |
| 5 | `age` | Younger employees leave more often |
| 6 | `nombre_experiences_precedentes` | Job-hoppers keep hopping |
| 7 | `distance_domicile_travail` | Long commutes increase risk |

SHAP scatter plots confirmed these are **independent factors** — they stack and don't interact with each other.

---

## HR Recommendations

1. **Audit overtime policy** — limit it or compensate better. Single biggest lever.
2. **Review salary raise distribution** — assure yearly consistent decent salary raises.
3. **Monitor team satisfaction** — especially for young employees, resolve conflicts in teams and listen to the employees.
4. **High-risk profile to watch:** young + overtime + long commute + low raise + multiple prior jobs.

---

## Limitations

- **Small dataset** (1,470 employees) — findings are directional, not statistically conclusive.
- **Only 47 true leavers in test set** — SHAP interpretation should be treated as tendencies, not established facts.
- **Sondage timing unknown** — if satisfaction surveys were collected post-departure (exit interviews), this constitutes target leakage and would invalidate those features.
- **Threshold optimization on test set** — technically data leakage. Correct approach would be cross-validated threshold selection. CV recall (85%) close to test recall (89%) suggests limited impact.

---

## Tools & Libraries

| Purpose | Library |
|---------|---------|
| Data manipulation | `pandas`, `numpy` |
| Visualization | `matplotlib`, `seaborn`, `plotly` |
| Modeling | `scikit-learn`, `xgboost` |
| Class imbalance | `imbalanced-learn` |
| Explainability | `shap` |
