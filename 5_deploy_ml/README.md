# Employee Attrition Classification API

A production-deployed machine-learning API that predicts the risk of an employee
leaving the company. Built with FastAPI, backed by PostgreSQL for full traceability,
and deployed to Google Cloud Run through an automated CI/CD pipeline.

The model classifies each employee as **at risk of attrition** or **not**, based on
HR, evaluation, and survey data. It is tuned to prioritise **recall** — catching as
many at-risk employees as possible, since missing a leaver is far costlier than a
false alarm.

- **Live API:** https://attrition-api-716422774465.europe-west1.run.app
- **Interactive docs (Swagger):** https://attrition-api-716422774465.europe-west1.run.app/docs
- **Model card:** [docs/MODEL.md](docs/MODEL.md)

---

## Architecture

```mermaid
flowchart LR
    dev[Developer] -->|git push / merge| gh[GitHub]
    gh -->|GitHub Actions CI| tests[Run tests]
    gh -->|GitHub Actions CD - WIF| run[Cloud Run: attrition-api]

    user[API client] -->|HTTPS + API key| run
    run -->|download model at startup| gcs[(Cloud Storage<br/>model artifact)]
    run -->|read dataset / write prediction logs| sql[(Cloud SQL<br/>PostgreSQL)]
    run -->|secrets at deploy| sm[(Secret Manager<br/>DATABASE_URL, API_KEY)]
```

Everything the running service depends on lives in Google Cloud (`europe-west1`) —
the API on **Cloud Run**, the model in **Cloud Storage**, the data in **Cloud SQL**,
and credentials in **Secret Manager**. No developer machine is involved in production.

---

## Tech stack

| Area | Tools |
|------|-------|
| ML / data | scikit-learn (Gradient Boosting), pandas |
| API | FastAPI, Pydantic, Uvicorn |
| Database | PostgreSQL (Cloud SQL), SQLAlchemy 2.0, psycopg 3 |
| Tests | pytest, pytest-cov |
| CI/CD | GitHub Actions, Workload Identity Federation |
| Cloud | Cloud Run, Cloud SQL, Cloud Storage, Secret Manager |
| Tooling | uv (dependency management), Docker |

---

## Project structure

```
attrition-classification/
├── app/                     # FastAPI application
│   ├── main.py              # endpoints + startup (model load, prediction logging)
│   ├── model.py             # model loading (local or from GCS) and inference
│   ├── preprocessing.py     # cleaning, encoding, feature engineering
│   ├── schemas.py           # Pydantic request/response models
│   ├── auth.py              # API-key authentication
│   └── database.py          # SQLAlchemy engine + ORM models
├── scripts/
│   ├── prepare_data.py      # bronze CSVs -> silver dataset (join + preprocess)
│   ├── train.py             # train the Gradient Boosting model
│   └── create_db.py         # create tables + seed the dataset into PostgreSQL
├── tests/                   # unit + functional tests
├── docs/MODEL.md            # model technical documentation
├── .github/workflows/       # ci.yml (tests) + cd.yml (deploy)
├── Dockerfile
└── pyproject.toml
```

---

## Local development

**Prerequisites:** Python 3.12+, [uv](https://docs.astral.sh/uv/), and (for the
database) access to a PostgreSQL instance.

```bash
git clone https://github.com/ash0nyx/attrition-classification.git
cd attrition-classification
uv sync
```

Create a `.env` file at the project root (it is git-ignored):

```
DATABASE_URL=postgresql+psycopg://user:password@localhost:5432/attrition
# Optional locally; required in production:
# API_KEY=your-local-key
# MODEL_BUCKET=attrition-classif-models   # only if the model isn't present locally
```

Reproduce the data and model, then run the API:

```bash
uv run python scripts/prepare_data.py     # bronze -> silver
uv run python scripts/train.py            # train the model
uv run uvicorn app.main:app --reload      # serve at http://localhost:8000
```

Open http://localhost:8000/docs for the interactive Swagger UI.

---

## API usage

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| GET | `/health` | none | Liveness + whether the model is loaded |
| GET | `/model/info` | API key | Model type, threshold, feature count |
| POST | `/predict` | API key | Predict attrition risk for one employee (logged) |
| GET | `/predictions` | API key | Retrieve the most recent logged predictions |

Protected endpoints require an `X-API-Key` header. Example:

```bash
curl -X POST https://attrition-api-716422774465.europe-west1.run.app/predict \
  -H "X-API-Key: YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
        "age": 35, "genre": "F", "revenu_mensuel": 5000,
        "statut_marital": "Marié(e)", "departement": "Consulting",
        "poste": "Consultant", "nombre_experiences_precedentes": 3,
        "annee_experience_totale": 10, "annees_dans_l_entreprise": 5,
        "annees_dans_le_poste_actuel": 3,
        "satisfaction_employee_environnement": 3,
        "satisfaction_employee_nature_travail": 3,
        "satisfaction_employee_equipe": 2,
        "satisfaction_employee_equilibre_pro_perso": 3,
        "note_evaluation_actuelle": 3, "note_evaluation_precedente": 3,
        "niveau_hierarchique_poste": 2, "heure_supplementaires": "Non",
        "augementation_salaire_precedente": "11 %",
        "nombre_participation_pee": 1, "nb_formations_suivies": 2,
        "nombre_employee_sous_responsabilite": 0,
        "distance_domicile_travail": 10, "niveau_education": 3,
        "domaine_etude": "Infra & Cloud", "frequence_deplacement": "Occasionnel",
        "annees_depuis_la_derniere_promotion": 1, "annes_sous_responsable_actuel": 4
      }'
```

Response:

```json
{ "attrition_risk": true, "probability": 0.12, "threshold": 0.07 }
```

Input validation is enforced by Pydantic (ranges, allowed categories); invalid
payloads return `422`, missing/invalid API keys return `401`.

---

## Database & data flow

Data follows a **medallion** structure, then lands in PostgreSQL:

- **Bronze** — three raw CSV extracts (`sirh`, `eval`, `sondage`).
- **Silver** — `prepare_data.py` joins them on `id_employee`, cleans, encodes, and
  engineers features, producing the training dataset.
- **PostgreSQL (Cloud SQL)** — `create_db.py` creates the schema and loads the
  reference dataset. Every prediction is then logged back to the database.

**All model interactions pass through the database:** each `/predict` call stores its
input and output in `prediction_logs`, giving complete, auditable traceability. This
also supports **analytical needs** — the logged inputs/outputs can be queried to
monitor prediction volume, drift, and behaviour over time, and the `employees` table
supports offline model evaluation.

### Schema (UML / ER diagram)

```mermaid
erDiagram
    models ||--o{ prediction_logs : "produces"

    employees {
        int id_employee PK
        smallint age
        string genre
        int revenu_mensuel
        string statut_marital
        string departement
        string poste
        string heure_supplementaires
        bool a_quitte_l_entreprise "ground-truth label"
        string others "27 feature columns with CHECK constraints"
    }

    models {
        int id PK
        string name
        string version
        float threshold
        smallint n_features
        timestamptz trained_at
        timestamptz created_at
    }

    prediction_logs {
        int id PK
        int model_id FK
        jsonb input_data "the /predict payload"
        float probability
        bool attrition_risk
        float threshold
        timestamptz created_at
    }
```

- **`employees`** — the reference dataset in human-readable form (mirrors the API
  input) plus the ground-truth label. `CHECK` constraints enforce the same domain
  rules as the Pydantic validation.
- **`models`** — a registry of trained models (name, version, threshold, features).
- **`prediction_logs`** — every prediction's input (`JSONB`) and output, linked to the
  model that produced it via `model_id`.

Recreate and seed the database (with `DATABASE_URL` pointing at your instance):

```bash
uv run python scripts/create_db.py
```

### Analytical use

Storing every prediction makes the database a source for analytics, not just an
operational store. From the `prediction_logs` table one can:

- **Monitor usage** — prediction volume over time, per model version (`model_id`).
- **Track the risk distribution** — how `probability` / `attrition_risk` shift over
  time, to detect **model drift** (e.g. a sudden rise in flagged employees).
- **Audit decisions** — trace exactly which input produced which prediction, and when.
- **Feed retraining** — accumulate real-world inputs (and, once outcomes are known,
  labels) to improve future models.

The `employees` table supports **offline model evaluation** — recomputing metrics on
the stored reference dataset. A natural next step would be a lightweight **dashboard**
(e.g. Looker Studio or Metabase connected to Cloud SQL) surfacing daily prediction
counts, the score distribution, and the flagged-employee rate — the analytical layer
this schema is designed to enable.

---

## Authentication & security

- **API-key authentication** — the data endpoints require an `X-API-Key` header,
  validated against the `API_KEY` secret. `/health` stays open for Cloud Run health
  checks. Enforced whenever `API_KEY` is set (always, in production).
- **Secret management** — `DATABASE_URL` (with the database password) and `API_KEY`
  are stored in **Secret Manager** and injected as environment variables at deploy
  time. They are never committed to git (`.env` is git-ignored).
- **Keyless CI/CD** — GitHub authenticates to GCP via **Workload Identity Federation**
  (short-lived OIDC tokens), so no long-lived service-account key is stored anywhere.
  The federation is restricted to this repository only.
- **Least privilege** — the deploy uses a dedicated `github-deployer` service account,
  separate from the runtime service account.
- **Database integrity** — a dedicated, non-superuser database role is used; `CHECK`
  constraints and `NOT NULL` rules protect against malformed data.

---

## Testing

```bash
uv run pytest                                  # run all tests (coverage prints)
uv run pytest --cov=app --cov-report=html      # detailed HTML report in htmlcov/
```

The suite covers the preprocessing pipeline, Pydantic validation, the model, the
database, and all API endpoints — including error scenarios (422 validation, 401 auth,
503 model-not-loaded, 500 failures). Tests use an in-memory SQLite database and a
mocked model, so they run anywhere (including CI) with no external dependencies.

**42 tests · ~90% line coverage.**

| Module | Coverage |
|--------|----------|
| `app/auth.py` | 100% |
| `app/schemas.py` | 100% |
| `app/preprocessing.py` | 100% |
| `app/database.py` | 95% |
| `app/model.py` | 79% |
| `app/main.py` | 76% |
| **Total** | **90%** |

The uncovered lines are the startup/lifespan code and the Cloud Storage download path,
which aren't exercised in the isolated test environment. Generate the full HTML report
with `uv run pytest --cov=app --cov-report=html` (then open `htmlcov/index.html`).

![Test coverage report](docs/coverage.png)

---

## CI/CD

Two GitHub Actions workflows:

- **CI (`ci.yml`)** — on every push and pull request: installs dependencies and runs
  the full test suite.
- **CD (`cd.yml`)** — on merge to `main`: re-runs the tests and, only if they pass,
  builds the image (Cloud Build) and deploys to Cloud Run. Authentication is keyless
  via Workload Identity Federation.

This maps to the environments as follows: pull requests and branches are the
**test/integration** stage (CI), and `main` is the **production** stage (CD → Cloud
Run).

---

## Deployment

Deployment is automated via the CD workflow. To deploy manually (e.g. the first time):

```bash
gcloud run deploy attrition-api \
  --source . \
  --region europe-west1 \
  --allow-unauthenticated \
  --add-cloudsql-instances attrition-classif:europe-west1:attrition-pg \
  --set-env-vars MODEL_BUCKET=attrition-classif-models \
  --set-secrets DATABASE_URL=DATABASE_URL:latest,API_KEY=API_KEY:latest
```

The image is built in the cloud by Cloud Build (no local Docker required). At startup
the container downloads the model from Cloud Storage and connects to Cloud SQL over a
secure socket.

---

## Key technical choices

- **Recall-first threshold (0.07)** — on an imbalanced dataset (84/16), a false
  negative (missing a leaver) is costlier than a false positive, so the decision
  threshold is lowered from 0.5 to 0.07. See [docs/MODEL.md](docs/MODEL.md).
- **Cloud SQL over BigQuery** — the project needs transactional storage of the dataset
  and per-request prediction logs; a relational database fits this better than a data
  warehouse.
- **Model in Cloud Storage, not git** — model artifacts don't belong in version
  control; the API downloads the model from GCS at startup.
- **API-key auth for a POC** — simple and sufficient to protect the endpoints; in
  production this could be upgraded to OAuth / IAM-based access.

---

## Maintenance & updates

- **Retraining** — rerun `scripts/prepare_data.py` and `scripts/train.py`, then upload
  the new artifact to the model bucket. Register the new version in the `models` table
  via `create_db.py`. Cloud Run picks up the new model on its next restart/deploy.
- **Deploying changes** — merge to `main`; the CD pipeline redeploys automatically.
- **Monitoring** — inspect `prediction_logs` to track prediction volume and detect
  drift; consider periodic retraining as new labelled data accumulates.

---

## License

MIT
