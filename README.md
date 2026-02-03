# 📊 Customer Churn Prediction System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://pre-commit.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A production-grade machine learning system for predicting customer churn in the telecom industry. This project demonstrates end-to-end MLOps practices including data validation, experiment tracking, model serving, and interactive dashboards.

---

## 🎯 Business Problem

Customer churn (customers leaving for competitors) costs telecom companies billions annually. This system:

- **Predicts** which customers are likely to churn in the next billing cycle
- **Segments** customers using RFM analysis for targeted retention strategies
- **Explains** predictions using SHAP values for actionable insights

---

## 🏗️ Architecture
```
┌─────────────────────────────────────────────────────────────┐
│                      USER INTERFACES                         │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────┐  │
│  │  Streamlit  │    │   FastAPI   │    │    MLflow UI    │  │
│  │  Dashboard  │    │   REST API  │    │   Experiments   │  │
│  │    :8501    │    │    :8000    │    │      :5000      │  │
│  └──────┬──────┘    └──────┬──────┘    └─────────────────┘  │
│         └────────┬─────────┘                                 │
│                  ▼                                           │
│         ┌───────────────┐                                    │
│         │  ML Pipeline  │                                    │
│         └───────┬───────┘                                    │
│    ┌────────────┼────────────┐                               │
│    ▼            ▼            ▼                               │
│ ┌──────┐   ┌─────────┐   ┌──────────┐                       │
│ │ Data │──▶│Features │──▶│  Models  │                       │
│ │Valid.│   │Engineer.│   │Train/Pred│                       │
│ │ (GE) │   │  (RFM)  │   │(XGBoost) │                       │
│ └──────┘   └─────────┘   └──────────┘                       │
└─────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Docker & Docker Compose (for containerized deployment)
- Git

### Local Development Setup
```bash
# Clone the repository
git clone https://github.com/pranshu1921/AINE-AI-Predicting-Customer-Churn-Telecom.git
cd AINE-AI-Predicting-Customer-Churn-Telecom

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run tests
pytest
```

### Docker Deployment
```bash
# Build and start all services
docker-compose up --build

# Access the applications:
# - Dashboard: http://localhost:8501
# - API Docs:  http://localhost:8000/docs
# - MLflow:    http://localhost:5000
```

---

## 📁 Project Structure
```
customer-churn-prediction/
│
├── src/                          # Production source code
│   ├── data/                     # Data loading & validation
│   │   ├── ingestion.py          # Data loading utilities
│   │   └── validation.py         # Great Expectations checks
│   ├── features/                 # Feature engineering
│   │   └── engineering.py        # RFM & transformations
│   ├── models/                   # ML models
│   │   ├── train.py              # Training pipeline
│   │   └── predict.py            # Inference logic
│   ├── api/                      # FastAPI application
│   │   └── main.py               # API endpoints
│   └── dashboard/                # Streamlit application
│       └── app.py                # Dashboard UI
│
├── tests/                        # Test suite
├── notebooks/                    # Jupyter notebooks (EDA only)
│   └── archive/                  # Original project files
├── data/                         # Data files (DVC tracked)
│   ├── raw/                      # Original, immutable data
│   └── processed/                # Transformed features
├── models/                       # Saved model artifacts
├── great_expectations/           # Data validation config
│
├── docker-compose.yml            # Multi-container setup
├── Dockerfile                    # Container definition
├── pyproject.toml                # Project configuration
└── README.md                     # This file
```

---

## 🔧 Tech Stack

| Category | Tools |
|----------|-------|
| **Data Processing** | Pandas, NumPy |
| **Machine Learning** | Scikit-learn, XGBoost, SHAP |
| **Data Validation** | Great Expectations |
| **Experiment Tracking** | MLflow |
| **API** | FastAPI, Pydantic |
| **Dashboard** | Streamlit, Plotly |
| **Containerization** | Docker, Docker Compose |
| **Code Quality** | Ruff, Pre-commit, Pytest |
| **Data Versioning** | DVC |

---

## 📊 Dataset

The project uses the [Telco Customer Churn](https://www.kaggle.com/blastchar/telco-customer-churn) dataset containing:

- **7,043 customers** with 21 features
- **Target variable:** Churn (Yes/No)
- **Features:** Demographics, account info, services subscribed

---

## 🧪 Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | TBD |
| Precision | TBD |
| Recall | TBD |
| F1 Score | TBD |
| AUC-ROC | TBD |

*Metrics will be updated after model training.*

---

## 📈 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/predict` | POST | Single customer prediction |
| `/predict/batch` | POST | Batch predictions |
| `/segment` | POST | Customer segmentation |

See full API documentation at `http://localhost:8000/docs` when running.

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'feat: add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Original analysis: [Video Walkthrough](https://www.youtube.com/watch?v=OBWhPbwo734)
- Dataset: [IBM Sample Data Sets](https://www.ibm.com/communities/analytics/watson-analytics-blog/guide-to-sample-datasets/)