# AI-Powered Business Experimentation and Decision Simulator
### Project Documentation

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Problem Statement](#problem-statement)
3. [System Architecture](#system-architecture)
4. [Data Science Components](#data-science-components)
5. [GenAI Layer](#genai-layer)
6. [Datasets & Data Strategy](#datasets--data-strategy)
7. [Tech Stack](#tech-stack)
8. [Project Structure](#project-structure)
9. [Module Details](#module-details)
10. [API Design](#api-design)
11. [Example Use Cases](#example-use-cases)
12. [Evaluation Metrics](#evaluation-metrics)
13. [Roadmap](#roadmap)

---

## Project Overview

The **AI-Powered Business Experimentation and Decision Simulator** is an intelligent decision-support platform that enables executives and analysts to simulate the future impact of business decisions — before committing to them.

Unlike traditional BI dashboards that look backward at historical data, this system looks **forward** using causal inference, uplift modeling, and GenAI-powered narrative generation to answer the most critical question in business:

> *"What happens if we do X?"*

---

## Problem Statement

Companies make high-stakes decisions with incomplete foresight:

| Decision | Risk Without Simulation |
|---|---|
| Increase product price by 8% | Unknown revenue vs. churn tradeoff |
| Launch a marketing campaign | Unclear incremental lift vs. organic growth |
| Enter a new geographic region | Opaque demand, competition, cost dynamics |
| Discontinue a product line | Hidden cannibalization effects |

Most dashboards show **what happened**. This system predicts **what will happen**.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    USER INTERFACE (Streamlit / FastAPI UI)  │
│         Executive Query: "What if we raise prices 8%?"     │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                   GenAI Orchestration Layer                 │
│         (LangChain / Claude API / GPT-4)                   │
│   - Parse intent  - Route to model  - Generate narrative   │
└────────┬──────────────┬──────────────────┬─────────────────┘
         │              │                  │
         ▼              ▼                  ▼
┌──────────────┐ ┌────────────┐ ┌──────────────────────┐
│ Causal Impact│ │  Uplift    │ │   Forecasting &      │
│    Model     │ │   Model    │ │ Scenario Simulation  │
│  (CausalPy / │ │ (causalml) │ │  (Prophet / ARIMA /  │
│   DoWhy)     │ │            │ │   XGBoost)           │
└──────────────┘ └────────────┘ └──────────────────────┘
         │              │                  │
         └──────────────┴──────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    Data Layer                               │
│   Feature Store | Historical Sales | CRM | Market Data     │
└─────────────────────────────────────────────────────────────┘
```

---

## Data Science Components

### 1. Causal Impact Modeling

**Goal:** Isolate the true effect of a business intervention (e.g., a price change or campaign) from confounding factors like seasonality and market trends.

**Approach:**
- Use **Google's CausalImpact** (R) or **CausalPy** (Python) to model counterfactual scenarios.
- Build a synthetic control group from untreated regions/products.
- Estimate the **Average Treatment Effect (ATE)** of the decision.

**Key Output:**
```
Price Increase of 8% in Mumbai (June–August):
  - Estimated Revenue Impact: -₹2.3 Cr (95% CI: -₹1.8 Cr to -₹2.9 Cr)
  - Probability of Positive Impact: 12%
```

**Libraries:** `causalpy`, `dowhy`, `statsmodels`

---

### 2. Uplift Modeling

**Goal:** Predict the **incremental** impact of a business action on individual customers or segments — not just the average.

**Approach:**
- Train a Two-Model Uplift approach: `Treatment Model - Control Model`
- Or use **Meta-Learners** (S-Learner, T-Learner, X-Learner)
- Segment customers by uplift quintile: Persuadables, Sure Things, Lost Causes, Do-Not-Disturbs

**Example:**
```
Campaign on Segment "Mid-Value Mumbai Customers":
  - Uplift Score: +0.14 (High Persuadable)
  - Recommended Action: Target this segment
  - Expected Incremental Conversions: 3,400
```

**Libraries:** `causalml`, `pylift`, `scikit-uplift`

---

### 3. Regression Analysis

**Goal:** Quantify relationships between business levers (price, ad spend, discount) and outcomes (revenue, churn, volume).

**Models Used:**
- **Price Elasticity Model:** Log-log regression → `ln(demand) = α + β·ln(price) + ε`
- **Marketing Mix Model (MMM):** Multi-variate regression with diminishing returns (Adstock transformation)
- **Churn Regression:** Logistic regression / Gradient Boosting on customer features

**Key Metric:** Price Elasticity of Demand (PED)
```
PED = % Change in Quantity Demanded / % Change in Price
PED = -1.4 → Elastic (raising price will reduce total revenue)
```

**Libraries:** `statsmodels`, `sklearn`, `pymc-marketing` (for MMM)

---

### 4. Forecasting Models

**Goal:** Project baseline KPIs (revenue, volume, churn) over a 3–12 month horizon without any intervention.

**Models:**
| Model | Use Case |
|---|---|
| Prophet | Trend + seasonality (weekly/monthly/holiday effects) |
| ARIMA / SARIMA | Stationary time series |
| XGBoost / LightGBM | Non-linear patterns with external regressors |
| LSTM | Long-range dependencies in sequential data |

**Output:**
```
Baseline Revenue Forecast (Next 6 months, Mumbai):
  Month 1: ₹18.2 Cr  |  Month 2: ₹17.8 Cr  |  ...
  (without any price change)
```

**Libraries:** `prophet`, `statsforecast`, `neuralforecast`, `skforecast`

---

### 5. Scenario Simulation Engine

**Goal:** Combine causal + forecast models to simulate future states under different decision scenarios.

**Architecture:**
```python
def simulate_scenario(decision: dict, region: str, horizon: int):
    baseline = forecast_model.predict(region, horizon)
    causal_effect = causal_model.estimate_effect(decision)
    uplift = uplift_model.score(region_customers)
    
    simulated = baseline + causal_effect + uplift
    return SimulationResult(
        revenue=simulated.revenue,
        churn=simulated.churn,
        profit=simulated.profit,
        confidence_interval=simulated.ci_95
    )
```

**Scenarios Supported:**
- Pricing change (increase / decrease / tiered)
- Campaign launch (targeted / broad)
- Regional expansion / exit
- Product launch / discontinuation
- Discount strategy

---

## GenAI Layer

### Natural Language Query Interface

Executives interact in plain English:

> *"What happens if we increase prices by 8% in Mumbai starting next quarter?"*

**Processing Pipeline:**
```
User Query
    │
    ▼
Intent Parsing (LLM)
    │  Extracts: {action: "price_increase", magnitude: 0.08,
    │             region: "Mumbai", timeline: "Q3 2025"}
    ▼
Parameter Validation & Enrichment
    │  Adds: {product_lines: [...], customer_segments: [...]}
    ▼
Simulation Engine (DS Models)
    │  Returns: {revenue_delta: -2.3Cr, churn_delta: +4.2%, ...}
    ▼
Narrative Generation (LLM)
    │
    ▼
Strategic Recommendation Report
```

### Sample GenAI Output

```markdown
**Strategic Simulation Report — Price Increase 8%, Mumbai**
Generated: June 2025

**Executive Summary**
Raising prices by 8% in the Mumbai region is projected to decrease
revenue by ₹2.3 Cr over 6 months, driven primarily by volume loss
in the mid-value customer segment (PED = -1.6).

**Key Predictions**
- Revenue Impact: -₹2.3 Cr (–12.6%)
- Customer Churn: +4.2% uplift in at-risk customers
- Profit Change: -₹0.8 Cr (margin compression from lost volume)

**Recommended Action**
Consider a selective 5% increase limited to premium SKUs only.
This is predicted to yield +₹0.6 Cr revenue with minimal churn risk.

**Confidence Level: Medium (CI: ±₹0.6 Cr)**
```

### LLM Integration

```python
from anthropic import Anthropic

client = Anthropic()

def generate_recommendation(simulation_result: dict, query: str) -> str:
    prompt = f"""
    You are a senior business strategy consultant.
    
    A business executive asked: "{query}"
    
    Simulation results:
    {simulation_result}
    
    Provide a structured strategic recommendation with:
    1. Executive Summary (2-3 sentences)
    2. Key Predictions (bullet points)
    3. Risk Factors
    4. Recommended Action
    5. Alternative Scenarios to Consider
    """
    
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text
```

---

## Datasets & Data Strategy

### Recommended Free Datasets

#### Primary Datasets

| Dataset | Source | Use Case | Link |
|---|---|---|---|
| **Instacart Market Basket** | Kaggle | Customer purchase behavior, price sensitivity | [Kaggle](https://www.kaggle.com/c/instacart-market-basket-analysis) |
| **Olist E-Commerce (Brazil)** | Kaggle | Regional sales, pricing, reviews, logistics | [Kaggle](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) |
| **Walmart Sales Forecasting** | Kaggle | Store-level sales, holiday effects, markdown impact | [Kaggle](https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting) |
| **UCI Online Retail Dataset** | UCI ML Repo | Transactional data, invoice-level pricing | [UCI](https://archive.ics.uci.edu/dataset/352/online+retail) |
| **M5 Forecasting Dataset** | Kaggle | Hierarchical sales forecasting (Walmart) | [Kaggle](https://www.kaggle.com/c/m5-forecasting-accuracy) |
| **Telco Customer Churn** | Kaggle/IBM | Churn modeling for uplift | [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) |

#### Causal Inference Specific

| Dataset | Source | Use Case |
|---|---|---|
| **LaLonde Job Training** | `dowhy` / built-in | Classic causal effect estimation benchmark |
| **IBM Causal Inference Benchmark** | `causalml` | Uplift model benchmarking |
| **GiveDirectly Cash Transfer Data** | Harvard Dataverse | Treatment effect in development economics |

---

### ⚠ Synthetic Data Strategy

For a **production-grade demo**, synthetic data is strongly recommended for the following modules:

#### Why Synthetic Data?

Real business datasets with price experiments, A/B test results, and regional breakdowns are **almost never public**. Synthetic data lets you:
- Control ground truth (you know the true causal effect)
- Validate your models rigorously
- Customize to Indian market context (Mumbai, Delhi, Bangalore regions)
- Demonstrate domain expertise

#### Synthetic Data Generation Plan

**1. Sales & Pricing Data**
```python
import numpy as np
import pandas as pd

def generate_sales_data(n_months=36, regions=["Mumbai", "Delhi", "Bangalore"]):
    data = []
    for region in regions:
        base_demand = {"Mumbai": 5000, "Delhi": 4200, "Bangalore": 3800}[region]
        
        for month in range(n_months):
            price = np.random.normal(500, 50)
            price_elasticity = -1.4  # known ground truth
            seasonality = 1 + 0.15 * np.sin(2 * np.pi * month / 12)
            noise = np.random.normal(0, 0.05)
            
            demand = base_demand * (price / 500) ** price_elasticity * seasonality * (1 + noise)
            revenue = demand * price
            
            data.append({
                "month": month, "region": region,
                "price": price, "demand": demand, "revenue": revenue
            })
    return pd.DataFrame(data)
```

**2. Customer-Level Data (for Uplift Modeling)**
```python
def generate_customer_data(n_customers=50000):
    df = pd.DataFrame({
        "customer_id": range(n_customers),
        "region": np.random.choice(["Mumbai", "Delhi", "Bangalore"], n_customers),
        "tenure_months": np.random.randint(1, 60, n_customers),
        "avg_order_value": np.random.lognormal(6.2, 0.5, n_customers),
        "purchase_frequency": np.random.poisson(3, n_customers),
        "is_premium": np.random.binomial(1, 0.3, n_customers),
        # Treatment assignment (e.g., received price increase)
        "treated": np.random.binomial(1, 0.5, n_customers),
    })
    # Generate outcome with known uplift
    base_churn_prob = 0.08
    treatment_effect = 0.04  # price increase raises churn by 4%
    df["churned"] = np.random.binomial(
        1, base_churn_prob + df["treated"] * treatment_effect
    )
    return df
```

**3. Recommended Synthetic Data Libraries**

| Library | Purpose |
|---|---|
| `faker` | Generate realistic customer profiles, company names |
| `sdv` (Synthetic Data Vault) | Generate synthetic tabular data preserving statistical properties |
| `gretel-synthetics` | Privacy-safe synthetic data generation |
| `ydata-synthetic` | GAN-based time series and tabular synthesis |

---

## Tech Stack

```
Layer               | Technologies
--------------------|--------------------------------------------------
Data Processing     | Python, Pandas, NumPy, Polars
Causal Inference    | DoWhy, CausalPy, CausalML, EconML
Forecasting         | Prophet, StatsForecast, NeuralForecast
ML Models           | Scikit-learn, XGBoost, LightGBM
GenAI / LLM         | Anthropic Claude API / OpenAI GPT-4, LangChain
API Backend         | FastAPI, Pydantic
Frontend / UI       | Streamlit (MVP) → React (Production)
Database            | PostgreSQL + Redis (caching simulations)
Experiment Tracking | MLflow / Weights & Biases
Visualization       | Plotly, Matplotlib, Seaborn
Deployment          | Docker, AWS / GCP, GitHub Actions (CI/CD)
```

---

## Project Structure

```
ai-business-simulator/
│
├── data/
│   ├── raw/                    # Downloaded public datasets
│   ├── synthetic/              # Generated synthetic data
│   └── processed/              # Feature-engineered datasets
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_price_elasticity.ipynb
│   ├── 03_causal_impact.ipynb
│   ├── 04_uplift_modeling.ipynb
│   ├── 05_forecasting.ipynb
│   └── 06_scenario_simulation.ipynb
│
├── src/
│   ├── data/
│   │   ├── synthetic_generator.py
│   │   └── feature_engineering.py
│   │
│   ├── models/
│   │   ├── causal_impact.py
│   │   ├── uplift_model.py
│   │   ├── price_elasticity.py
│   │   ├── forecasting.py
│   │   └── scenario_simulator.py
│   │
│   ├── genai/
│   │   ├── intent_parser.py
│   │   ├── recommendation_engine.py
│   │   └── prompts.py
│   │
│   └── api/
│       ├── main.py             # FastAPI app
│       ├── routes/
│       │   ├── simulate.py
│       │   └── query.py
│       └── schemas.py
│
├── app/
│   └── streamlit_app.py        # Interactive UI
│
├── tests/
│   ├── test_causal_model.py
│   ├── test_uplift_model.py
│   └── test_simulation.py
│
├── mlflow/                     # Model registry and experiment tracking
├── docker/
│   └── Dockerfile
├── requirements.txt
├── .env.example
└── README.md
```

---

## Module Details

### `scenario_simulator.py`

The core orchestration module that connects all DS components.

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class SimulationRequest:
    action: str              # "price_increase", "campaign_launch", etc.
    magnitude: float         # e.g., 0.08 for 8% price increase
    region: str              # e.g., "Mumbai"
    product_line: str        # e.g., "Electronics"
    horizon_months: int      # e.g., 6
    target_segment: Optional[str] = None

@dataclass
class SimulationResult:
    revenue_delta: float
    revenue_delta_pct: float
    customer_churn_delta: float
    profit_delta: float
    confidence_interval_lower: float
    confidence_interval_upper: float
    recommendation: str
    risk_level: str          # "Low", "Medium", "High"

class ScenarioSimulator:
    def __init__(self, causal_model, uplift_model, forecast_model, llm_client):
        self.causal_model = causal_model
        self.uplift_model = uplift_model
        self.forecast_model = forecast_model
        self.llm_client = llm_client

    def simulate(self, request: SimulationRequest) -> SimulationResult:
        baseline = self.forecast_model.predict(
            region=request.region, horizon=request.horizon_months
        )
        causal_effect = self.causal_model.estimate_effect(
            treatment=request.action, magnitude=request.magnitude
        )
        uplift_scores = self.uplift_model.score(
            region=request.region, segment=request.target_segment
        )
        # Combine outputs
        ...
        return SimulationResult(...)
```

---

## API Design

### POST `/api/v1/simulate`

**Request:**
```json
{
  "action": "price_increase",
  "magnitude": 0.08,
  "region": "Mumbai",
  "product_line": "Electronics",
  "horizon_months": 6
}
```

**Response:**
```json
{
  "revenue_delta": -23000000,
  "revenue_delta_pct": -12.6,
  "customer_churn_delta": 4.2,
  "profit_delta": -8000000,
  "confidence_interval": [-29000000, -17000000],
  "risk_level": "High",
  "recommendation": "Consider limiting the price increase to 3-4%...",
  "alternative_scenarios": [
    {"action": "price_increase", "magnitude": 0.03, "predicted_revenue_delta": 2000000}
  ]
}
```

### POST `/api/v1/query`

**Request:**
```json
{
  "query": "What happens if we increase prices by 8% in Mumbai next quarter?"
}
```

**Response:**
```json
{
  "parsed_intent": {...},
  "simulation_result": {...},
  "narrative_recommendation": "Based on historical price elasticity..."
}
```

---

## Example Use Cases

### Use Case 1: Price Optimization

**Query:** *"Should we increase prices by 5% or 10% in Bangalore?"*

**System Output:**
- 5% increase → Revenue +₹0.4 Cr, Churn +1.8% → **Recommended**
- 10% increase → Revenue -₹1.1 Cr, Churn +5.6% → **Not Recommended**

### Use Case 2: Campaign ROI Prediction

**Query:** *"We plan to run a ₹50L campaign in Delhi. What's the expected uplift?"*

**System Output:**
- Expected incremental revenue: ₹1.8 Cr
- ROI: 3.6x
- Top persuadable segments: Mid-value, 25–35 age group

### Use Case 3: Regional Expansion

**Query:** *"Is it profitable to enter the Pune market in Q3?"*

**System Output:**
- Projected Year 1 Revenue: ₹6.2 Cr
- Break-even: Month 9
- Risk: Medium (low brand awareness, 3 established competitors)

---

## Evaluation Metrics

### Model Performance

| Model | Metric | Target |
|---|---|---|
| Causal Impact | MAPE on holdout periods | < 10% |
| Uplift Model | Qini Coefficient | > 0.15 |
| Price Elasticity | R² | > 0.80 |
| Forecasting | MASE | < 1.0 |
| Churn Model | AUC-ROC | > 0.78 |

### Business Metrics

- Simulation Accuracy: Backtested vs. actual outcomes on historical experiments
- Executive Adoption Rate: % of decisions aided by simulator
- Decision Lead Time: Reduction in time from question to strategic recommendation

---

## Roadmap

### Phase 1 — MVP (Weeks 1–4)
- [ ] Synthetic data generation pipeline
- [ ] Price elasticity regression model
- [ ] Basic Prophet forecasting
- [ ] Streamlit UI with single scenario simulation

### Phase 2 — Core DS (Weeks 5–8)
- [ ] Causal Impact model (CausalPy / DoWhy)
- [ ] Uplift model with customer segmentation
- [ ] Scenario simulation engine
- [ ] MLflow experiment tracking

### Phase 3 — GenAI Integration (Weeks 9–12)
- [ ] LLM intent parsing (Claude / GPT-4)
- [ ] Narrative recommendation generation
- [ ] Multi-scenario comparison
- [ ] FastAPI backend

### Phase 4 — Production (Weeks 13–16)
- [ ] Dockerized deployment
- [ ] PostgreSQL + Redis integration
- [ ] A/B testing framework for model validation
- [ ] Business report export (PDF)

---

## Why This Project Stands Out

| Dimension | Typical Fresher Project | This Project |
|---|---|---|
| Methodology | Descriptive / Predictive | Causal + Prescriptive |
| Business Relevance | Generic | Consulting-grade |
| AI Integration | Dashboard only | LLM + Simulation |
| Novelty | High competition | Very few build causal systems |
| Interview Value | "I built a churn model" | "I built a business decision simulator" |

---

*Documentation Version: 1.0 | Last Updated: June 2025*
