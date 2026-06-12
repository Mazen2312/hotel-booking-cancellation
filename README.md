# 🏨 Hotel Booking Cancellation Prediction

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6-orange?logo=scikit-learn)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.1-red)](https://xgboost.readthedocs.io)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.6-green)](https://lightgbm.readthedocs.io)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.41-FF4B4B?logo=streamlit)](https://streamlit.io)
[![Plotly](https://img.shields.io/badge/Plotly-6.5-3D5AFE?logo=plotly)](https://plotly.com)

---

## 📋 Overview

End-to-end machine learning pipeline that predicts whether a hotel booking will be canceled. Built from raw Kaggle data through cleaning, feature engineering, preprocessing, **6-model comparison with 5-fold cross-validation**, hyperparameter tuning (GridSearchCV), and deployment as an **interactive Streamlit dashboard**.

**Key highlights:**
- ✅ **86,494 clean rows** processed from 119,390 raw — missing values handled per mindmap rules
- ✅ **70/30 stratified train/test split** — models evaluated on truly unseen data
- ✅ **6 models compared** — XGBoost, LightGBM, Random Forest, Logistic Regression, KNN, Decision Tree
- ✅ **Impute After Split** — all imputation inside `ColumnTransformer`, zero data leakage
- ✅ **SMOTE oversampling** inside `ImbPipeline` to handle class imbalance (27.4% canceled)
- ✅ **Streamlit dashboard** with EDA, real-time prediction, and model performance page

---

## 🏆 Results

### 5-Fold Cross-Validation (on 70% training data)

| Rank | Model | F1 (CV) | Accuracy (CV) | F1 Train | Acc Train |
|:----:|-------|:-------:|:-------------:|:--------:|:---------:|
| 1 | **XGBoost** 🥇 | **0.6867** | **0.8323** | 0.7311 | 0.8556 |
| 2 | LightGBM | 0.6714 | 0.8183 | 0.6848 | 0.8255 |
| 3 | Random Forest | 0.6456 | 0.7639 | 0.6604 | 0.7731 |
| 4 | Logistic Regression | 0.6144 | 0.7355 | 0.6153 | 0.7363 |
| 5 | K-Nearest Neighbors | 0.5921 | 0.7074 | 0.7365 | 0.8160 |
| 6 | Decision Tree | 0.5829 | 0.7620 | 0.9968 | 0.9983 |

### Held-Out Test Set (30% unseen data)

| Metric | Score |
|--------|:-----:|
| **Test F1 Score** | **0.6878** |
| **Test Accuracy** | **0.8299** |
| Precision (Canceled) | 0.69 |
| Recall (Canceled) | 0.68 |

> **Final Model:** XGBoost — `learning_rate=0.1`, `max_depth=6`, `n_estimators=200`  
> **Decision Tree** shows near-perfect training scores but **severe overfitting** (F1 train=0.9968 → test=0.5829).  
> **XGBoost** achieves the best balance of bias and variance.

---

## 📊 Dataset

**Source:** [Kaggle — Hotel Booking Demand](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand)

| Property | Value |
|----------|-------|
| Original rows | 119,390 |
| Original columns | 32 |
| Cleaned rows | 86,494 |
| Features (after engineering) | 28 |
| Missing values (final) | 0 |
| Target | `is_canceled` (0 = Not Canceled, 1 = Canceled) |
| Class balance | 72.6% Not Canceled / 27.4% Canceled |

### Cleaning Rules (Mindmap "Impute After Split")

| Missing % | Rule | Action | Columns |
|:---------:|:----:|:------:|:-------:|
| > 40% | Drop column | Removed | `company` (94.3%) |
| 5% – 40% | Impute in Pipeline | `SimpleImputer` | `agent` (13.6%) |
| < 5% | Drop rows | Removed | `country` (0.4%), `children` (0.003%) |

### Additional Cleaning

- Dropped **leakage columns**: `reservation_status`, `reservation_status_date` (post-booking info)
- Dropped **agent**: ID column with 332 unique values, not a predictive feature
- **Clipped ADR**: 1 negative value (-6.38) set to 0
- Removed **161 bookings** with `total_people = 0` (invalid)
- Removed **duplicate rows**

---

## 🔧 Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         DATA CLEANING                              │
│  hotel_bookings.csv → drop cols/rows → clip ADR → drop dupes       │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      FEATURE ENGINEERING                           │
│  total_stay = weekend + week nights                                │
│  total_people = adults + children + babies                         │
│  arrival_month_num (string → int 1–12)                             │
│  arrival_season (Winter/Spring/Summer/Autumn)                      │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     TRAIN / TEST SPLIT (70/30)                     │
│  stratified by target → preserve class distribution                │
└─────────────────────────────────────────────────────────────────────┘
                                    │
           ┌────────────────────────┼────────────────────────┐
           ▼                        ▼                        ▼
┌──────────────────────┐  ┌──────────────────────┐  ┌──────────────────────┐
│   NUMERICAL (18)     │  │   COUNTRY (1)        │  │   CATEGORICAL (7)    │
│  SimpleImputer(median)│  │  SimpleImputer(Other)│  │  SimpleImputer(mode) │
│  → RobustScaler()    │  │  → BinaryEncoder()   │  │  → OneHotEncoder()   │
└──────────────────────┘  └──────────────────────┘  └──────────────────────┘
                                    │                        │
                    ┌───────────────┘                        │
                    ▼                                        ▼
          ┌──────────────────────┐                 ┌──────────────────────┐
          │   ORDINAL (1)        │                 │   SMOTE              │
          │  OrdinalEncoder()    │                 │  oversampling        │
          │  Winter<Spring<...   │                 └──────────────────────┘
          └──────────────────────┘                           │
                    │                                        ▼
                    ▼                              ┌──────────────────────┐
          ┌──────────────────────┐                 │   CLASSIFIER         │
          │  COLUMNTRANSFORMER   │◄────────────────│  XGBoost (tuned)     │
          │  (4 pipelines)       │                 └──────────────────────┘
          └──────────────────────┘                           │
                    │                                        ▼
                    ▼                              ┌──────────────────────┐
          ┌──────────────────────┐                 │   PREDICTION         │
          │   ImbPipeline        │────────────────►│  Cancel / Not Cancel │
          │   (SMOTE + Model)    │                 │  + Probability       │
          └──────────────────────┘                 └──────────────────────┘
```

---

## 📁 File Structure

```
📦 hotel-booking-cancellation
├── app.py                              # Streamlit dashboard (EDA + Prediction + Performance + About)
├── hotel_booking_pipeline.ipynb        # Full notebook (end-to-end pipeline with 6 models)
├── hotel_pipeline.pkl                  # Trained XGBoost pipeline (ImbPipeline: Preprocessing → SMOTE → XGBoost)
├── hotel_bookings.csv                  # Original Kaggle dataset (119,390 × 32)
├── requirements.txt                    # Python dependencies
└── README.md                           # This file
```

---

## 🚀 Installation & Usage

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/hotel-booking-cancellation.git
cd hotel-booking-cancellation
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> **ℹ️** If you're using a virtual environment (recommended):
> ```bash
> python -m venv venv
> .\venv\Scripts\activate  # Windows
> source venv/bin/activate  # Linux / macOS
> pip install -r requirements.txt
> ```

### 3. Run the Streamlit dashboard

```bash
streamlit run app.py
```

The dashboard opens in your browser with 4 pages:

| Page | Description |
|------|-------------|
| **EDA Dashboard** | Metrics cards, interactive histograms by cancel status, correlation heatmap, categorical analysis |
| **Prediction** | Input form with 22 fields → real-time prediction + probability chart |
| **Model Performance** | 6-model CV comparison (F1 + Accuracy), hold-out test metrics, full results table |
| **About** | Project info and technical details |

### 4. (Optional) Run the notebook

Open `hotel_booking_pipeline.ipynb` in Jupyter and run **Restart & Run All** to reproduce the full pipeline.

---

## 🛠️ Tech Stack

| Library | Purpose |
|---------|---------|
| [pandas](https://pandas.pydata.org/) | Data manipulation & cleaning |
| [numpy](https://numpy.org/) | Numerical computing |
| [scikit-learn](https://scikit-learn.org/) | Preprocessing, models, CV, GridSearch |
| [XGBoost](https://xgboost.readthedocs.io/) | Gradient boosting (best model) |
| [LightGBM](https://lightgbm.readthedocs.io/) | Gradient boosting (runner-up) |
| [imbalanced-learn](https://imbalanced-learn.org/) | SMOTE oversampling |
| [category-encoders](https://contrib.scikit-learn.org/category_encoders/) | BinaryEncoder for high-cardinality |
| [joblib](https://joblib.readthedocs.io/) | Model serialization |
| [plotly](https://plotly.com/python/) | Interactive visualizations |
| [Streamlit](https://streamlit.io/) | Web dashboard |
| [python-pptx](https://python-pptx.readthedocs.io/) | PowerPoint generation |

---

## 📈 Business Impact

- ✅ **Identifies ~69% of cancellations** correctly — enables proactive overbooking strategies
- ✅ **Reduces last-minute vacancy** — optimize room allocation and pricing
- ✅ **Supports data-driven revenue management** — dynamic pricing + cancellation policies
- ✅ **Deployed as interactive tool** — non-technical staff can use it without writing code

---

## 👤 Author

**Mazen Raafat**
---

## 📄 License

This project is for educational purposes as part of the **Data Science Course (Final Project)**.  
The original dataset is from [Kaggle](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand).
