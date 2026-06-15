# Hotel Booking Cancellation Prediction

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.7-orange?logo=scikit-learn)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.1-red)](https://xgboost.readthedocs.io)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.6-green)](https://lightgbm.readthedocs.io)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.41-FF4B4B?logo=streamlit)](https://streamlit.io)
[![Plotly](https://img.shields.io/badge/Plotly-6.5-3D5AFE?logo=plotly)](https://plotly.com)

---

## Overview

End-to-end machine learning pipeline that predicts whether a hotel booking will be canceled. Built from raw Kaggle data through cleaning, feature engineering, preprocessing, **6-model comparison with 5-fold cross-validation**, hyperparameter tuning (GridSearchCV on top 4 models), and deployment as an **interactive Streamlit dashboard**.

**Key highlights:**
- **86,539 clean rows** processed from 119,390 raw — missing values handled per mindmap rules
- **No train/test split** — cross-validation is the sole evaluation metric (no data waste)
- **6 models compared** — Logistic Regression, LightGBM, KNN, XGBoost, Random Forest, Decision Tree
- **GridSearchCV on top 4 models** — XGBoost best with F1=0.5902 (lr=0.01, max_depth=3, n_estimators=200)
- **Impute in Pipeline** — all imputation inside `ColumnTransformer`, zero data leakage
- **SMOTE oversampling** inside `ImbPipeline` to handle class imbalance (27.4% canceled)
- **Streamlit dashboard** with EDA, real-time prediction, and model performance page

---

## Results

### 5-Fold Cross-Validation (6 Models)

| Rank | Model | CV F1 | CV Accuracy |
|:----:|-------|:-----:|:-----------:|
| 1 | Logistic Regression | 0.5708 | 0.6896 |
| 2 | LightGBM | 0.4891 | 0.7312 |
| 3 | K-Nearest Neighbors | 0.4579 | 0.5949 |
| 4 | XGBoost | 0.4547 | 0.7126 |
| 5 | Random Forest | 0.4438 | 0.7165 |
| 6 | Decision Tree | 0.4167 | 0.6685 |

### GridSearchCV (Top 4 Tuned)

| Model | Best Params | Tuned CV F1 |
|-------|-------------|:-----------:|
| **XGBoost** (Best) | lr=0.01, max_depth=3, n_estimators=200 | **0.5902** |
| LightGBM | lr=0.01, max_depth=3, n_estimators=200 | 0.5880 |
| Logistic Regression | C=10 | 0.5708 |
| K-Nearest Neighbors | n_neighbors=11, weights=uniform | 0.4722 |

> **Final Model:** XGBoost (tuned) trained on full dataset  
> **Training F1:** 0.6166 | **Training Accuracy:** 0.7158  
> **No hold-out test set** — 5-fold CV is the sole evaluation, maximizing data for training.

---

## Dataset

**Source:** [Kaggle — Hotel Booking Demand](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand)

| Property | Value |
|----------|-------|
| Original rows | 119,390 |
| Original columns | 32 |
| Cleaned rows | 86,539 |
| Features (after engineering) | 28 |
| Missing values (final) | 0 |
| Target | `is_canceled` (0 = Not Canceled, 1 = Canceled) |
| Class balance | 72.6% Not Canceled / 27.4% Canceled |

### Cleaning Rules (Mindmap "Impute in Pipeline")

| Missing % | Rule | Action | Columns |
|:---------:|:----:|:------:|:-------:|
| > 40% | Drop column | Removed | `company` (94.3%) |
| 5% – 40% | Impute in Pipeline | `SimpleImputer` | `agent` (13.6%) |
| < 5% | Drop rows | Removed | `country` (0.4%), `children` (0.003%) |

### Additional Cleaning

- Dropped **leakage columns**: `reservation_status`, `reservation_status_date` (post-booking info)
- Dropped **agent**: ID column with 332 unique values, not a predictive feature
- Removed **countries with < 5 occurrences** (64 rare countries) — done after all other cleaning including duplicate removal
- **Clipped ADR**: 1 negative value (-6.38) set to 0
- Removed **bookings with total_people = 0** (invalid)
- Removed **duplicate rows**

---

## Pipeline Architecture

```
                            DATA CLEANING
 hotel_bookings.csv → drop cols/rows → country filter → clip ADR → drop dupes
                                    │
                                    ▼
                          FEATURE ENGINEERING
        total_stay = weekend + week nights
        total_people = adults + children + babies
        arrival_month_num (string → int 1–12)
        arrival_season (Winter/Spring/Summer/Autumn)
                                    │
                                    ▼
┌───────────────────────────────────────────────────────────────┐
│                     COLUMNTRANSFORMER                         │
│                                                               │
│  ┌─────────────┐  ┌─────────────┐  ┌──────────────────────┐  │
│  │ NUMERICAL   │  │ CATEGORICAL │  │ ORDINAL              │  │
│  │ (18 cols)   │  │ (8 cols)    │  │ (1 col: season)      │  │
│  │ Impute(med) │  │ Impute(mode)│  │ OrdinalEncoder()     │  │
│  │ RobustScaler│  │ OneHot      │  │ Winter<Spring<...    │  │
│  └─────────────┘  └─────────────┘  └──────────────────────┘  │
└───────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
                          SMOTE Oversampling
                                    │
                                    ▼
                    XGBoost (tuned: lr=0.01, depth=3, n_est=200)
                                    │
                                    ▼
                         Cancel / Not Cancel
                          + Probability
```

---

## File Structure

```
hotel-booking-cancellation
├── app.py                              # Streamlit dashboard (EDA + Prediction + Performance)
├── hotel_booking_pipeline.ipynb        # Full notebook (end-to-end pipeline with 6 models)
├── hotel_pipeline.pkl                  # Trained XGBoost pipeline (ImbPipeline)
├── hotel_bookings.csv                  # Original Kaggle dataset (119,390 x 32)
├── cleaned_df.csv                      # Cleaned dataset (86,539 x 28)
├── Hotel_Booking_Cancellation.pptx     # Project presentation
├── requirements.txt                    # Python dependencies
└── README.md                           # This file
```

---

## Installation & Usage

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/hotel-booking-cancellation.git
cd hotel-booking-cancellation
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> If you are using a virtual environment (recommended):
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
| **EDA Dashboard** | Metrics cards, interactive histograms by cancel status, categorical analysis |
| **Prediction** | Input form with 22 fields → real-time prediction + probability chart |
| **Model Performance** | 6-model CV comparison, GridSearchCV tuning results, final model details |
| **About** | Project info and technical details |

### 4. (Optional) Run the notebook

Open `hotel_booking_pipeline.ipynb` in Jupyter and run **Restart & Run All** to reproduce the full pipeline.

---

## Tech Stack

| Library | Purpose |
|---------|---------|
| [pandas](https://pandas.pydata.org/) | Data manipulation & cleaning |
| [numpy](https://numpy.org/) | Numerical computing |
| [scikit-learn](https://scikit-learn.org/) | Preprocessing, models, CV, GridSearch |
| [XGBoost](https://xgboost.readthedocs.io/) | Gradient boosting (best model) |
| [LightGBM](https://lightgbm.readthedocs.io/) | Gradient boosting (runner-up) |
| [imbalanced-learn](https://imbalanced-learn.org/) | SMOTE oversampling |
| [joblib](https://joblib.readthedocs.io/) | Model serialization |
| [plotly](https://plotly.com/python/) | Interactive visualizations |
| [Streamlit](https://streamlit.io/) | Web dashboard |
| [python-pptx](https://python-pptx.readthedocs.io/) | PowerPoint generation |

---

## Business Impact

- Identifies cancellations with competitive F1 score — enables proactive overbooking strategies
- Reduces last-minute vacancy — optimize room allocation and pricing
- Supports data-driven revenue management — dynamic pricing + cancellation policies
- Deployed as interactive tool — non-technical staff can use it without writing code

---

## Author

**Mazen Raafat**

- [LinkedIn](www.linkedin.com/in/mazen-raafat1)
- [GitHub]([https://github.com/<your-username>](https://github.com/Mazen2312))

---

## License

This project is for educational purposes as part of the **Data Science Course (Final Project)**.
The original dataset is from [Kaggle](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand).
