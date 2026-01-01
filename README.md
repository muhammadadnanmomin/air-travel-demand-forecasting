# Air Travel Demand Forecasting using Time Series & Machine Learning

App URL: https://air-travel-demand-forecasting.streamlit.app/

## Project Overview

This project focuses on forecasting monthly air passenger demand using historical aviation data.
It combines time series analysis, feature-based machine learning models, and end-to-end deployment to build a production-ready forecasting system.

The final model is deployed as an interactive Streamlit web application, allowing users to forecast future passenger demand dynamically.

## Problem Statement

Accurate forecasting of air travel demand is critical for:

- Airline capacity planning

- Airport operations

- Revenue management

- Infrastructure optimization

The goal of this project is to predict future monthly passenger demand using historical data while capturing:

- Long-term trends

- Seasonality

- Structural breaks (e.g., COVID-19 impact)

## Dataset Description

- Source: San Francisco International Airport (DataSF)

- Link: https://data.sfgov.org/Transportation/Air-Traffic-Passenger-Statistics/rkru-6vcg/about_data

- Time Period: July 1999 – September 2025

- Frequency: Monthly

- Target Variable: Passenger Count

### Key Features Used

- year

- month

- lag_1 (previous month passenger count)

- lag_12 (same month previous year passenger count)

## Project Structure
```air-travel-demand-forecasting/
│
├── data/
│   ├── raw/
│   │   └── Air_Traffic_Passenger_Statistics.csv
│   │
│   └── processed/
│       └── final_timeseries.csv
│
├── models/
│   └── lightgbm_model.pkl
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_time_series_preparation.ipynb
│   └── 03_modeling_forecasting.ipynb
│
├── app.py
├── Dockerfile
├── requirements.txt
├── README.md
├── .gitignore
└── venv/
```

## Exploratory Data Analysis (EDA)

Key insights from EDA:

- Strong upward long-term trend in passenger demand

- Clear annual seasonality (summer peaks)

- Significant demand collapse in 2020 due to COVID-19

- Post-2021 recovery to pre-pandemic levels

- Increasing variance over time

Notebook: 01_eda.ipynb

## Time Series Preparation & Feature Engineering

Steps performed:

- Visual stationarity checks

- Augmented Dickey-Fuller (ADF) test

- KPSS test

- Log transformation for variance stabilization

- Trend and seasonal differencing

- Lag feature creation for ML models

Notebook: 02_time_series_preparation.ipynb

## Modeling & Forecasting

Models Evaluated

- SARIMA

- Prophet

- Holt-Winters

- XGBoost

- LightGBM

- CatBoost

Evaluation Metrics

- MAE (Mean Absolute Error)

- RMSE (Root Mean Squared Error)

- MAPE (Mean Absolute Percentage Error)

Final Model Selection

- LightGBM achieved the best overall performance and was selected for deployment.

Notebook: 03_modeling_forecasting.ipynb

## Deployment
Streamlit Web Application

The trained LightGBM model is deployed using Streamlit, allowing users to:

- Select forecast horizon

- Visualize historical vs forecasted demand

- View forecasted passenger counts in tabular format

Docker Support

The project includes a Dockerfile for containerized deployment, ensuring:

- Environment consistency

- Easy reproducibility

- Platform independence

## How to Run Locally

1. Clone the Repository
```
git clone https://github.com/muhammadadnanmomin/air-travel-demand-forecasting
cd air-travel-demand-forecasting
```

2. Install Dependencies
```
pip install -r requirements.txt
```

3. Run Streamlit App
```
streamlit run app.py
```

## Run Using Docker
```
docker build -t air-travel-forecast .
docker run -d -p 8501:8501 air-travel-forecast
```


Open in browser:

http://localhost:8501

## Results & Insights

- ML-based time series models outperform classical statistical models

- Feature-based forecasting handles non-stationarity effectively

- Lag-based models capture seasonality without strict stationarity assumptions

- LightGBM provides a strong balance between accuracy and scalability

## Key Learnings

- Practical application of time series theory

- Difference between statistical and ML-based forecasting

- Importance of temporal train-test split

- Feature consistency between training and inference

- End-to-end ML deployment workflow

## Future Improvements

- Add confidence intervals to forecasts

- Incorporate external variables (holidays, fuel prices)

- Automate retraining pipeline

- Add model monitoring and drift detection


## Author

Adnan Momin

LinkedIn: https://www.linkedin.com/in/adnanmomin/

GitHub: https://github.com/muhammadadnanmomin
