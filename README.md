# Time Series Forecasting – One Piece Live Action Ratings Prediction

## Overview

This project explores time series forecasting techniques to predict the audience ratings of future episodes of the *One Piece Live Action* series.

The objective is not to build a production-ready predictive system, but to apply and compare classical and advanced forecasting models on a real-world dataset with strong limitations.

Due to the very small size of the dataset (approximately 16 observations), the predictive capacity of the models is inherently limited. The focus is therefore placed on:

- methodological comparison of forecasting approaches  
- statistical validation of time series assumptions  
- evaluation of model behavior under data scarcity  

---

## Important Note on Data Limitations

The dataset used in this project contains a very small number of observations (16 episodes).

As a result:
- predictions are not statistically robust  
- conclusions should be interpreted as methodological experiments rather than accurate forecasts  
- the main objective is learning and benchmarking forecasting techniques  

---

## Methods Used

### 1. Statistical Analysis

Before modeling, several statistical tests are performed:

- Augmented Dickey-Fuller (ADF) tests  
- Ljung-Box tests  

These tests are used to assess:
- stationarity of the series  
- presence of autocorrelation  

---

### 2. Baseline Forecasting Models

Simple forecasting approaches are implemented as benchmarks:

- Naive forecasting  
- Average (mean) forecasting  
- Drift forecasting  

---

### 3. ARIMA Models

More advanced models are used:

- Auto-ARIMA for parameter selection  
- Manual ARIMA configurations:
  - ARIMA(1,1,0)  
  - ARIMA(1,1,1)  

These models are evaluated against baseline methods.

---

## Evaluation Metrics

Models are evaluated using:

- Mean Absolute Error (MAE)  
- Root Mean Squared Error (RMSE)  

Results are compared on a train/test split of the dataset.

---

## Final Prediction

A final forecast is produced using an ARIMA model trained on the full dataset to predict the next 8 episodes’ ratings.

---

## Project Structure

```text
code/
├── code_julia/
│   ├── main.jl
│   ├── arima.jl
│   ├── basicForecasting.jl
│   ├── evaluationTests.jl
│   ├── dataset.jl
│   └── tools.jl
│
├── code_r/
│   └── succes.R
│
├── data/
│   ├── title.episode.tsv.gz
│   └── title.ratings.tsv.gz
│
├── outputs/
│   ├── figures/
│   └── results/
│
report.pdf
```

---

## Context

This is an academic project developed under time constraints as part of coursework in data analysis and forecasting.

The primary objective was to apply a full time series forecasting pipeline using methods covered in class, including statistical testing, baseline models, and ARIMA-based approaches.

As a result of these constraints, the codebase was not fully refactored after initial development. It reflects an iterative academic workflow rather than a production-oriented software architecture.

Despite this, the project was designed to remain structured and reproducible, with clear separation between data processing, modeling, and evaluation components.
