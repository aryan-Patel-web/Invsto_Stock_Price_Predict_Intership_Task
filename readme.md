# Stock Price Prediction using ARIMA and Gradient Boosting
### Invsto Data Science Internship Assignment
**Author:** Aryan Patel  
**Date:** December 2024

---

invsto depoyed model in streamlit url = https://invsto-stock-prediction.streamlit.app/

## 📋 Project Overview

This project implements a comprehensive stock price prediction pipeline for hedge fund trading, using both statistical (ARIMA) and machine learning (Gradient Boosting) approaches to forecast stock prices and evaluate trading strategies.

## 🎯 Objectives

1. Build robust data pipeline for processing historical OHLC stock data
2. Perform extensive exploratory data analysis (EDA)
3. Engineer predictive features for time series modeling
4. Develop and optimize ARIMA and Gradient Boosting models
5. Evaluate model performance using multiple metrics
6. Analyze trading strategy implications

## 📊 Dataset

- **Source:** Yahoo Finance (via yfinance API)
- **Stocks Analyzed:** AAPL, MSFT, GOOGL, AMZN, TSLA, JPM, JNJ, WMT
- **Time Period:** January 2020 - Present
- **Data Format:** Daily OHLC (Open, High, Low, Close) + Volume

## 🛠️ Technologies Used

**Core Libraries:**
- Python 3.8+
- pandas & numpy (data manipulation)
- yfinance (data collection)
- matplotlib & seaborn (visualization)

**Modeling:**
- statsmodels (ARIMA)
- scikit-learn (Gradient Boosting, preprocessing, metrics)
- scipy (statistical tests)

## 📁 Project Structure
```
stock-price-prediction/
│
├── stock_prediction_notebook.ipynb    # Main Jupyter notebook
├── stock_prediction.py                 # Python script version
├── requirements.txt                    # Dependencies
├── README.md                           # This file
├── REPORT.md                           # Detailed analysis report
│
└── outputs/                            # Generated visualizations
    ├── AAPL_trends.png
    ├── AAPL_distributions.png
    ├── AAPL_technical_indicators.png
    ├── AAPL_arima_forecast.png
    ├── AAPL_gb_predictions.png
    ├── AAPL_model_comparison.png
    └── ... (15+ visualization files)
```

## 🚀 Quick Start

### Installation
```bash
# Clone the repository
git clone <your-repo-url>
cd stock-price-prediction

# Install dependencies
pip install -r requirements.txt
```

### Running the Analysis

**Option 1: Jupyter Notebook (Recommended)**
```bash
jupyter notebook stock_prediction_notebook.ipynb
```

**Option 2: Python Script**
```bash
python stock_prediction.py
```

## 📈 Methodology

### 1. Data Preparation
- Automated data download from Yahoo Finance
- Missing value imputation using forward/backward fill
- Outlier detection using IQR method
- Time series formatting and validation

### 2. Exploratory Data Analysis
- Price and volume trend analysis
- Multi-stock performance comparison
- Distribution analysis (returns, volume)
- Seasonality detection (monthly, day-of-week patterns)
- Correlation analysis

### 3. Feature Engineering
Created 60+ features including:
- **Lagged features:** Previous 1, 2, 3, 5, 10 day prices/returns
- **Moving averages:** SMA and EMA (5, 10, 20, 50 day windows)
- **Volatility measures:** Rolling standard deviation
- **Volume indicators:** Volume ratios and moving averages
- **Technical indicators:** RSI, MACD, Bollinger Bands
- **Time-based features:** Day of week, month, quarter

### 4. ARIMA Model Development
- Stationarity testing (Augmented Dickey-Fuller test)
- ACF/PACF analysis for parameter selection
- Grid search for optimal (p, d, q) parameters
- Model diagnostics and residual analysis
- Out-of-sample forecasting

### 5. Gradient Boosting Model
- Feature scaling with StandardScaler
- Baseline model training
- Hyperparameter optimization via GridSearchCV
- Feature importance analysis
- Prediction and error analysis

### 6. Model Evaluation
**Metrics Used:**
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- MAPE (Mean Absolute Percentage Error)
- Directional Accuracy (% correct up/down predictions)

### 7. Trading Strategy Simulation
- Simple momentum strategy based on predictions
- Performance comparison vs. buy-and-hold benchmark
- Risk-return analysis

## 📊 Key Results

### Model Performance (AAPL)

| Model | RMSE | MAE | MAPE | Dir. Accuracy |
|-------|------|-----|------|---------------|
| ARIMA | ~$3.50 | ~$2.80 | ~1.5% | ~52% |
| Gradient Boosting | ~$2.20 | ~$1.60 | ~0.9% | ~56% |

**Winner:** Gradient Boosting (superior accuracy across all metrics)

### Trading Strategy Results

| Strategy | Total Return | Number of Trades |
|----------|--------------|------------------|
| ARIMA Strategy | Variable | ~15-25 |
| GB Strategy | Variable | ~20-30 |
| Buy & Hold | Variable | 1 |

*Note: Results vary based on time period and market conditions*

## 🎓 Key Insights

1. **Model Performance:**
   - Gradient Boosting outperforms ARIMA in accuracy
   - ARIMA better captures long-term trends
   - Ensemble approach recommended for production

2. **Feature Importance:**
   - Lagged prices and returns are top predictors
   - Technical indicators (RSI, MACD) add significant value
   - Volume features improve prediction accuracy

3. **Trading Implications:**
   - Both models struggle in high-volatility periods
   - Directional accuracy more important than exact price prediction
   - Transaction costs significantly impact strategy returns

4. **Risk Considerations:**
   - Models trained on historical data may not capture regime changes
   - Continuous retraining necessary for maintained performance
   - Proper risk management essential for live trading

## 💡 Recommendations for Hedge Fund Trading

1. **Model Deployment:**
   - Use Gradient Boosting as primary model
   - Implement ARIMA for trend confirmation
   - Weekly retraining schedule

2. **Risk Management:**
   - Stop-loss at 1.5× model RMSE
   - Position sizing based on prediction confidence
   - Portfolio diversification across multiple stocks

3. **Future Enhancements:**
   - Add sentiment analysis from news/social media
   - Incorporate macro-economic indicators
   - Implement deep learning models (LSTM, Transformers)
   - Develop multi-asset portfolio optimization

## 📝 Assignment Requirements Checklist

- ✅ Data Preparation: Cleaning, handling missing values, time series formatting
- ✅ EDA: Detailed price/volume analysis with visualizations
- ✅ Feature Engineering: Lagged variables, rolling means, percentage changes
- ✅ ARIMA Model: Parameter optimization, forecasting, comparison
- ✅ Gradient Boosting: Feature engineering, hyperparameter tuning
- ✅ Model Evaluation: RMSE, MAE, MAPE metrics
- ✅ Comprehensive Report: Methodology, findings, recommendations
- ✅ Visualizations: 15+ professional charts
- ✅ Trading Strategy Discussion: Implications and recommendations

## 📧 Contact

**Aryan Patel**  
Email: [Your Email]  
Phone: +91 91407 82212  
LinkedIn: [Your LinkedIn]  
GitHub: [Your GitHub]

## 📄 License

This project is created as part of the Invsto Data Science Internship application.

---

**Last Updated:** December 2024