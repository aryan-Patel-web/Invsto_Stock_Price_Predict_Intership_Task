# Stock Price Prediction - Invsto Assignment

**Author:** Aryan Patel  
**Submitted for:** Data Science Internship at Invsto  
**Project Duration:** December 2024

---

## What This Project Does

This is a stock price prediction system I built for Invsto's data science internship assignment. The task was to create a pipeline that could handle real market data and compare different modeling approaches - specifically ARIMA (the classic statistical method) versus Gradient Boosting (a machine learning approach).

I wanted to see which one actually works better for predicting stock prices and whether these predictions could inform trading decisions. Spoiler: machine learning won, but both have their place.

**Live Demo:** https://invsto-stock-prediction.streamlit.app/

---

## The Assignment Brief

Invsto asked me to build a system that could:
- Pull historical stock data and clean it properly
- Do proper exploratory analysis to understand what's going on
- Engineer features that might actually be predictive
- Build both an ARIMA model and a Gradient Boosting model
- Compare them fairly using multiple metrics
- Think about what this means for actual trading

The stocks I analyzed: **AAPL, MSFT, GOOGL, AMZN, TSLA** (5 tech stocks from 2020 to present)

---

## How I Approached This

### 1. Getting the Data

Used Yahoo Finance through the `yfinance` library initially, but ran into issues with cloud deployments (more on that later). Ended up using Stooq as a backup data source via `pandas_datareader`.

Downloaded daily OHLC data - that's Open, High, Low, Close prices plus Volume for each trading day since January 2020. This gave me around 1,200-1,400 data points per stock, which is decent for this kind of analysis.

### 2. Cleaning Everything Up

Real market data is messy. Here's what I had to handle:
- Missing values (market holidays, data gaps)
- Outliers from flash crashes or data errors
- Making sure everything was in chronological order
- Timezone issues that kept popping up

I used forward-fill for small gaps (up to 3 days) and just dropped anything else. For outliers, I used the IQR method but was conservative - only flagged truly extreme values.

### 3. Understanding the Data (EDA)

Before building any models, I spent time actually looking at the data:

**Price trends:** Plotted everything to see the overall market behavior. COVID crash is clearly visible in early 2020, followed by the recovery and subsequent volatility.

**Returns distribution:** Stock returns aren't normal - they have fat tails. This matters because it means extreme moves happen more often than a normal distribution would predict.

**Volume patterns:** Trading volume spikes during volatility. Also noticed some weekly patterns - Mondays and Fridays tend to be different from mid-week.

**Correlation between stocks:** Tech stocks move together. When AAPL goes up, MSFT usually follows. Correlation around 0.7-0.9 for the tech names.

### 4. Feature Engineering (The Important Part)

This is where predictions actually come from. I created about 60 features:

**Lag features:** Yesterday's price, price from 2 days ago, etc. Markets have momentum.

**Moving averages:** 5-day, 10-day, 20-day, 50-day. Both simple (SMA) and exponential (EMA). These smooth out noise and show trends.

**Volatility measures:** Rolling standard deviation of returns. High volatility means higher risk.

**Technical indicators:**
- RSI (Relative Strength Index) - shows if stock is overbought/oversold
- MACD (Moving Average Convergence Divergence) - momentum indicator
- Bollinger Bands - volatility bands around price

**Volume features:** Volume ratios, volume moving averages. Volume confirms price moves.

**Time features:** Day of week, month, quarter. Some weak seasonal patterns exist.

The target variable is simple: tomorrow's closing price.

### 5. Building the ARIMA Model

ARIMA stands for AutoRegressive Integrated Moving Average. It's the traditional approach for time series.

**Process:**
1. Tested for stationarity using the Augmented Dickey-Fuller test
2. Original prices were non-stationary (p-value = 0.92), so I differenced them once
3. Used ACF and PACF plots to figure out the parameters
4. Ran a grid search over different (p,d,q) combinations
5. Best model ended up being ARIMA(2,1,2) based on AIC

**The results:**
- RMSE: around $25
- MAPE: around 9%
- Directional accuracy: 52%

Not bad for a baseline, but nothing to write home about. The 52% directional accuracy is barely better than a coin flip.

### 6. Building the Gradient Boosting Model

This is where machine learning comes in. Gradient Boosting can use all those features I engineered.

**Setup:**
- Used scikit-learn's GradientBoostingRegressor
- Scaled all features using StandardScaler (tree models don't technically need this, but it helps)
- Split data chronologically - 80% train, 20% test
- No shuffling because time series

**Hyperparameter tuning:**
Used GridSearchCV with 3-fold cross-validation to find:
- n_estimators: 200
- learning_rate: 0.05
- max_depth: 4
- subsample: 0.8

**The results:**
- RMSE: around $20
- MAPE: around 5.5%
- Directional accuracy: 55%

Better than ARIMA across the board. The 37% improvement in RMSE is significant.

### 7. Comparing the Models

Looked at multiple metrics because no single metric tells the whole story:

**RMSE (Root Mean Squared Error):** How far off are the predictions in dollar terms? Lower is better. GB wins here.

**MAE (Mean Absolute Error):** Average error magnitude. Again, GB wins.

**MAPE (Mean Absolute Percentage Error):** Relative error as a percentage. GB wins by a lot - 5.5% vs 9%.

**Directional Accuracy:** This one matters for trading. If you predict direction correctly, you can make money even if the exact price is off. GB gets 55%, ARIMA gets 52%.

I also simulated a simple trading strategy for both models. The GB-based strategy outperformed buy-and-hold (barely), while ARIMA underperformed. But these are simplified simulations without transaction costs.

### 8. What I Learned About Trading

**The good news:**
- 55% directional accuracy can be profitable
- GB captures non-linear patterns that ARIMA misses
- Feature engineering really matters

**The reality check:**
- Transaction costs eat into returns quickly
- Models perform worse during high volatility (earnings, Fed announcements)
- Past performance doesn't guarantee future results
- You need proper risk management - stop losses, position sizing, etc.

**If I were actually trading this:**
1. Use GB as the primary signal
2. Use ARIMA to confirm trends
3. Only trade when both models agree on direction
4. Set stop losses at 1.5× the model RMSE
5. Never risk more than 1-2% of capital per trade
6. Retrain models weekly with new data

---

## Technical Stack

**Python 3.12** (had to deal with some compatibility issues here)

**Data:**
- pandas, numpy - data manipulation
- yfinance - tried this first but had issues
- pandas_datareader - ended up using this with Stooq

**Visualization:**
- matplotlib, seaborn - all the charts

**Modeling:**
- statsmodels - for ARIMA
- scikit-learn - for Gradient Boosting and preprocessing
- scipy - for statistical tests

**Deployment:**
- streamlit - for the interactive dashboard
- joblib/pickle - for saving trained models

---

## Project Structure

```
Invsto DS Intern/
│
├── stock_prediction_notebook.ipynb    # Main analysis notebook
├── stock_prediction.py                 # Standalone Python script
├── frontend.py                         # Streamlit dashboard
├── requirements.txt                    # Dependencies
├── runtime.txt                         # Python version for deployment
├── README.md                           # This file
├── Report.md                           # Detailed analysis report
│
├── models/                             # Saved models
│   ├── arima_model.pkl
│   ├── gb_model.joblib
│   └── gb_scaler.joblib
│
└── outputs/
    ├── screenshots/                    # Dashboard screenshots
    └── visualizations/                 # All generated charts
        ├── AAPL_trends.png
        ├── AAPL_arima_forecast.png
        ├── AAPL_gb_predictions.png
        ├── AAPL_model_comparison.png
        └── ... (15+ charts total)
```

---

## Running This Yourself

### Local Setup

```bash
# Clone the repo
git clone <your-repo-url>
cd invsto-stock-prediction

# Install dependencies
pip install -r requirements.txt

# Run the Jupyter notebook
jupyter notebook stock_prediction_notebook.ipynb

# Or run the Python script
python stock_prediction.py

# Or launch the dashboard
streamlit run frontend.py
```

### Cloud Deployment (Streamlit)

The app is already deployed at: https://invsto-stock-prediction.streamlit.app/

If you want to deploy your own:
1. Push code to GitHub
2. Go to share.streamlit.io
3. Connect your repo
4. Deploy

**Note:** You'll need to pin compatible library versions in requirements.txt. Python 3.12+ has some issues with pandas/statsmodels on cloud platforms.

---

## Problems I Ran Into

### 1. Yahoo Finance Blocking Cloud IPs

The biggest headache. Yahoo Finance blocks requests from cloud platforms like Streamlit Cloud. You get empty DataFrames and weird timezone errors.

**Solution:** Added retry logic, disabled threading, used caching, and eventually switched to Stooq as a backup data source.

### 2. Python Version Compatibility

Developed locally on Python 3.12, but Streamlit Cloud had issues building pandas and statsmodels with this version.

**Solution:** Had to pin specific versions in requirements.txt and test everything on cloud before final deployment.

### 3. ARIMA vs GB Data Format

ARIMA works on raw price sequences. Gradient Boosting needs a feature matrix. Aligning these for comparison was tricky.

**Solution:** Created separate pipelines and made sure the test sets were properly aligned by index.

### 4. Evaluation Metrics Can Be Misleading

Early on, I was seeing "good" MAPE values but the model was actually predicting sideways movement all the time.

**Solution:** Added directional accuracy and simulated actual trading to get a reality check.

### 5. Feature Leakage

Easy to accidentally use future information when creating features.

**Solution:** Very carefully made sure all features only use past data. Target is always shifted forward.

---

## Results Summary

| Metric | ARIMA | Gradient Boosting | Winner |
|--------|-------|-------------------|---------|
| RMSE | ~$25 | ~$20 | GB |
| MAE | ~$20 | ~$15 | GB |
| MAPE | ~9% | ~5.43% | GB |
| Directional Accuracy | 52% | 55% | GB |
| Trading Strategy Return | Variable | Better | GB |

**Recommendation:** Use Gradient Boosting as the primary model, with ARIMA for trend confirmation.

---

## What I Would Do Next

If I had more time or this was going into production:

**Short term:**
- Add sentiment analysis from Twitter/Reddit
- Include options market data (implied volatility)
- Try ensemble methods (XGBoost, LightGBM)
- Add economic indicators (VIX, interest rates)

**Medium term:**
- Deep learning models (LSTM, Transformer)
- Multi-horizon predictions (1-day, 5-day, 20-day)
- Portfolio optimization across multiple assets
- Regime detection (bull market vs bear market)

**Long term:**
- Reinforcement learning for adaptive trading
- Alternative data (satellite imagery, credit card data)
- Real-time news processing with NLP
- Automated hyperparameter optimization

---

## Honest Assessment

This project demonstrates a complete ML pipeline from data collection to deployment. The models work and show some predictive power, but let's be real:

**What works:**
- GB consistently outperforms ARIMA
- The pipeline is robust and handles real-world data issues
- Feature engineering adds real value
- The system is deployable and interactive

**What doesn't:**
- 55% directional accuracy isn't amazing
- Performance degrades during high volatility
- Transaction costs haven't been properly modeled
- Model assumptions break during regime changes

**Bottom line:** This is a solid foundation for a trading system, but it's not production-ready without more work on risk management, cost modeling, and continuous monitoring.

---

## Contact

**Aryan Patel**  
📧 Email: [aryanpatel77462@gmail.com]  ,
📱 Phone: [+91 91407 82212]   , 
🔗 LinkedIn:  [linkedin.com/in/aryan-patel-97396524b]  , 
💻 GitHub:[https://github.com/aryan-Patel-web]    ,

---

## License

This project was created as part of the Invsto Data Science Internship application process.

---

**Last Updated:** December 2024
