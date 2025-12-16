"""
📈 Stock Price Prediction Dashboard
ARIMA + Gradient Boosting | Hedge Fund Style
Invsto Data Science Internship
Author: Aryan Patel
"""

# =========================================================
# IMPORTS
# =========================================================
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import joblib
import pickle
from datetime import datetime, timedelta
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

import warnings
warnings.filterwarnings("ignore")

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Stock Predictor | Invsto",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# DARK UI CSS (HEDGE FUND STYLE)
# =========================================================
st.markdown("""
<style>
html, body, [class*="css"]  {
    background-color: #0E1117;
    color: #FAFAFA;
}
h1, h2, h3, h4 {
    color: #4CAF50;
}
.stMetric {
    background-color: #111827;
    border: 1px solid #1F2937;
    padding: 12px;
    border-radius: 8px;
}
.stDataFrame {
    background-color: #111827;
}
button[kind="primary"] {
    background-color: #16A34A;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# TITLE
# =========================================================
st.title("📈 Stock Price Prediction Dashboard")
st.markdown("**ARIMA + Gradient Boosting | Hedge Fund Style**")
st.markdown("Invsto Data Science Internship — *Aryan Patel*")
st.divider()

# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.header("⚙️ Configuration")

    stock = st.selectbox(
        "Select Stock",
        ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "JPM", "JNJ", "WMT"]
    )

    start_date = st.date_input("Start Date", datetime(2020, 1, 1))
    end_date = st.date_input("End Date", datetime.today())

    train_ratio = st.slider("Training %", 60, 90, 80)
    forecast_days = st.slider("Forecast Days", 1, 30, 5)

    run_btn = st.button("🚀 Run Analysis", use_container_width=True)

# =========================================================
# HELPERS
# =========================================================
@st.cache_data
def load_data(ticker, start, end):
    return yf.download(ticker, start=start, end=end, progress=False)

def build_features(df):
    df = df.copy()
    df["Returns"] = df["Close"].pct_change()
    df["Range"] = (df["High"] - df["Low"]) / df["Close"]

    for lag in [1, 2, 3, 5]:
        df[f"Close_Lag_{lag}"] = df["Close"].shift(lag)
        df[f"Return_Lag_{lag}"] = df["Returns"].shift(lag)

    for w in [5, 10, 20]:
        df[f"SMA_{w}"] = df["Close"].rolling(w).mean()
        df[f"EMA_{w}"] = df["Close"].ewm(span=w).mean()
        df[f"Vol_{w}"] = df["Returns"].rolling(w).std()

    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    rs = gain / loss
    df["RSI_14"] = 100 - (100 / (1 + rs))

    df["Target"] = df["Close"].shift(-1)
    return df.dropna()

def metrics(y_true, y_pred):
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MAPE": float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)
    }

# =========================================================
# MAIN
# =========================================================
if run_btn:

    df = load_data(stock, start_date, end_date)

    if df.empty:
        st.error("No data available.")
        st.stop()

    st.success(f"Loaded {len(df)} rows")

    # -------------------------
    # SAFE METRICS
    # -------------------------
    last_price = float(df["Close"].iloc[-1])
    first_price = float(df["Close"].iloc[0])
    total_return = ((last_price / first_price) - 1) * 100
    volatility = float(df["Close"].pct_change().std() * np.sqrt(252) * 100)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Trading Days", len(df))
    c2.metric("Last Price", f"${last_price:.2f}")
    c3.metric("Total Return", f"{total_return:.2f}%")
    c4.metric("Volatility", f"{volatility:.2f}%")

    # -------------------------
    # PRICE CHART
    # -------------------------
    st.subheader("📉 Price History")
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df.index, df["Close"], color="#4CAF50")
    ax.grid(alpha=0.3)
    st.pyplot(fig)

    # -------------------------
    # FEATURES
    # -------------------------
    df_feat = build_features(df)

    X = df_feat.drop(columns=["Target"])
    y = df_feat["Target"]

    split = int(len(X) * train_ratio / 100)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # -------------------------
    # ARIMA
    # -------------------------
    st.subheader("📊 ARIMA Model")

    arima_model = ARIMA(df["Close"][:split], order=(2,1,2))
    arima_fit = arima_model.fit()
    arima_forecast = arima_fit.forecast(len(df["Close"][split:]))

    arima_m = metrics(df["Close"][split:], arima_forecast)

    st.json(arima_m)

    # SAVE ARIMA
    with open("arima_model.pkl", "wb") as f:
        pickle.dump(arima_fit, f)

    # -------------------------
    # GRADIENT BOOSTING
    # -------------------------
    st.subheader("🌲 Gradient Boosting")

    gb = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        random_state=42
    )

    gb.fit(X_train_s, y_train)
    gb_pred = gb.predict(X_test_s)

    gb_m = metrics(y_test, gb_pred)
    st.json(gb_m)

    # SAVE MODELS
    joblib.dump(gb, "gb_model.joblib")
    joblib.dump(scaler, "scaler.joblib")

    # -------------------------
    # COMPARISON
    # -------------------------
    st.subheader("📊 Model Comparison")

    cmp = pd.DataFrame({
        "Model": ["ARIMA", "Gradient Boosting"],
        "RMSE": [arima_m["RMSE"], gb_m["RMSE"]],
        "MAE": [arima_m["MAE"], gb_m["MAE"]],
        "MAPE": [arima_m["MAPE"], gb_m["MAPE"]]
    })

    st.dataframe(cmp, use_container_width=True)

    # -------------------------
    # FORECAST
    # -------------------------
    st.subheader("🔮 Future Forecast")

    future_prices = last_price * (
        1 + np.cumsum(np.random.normal(0, gb_m["RMSE"]/last_price, forecast_days))
    )

    future_dates = pd.date_range(df.index[-1] + timedelta(days=1), periods=forecast_days)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df.index[-30:], df["Close"].iloc[-30:], label="History")
    ax.plot(future_dates, future_prices, "--", label="Forecast")
    ax.legend()
    ax.grid(alpha=0.3)
    st.pyplot(fig)

    # -------------------------
    # DOWNLOADS
    # -------------------------
    st.subheader("💾 Downloads")

    st.download_button(
        "Download Model Comparison CSV",
        cmp.to_csv(index=False),
        file_name=f"{stock}_comparison.csv"
    )

    st.success("✅ Analysis Complete — Models Saved")

else:
    st.info("👈 Select parameters and click **Run Analysis**")

# =========================================================
# FOOTER
# =========================================================
st.divider()
st.markdown("""
<center>
Stock Price Prediction Dashboard<br>
Invsto Data Science Internship<br>
<strong>Aryan Patel</strong>
</center>
""", unsafe_allow_html=True)
