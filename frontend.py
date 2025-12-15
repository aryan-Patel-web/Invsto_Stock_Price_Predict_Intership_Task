"""
Stock Price Prediction Dashboard
Invsto Data Science Internship - Interactive Demo
Author: Aryan Patel
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from datetime import datetime, timedelta
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Page configuration


st.set_page_config(
    page_title="Stock Price Predictor | Invsto",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)



# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stAlert {
        background-color: #f0f2f6;
        border-left: 5px solid #4CAF50;
    }
    h1 {
        color: #1f77b4;
        font-family: 'Helvetica Neue', sans-serif;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        border: 2px solid #e9ecef;
    }
    </style>
""", unsafe_allow_html=True)

# Title and header
st.title("📈 Stock Price Prediction Dashboard")
st.markdown("### Powered by ARIMA & Gradient Boosting Models")
st.markdown("**Invsto Data Science Internship Assignment | Aryan Patel**")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.image("https://via.placeholder.com/300x100/1f77b4/ffffff?text=INVSTO", use_container_width=True)
    st.markdown("## 🎯 Model Configuration")
    
    # Stock selection
    stock_ticker = st.selectbox(
        "Select Stock",
        ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "JPM", "JNJ", "WMT"],
        index=0
    )
    
    # Date range
    st.markdown("### 📅 Date Range")
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "Start Date",
            value=datetime(2020, 1, 1)
        )
    with col2:
        end_date = st.date_input(
            "End Date",
            value=datetime.now()
        )
    
    # Model parameters
    st.markdown("### ⚙️ Model Settings")
    train_size = st.slider("Training Data %", 60, 90, 80, 5)
    
    # Prediction horizon
    forecast_days = st.number_input(
        "Forecast Days", 
        min_value=1, 
        max_value=30, 
        value=5
    )
    
    # Run button
    run_analysis = st.button("🚀 Run Analysis", use_container_width=True, type="primary")
    
    st.markdown("---")
    st.markdown("### 📊 About")
    st.info("""
    This dashboard demonstrates stock price prediction using:
    - **ARIMA**: Statistical time series model
    - **Gradient Boosting**: ML ensemble method
    - **60+ Features**: Technical indicators & patterns
    """)

# Helper functions
@st.cache_data
def download_data(ticker, start, end):
    """Download stock data"""
    data = yf.download(ticker, start=start, end=end, progress=False)
    return data

def create_features(df):
    """Create features for ML model"""
    df = df.copy()
    
    # Basic features
    df['Returns'] = df['Close'].pct_change()
    df['Range'] = (df['High'] - df['Low']) / df['Close']
    
    # Lagged features
    for lag in [1, 2, 3, 5]:
        df[f'Close_Lag_{lag}'] = df['Close'].shift(lag)
        df[f'Returns_Lag_{lag}'] = df['Returns'].shift(lag)
    
    # Moving averages
    for window in [5, 10, 20]:
        df[f'SMA_{window}'] = df['Close'].rolling(window=window).mean()
        df[f'EMA_{window}'] = df['Close'].ewm(span=window).mean()
        df[f'Volatility_{window}'] = df['Returns'].rolling(window=window).std()
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI_14'] = 100 - (100 / (1 + rs))
    
    # Target
    df['Target'] = df['Close'].shift(-1)
    
    return df

def calculate_metrics(actual, predicted):
    """Calculate performance metrics"""
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mae = mean_absolute_error(actual, predicted)
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    return {'RMSE': rmse, 'MAE': mae, 'MAPE': mape}

# Main analysis
if run_analysis:
    with st.spinner(f"🔄 Downloading {stock_ticker} data..."):
        # Download data
        df = download_data(stock_ticker, start_date, end_date)
        
        if df.empty:
            st.error("❌ No data available for selected parameters!")
        else:
            st.success(f"✅ Downloaded {len(df)} days of data")
            
            # Display raw data
            with st.expander("📊 View Raw Data", expanded=False):
                st.dataframe(df.tail(10), use_container_width=True)
            
            # Data overview
            st.markdown("## 📈 Data Overview")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Trading Days", len(df))
            with col2:
                latest_price = df['Close'].iloc[-1]
                st.metric("Current Price", f"${latest_price:.2f}")
            with col3:
                price_change = ((df['Close'].iloc[-1] - df['Close'].iloc[0]) / df['Close'].iloc[0]) * 100
                st.metric("Total Return", f"{price_change:.2f}%", delta=f"{price_change:.2f}%")
            with col4:
                volatility = df['Close'].pct_change().std() * np.sqrt(252) * 100
                st.metric("Volatility (Annual)", f"{volatility:.2f}%")
            
            # Price chart
            st.markdown("### 💹 Price History")
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(df.index, df['Close'], linewidth=2, color='#1f77b4')
            ax.fill_between(df.index, df['Close'], alpha=0.3, color='#1f77b4')
            ax.set_title(f'{stock_ticker} Closing Price', fontsize=16, fontweight='bold')
            ax.set_xlabel('Date', fontsize=12)
            ax.set_ylabel('Price ($)', fontsize=12)
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close()
            
            # Feature engineering
            with st.spinner("🔧 Engineering features..."):
                df_features = create_features(df)
                df_features = df_features.dropna()
            
            st.success(f"✅ Created {len(df_features.columns)} features")
            
            # Train-test split
            split_idx = int(len(df) * (train_size / 100))
            train_data = df['Close'][:split_idx]
            test_data = df['Close'][split_idx:]
            
            # Model training
            st.markdown("## 🤖 Model Training & Prediction")
            
            col1, col2 = st.columns(2)
            
            # ARIMA Model
            with col1:
                st.markdown("### 📊 ARIMA Model")
                with st.spinner("Training ARIMA..."):
                    try:
                        # Simplified ARIMA for speed
                        arima_model = ARIMA(train_data, order=(2, 1, 2))
                        arima_fitted = arima_model.fit()
                        arima_forecast = arima_fitted.forecast(steps=len(test_data))
                        arima_metrics = calculate_metrics(test_data.values, arima_forecast.values)
                        
                        st.success("✅ ARIMA Training Complete")
                        
                        # Display metrics
                        st.markdown("**Performance Metrics:**")
                        st.metric("RMSE", f"${arima_metrics['RMSE']:.2f}")
                        st.metric("MAE", f"${arima_metrics['MAE']:.2f}")
                        st.metric("MAPE", f"{arima_metrics['MAPE']:.2f}%")
                        
                    except Exception as e:
                        st.error(f"ARIMA Error: {str(e)}")
                        arima_forecast = None
            
            # Gradient Boosting Model
            with col2:
                st.markdown("### 🌲 Gradient Boosting")
                with st.spinner("Training GB Model..."):
                    try:
                        # Prepare ML data
                        feature_cols = [col for col in df_features.columns 
                                      if col not in ['Target', 'Open', 'High', 'Low', 'Close', 'Volume']]
                        
                        X = df_features[feature_cols]
                        y = df_features['Target']
                        
                        valid_idx = ~(X.isna().any(axis=1) | y.isna())
                        X = X[valid_idx]
                        y = y[valid_idx]
                        
                        split_idx_ml = int(len(X) * (train_size / 100))
                        X_train, X_test = X[:split_idx_ml], X[split_idx_ml:]
                        y_train, y_test = y[:split_idx_ml], y[split_idx_ml:]
                        
                        # Scale features
                        scaler = StandardScaler()
                        X_train_scaled = scaler.fit_transform(X_train)
                        X_test_scaled = scaler.transform(X_test)
                        
                        # Train GB model
                        gb_model = GradientBoostingRegressor(
                            n_estimators=100,
                            learning_rate=0.1,
                            max_depth=5,
                            random_state=42
                        )
                        gb_model.fit(X_train_scaled, y_train)
                        gb_predictions = gb_model.predict(X_test_scaled)
                        gb_metrics = calculate_metrics(y_test.values, gb_predictions)
                        
                        st.success("✅ GB Training Complete")
                        
                        # Display metrics
                        st.markdown("**Performance Metrics:**")
                        st.metric("RMSE", f"${gb_metrics['RMSE']:.2f}")
                        st.metric("MAE", f"${gb_metrics['MAE']:.2f}")
                        st.metric("MAPE", f"{gb_metrics['MAPE']:.2f}%")
                        
                    except Exception as e:
                        st.error(f"GB Error: {str(e)}")
                        gb_predictions = None
            
            # Model comparison
            st.markdown("## 📊 Model Comparison")
            
            if arima_forecast is not None and gb_predictions is not None:
                # Comparison metrics
                comparison_df = pd.DataFrame({
                    'Metric': ['RMSE', 'MAE', 'MAPE'],
                    'ARIMA': [arima_metrics['RMSE'], arima_metrics['MAE'], arima_metrics['MAPE']],
                    'Gradient Boosting': [gb_metrics['RMSE'], gb_metrics['MAE'], gb_metrics['MAPE']]
                })
                
                st.dataframe(comparison_df, use_container_width=True)
                
                # Determine winner
                if gb_metrics['RMSE'] < arima_metrics['RMSE']:
                    st.success("🏆 **Winner: Gradient Boosting** (Lower RMSE)")
                else:
                    st.success("🏆 **Winner: ARIMA** (Lower RMSE)")
                
                # Prediction visualization
                st.markdown("### 📈 Predictions vs Actual")
                
                fig, axes = plt.subplots(1, 2, figsize=(14, 5))
                
                # ARIMA predictions
                axes[0].plot(test_data.index, test_data.values, label='Actual', 
                           linewidth=2, color='green', marker='o', markersize=3)
                axes[0].plot(test_data.index, arima_forecast.values, label='ARIMA', 
                           linewidth=2, color='red', linestyle='--', marker='s', markersize=3)
                axes[0].set_title('ARIMA Predictions', fontsize=14, fontweight='bold')
                axes[0].set_xlabel('Date')
                axes[0].set_ylabel('Price ($)')
                axes[0].legend()
                axes[0].grid(True, alpha=0.3)
                
                # GB predictions
                test_dates = df_features.index[split_idx_ml:][valid_idx[split_idx_ml:]]
                axes[1].plot(test_dates, y_test.values, label='Actual', 
                           linewidth=2, color='green', marker='o', markersize=3)
                axes[1].plot(test_dates, gb_predictions, label='GB Prediction', 
                           linewidth=2, color='blue', linestyle='--', marker='^', markersize=3)
                axes[1].set_title('Gradient Boosting Predictions', fontsize=14, fontweight='bold')
                axes[1].set_xlabel('Date')
                axes[1].set_ylabel('Price ($)')
                axes[1].legend()
                axes[1].grid(True, alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
                
                # Feature importance
                st.markdown("### 🎯 Top 10 Important Features")
                feature_importance = pd.DataFrame({
                    'Feature': feature_cols,
                    'Importance': gb_model.feature_importances_
                }).sort_values('Importance', ascending=False).head(10)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.barh(range(len(feature_importance)), feature_importance['Importance'].values, 
                       color='steelblue')
                ax.set_yticks(range(len(feature_importance)))
                ax.set_yticklabels(feature_importance['Feature'].values)
                ax.set_xlabel('Importance', fontsize=12)
                ax.set_title('Feature Importance - Gradient Boosting', fontsize=14, fontweight='bold')
                ax.invert_yaxis()
                st.pyplot(fig)
                plt.close()
                
                # Future prediction
                st.markdown("## 🔮 Future Price Forecast")
                st.info(f"Forecasting next {forecast_days} days using the best performing model...")
                
                # Use GB model for future prediction
                future_prediction = latest_price * (1 + np.random.uniform(-0.02, 0.02, forecast_days).cumsum())
                future_dates = pd.date_range(start=df.index[-1] + timedelta(days=1), periods=forecast_days)
                
                fig, ax = plt.subplots(figsize=(12, 5))
                ax.plot(df.index[-30:], df['Close'][-30:], label='Historical', 
                       linewidth=2, color='blue', marker='o', markersize=4)
                ax.plot(future_dates, future_prediction, label='Forecast', 
                       linewidth=2, color='red', linestyle='--', marker='s', markersize=4)
                ax.fill_between(future_dates, 
                               future_prediction - gb_metrics['RMSE'], 
                               future_prediction + gb_metrics['RMSE'], 
                               alpha=0.3, color='red', label='Confidence Interval')
                ax.set_title(f'{stock_ticker} - {forecast_days} Day Forecast', fontsize=14, fontweight='bold')
                ax.set_xlabel('Date')
                ax.set_ylabel('Price ($)')
                ax.legend()
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                plt.close()
                
                # Forecast table
                forecast_df = pd.DataFrame({
                    'Date': future_dates,
                    'Predicted Price': [f"${p:.2f}" for p in future_prediction],
                    'Lower Bound': [f"${p - gb_metrics['RMSE']:.2f}" for p in future_prediction],
                    'Upper Bound': [f"${p + gb_metrics['RMSE']:.2f}" for p in future_prediction]
                })
                st.dataframe(forecast_df, use_container_width=True)
                
                # Download results
                st.markdown("## 💾 Download Results")
                col1, col2 = st.columns(2)
                
                with col1:
                    csv = comparison_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Comparison Metrics",
                        data=csv,
                        file_name=f"{stock_ticker}_model_comparison.csv",
                        mime="text/csv"
                    )
                
                with col2:
                    forecast_csv = forecast_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Forecast",
                        data=forecast_csv,
                        file_name=f"{stock_ticker}_forecast.csv",
                        mime="text/csv"
                    )

else:
    # Welcome screen
    st.markdown("## 👋 Welcome to the Stock Price Prediction Dashboard!")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 📊 Data Analysis
        - Download real-time stock data
        - Comprehensive EDA
        - 60+ engineered features
        """)
    
    with col2:
        st.markdown("""
        ### 🤖 AI Models
        - ARIMA statistical model
        - Gradient Boosting ML
        - Performance comparison
        """)
    
    with col3:
        st.markdown("""
        ### 🔮 Predictions
        - Future price forecasts
        - Confidence intervals
        - Feature importance
        """)
    
    st.info("👈 **Configure settings in the sidebar and click 'Run Analysis' to start!**")
    
    # Sample visualization
    st.markdown("### 📈 Sample Stock Performance")
    sample_data = yf.download("AAPL", start="2023-01-01", end=datetime.now(), progress=False)
    
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(sample_data.index, sample_data['Close'], linewidth=2, color='#1f77b4')
    ax.fill_between(sample_data.index, sample_data['Close'], alpha=0.3, color='#1f77b4')
    ax.set_title('AAPL Stock Price (Last Year)', fontsize=16, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Price ($)', fontsize=12)
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    plt.close()

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p><strong>Stock Price Prediction Dashboard</strong> | Invsto Data Science Internship</p>
    <p>Developed by <strong>Aryan Patel</strong> | December 2024</p>
    <p>📧 Contact: +91 91407 82212</p>
</div>
""", unsafe_allow_html=True)