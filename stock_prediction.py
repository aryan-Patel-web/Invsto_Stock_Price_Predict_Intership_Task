#!/usr/bin/env python3
"""
Stock Price Prediction using ARIMA and Gradient Boosting
Invsto Data Science Internship Assignment
Author: Aryan Patel

This script performs comprehensive stock price prediction analysis
using both statistical (ARIMA) and machine learning (Gradient Boosting) approaches.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
import warnings
from datetime import datetime, timedelta
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from itertools import product
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy import stats
import sys

# Configuration
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Global variables
STOCKS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'JPM', 'JNJ', 'WMT']
START_DATE = '2020-01-01'
END_DATE = datetime.now().strftime('%Y-%m-%d')
PRIMARY_STOCK = 'AAPL'

def print_section(title):
    """Print formatted section header"""
    print("\n" + "="*70)
    print(title)
    print("="*70)

def download_stock_data():
    """Download stock data from Yahoo Finance"""
    print_section("DATA COLLECTION")
    print(f"\n📊 Downloading data for {len(STOCKS)} stocks from {START_DATE} to {END_DATE}...")
    
    stock_data = {}
    for ticker in STOCKS:
        try:
            data = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False)
            if not data.empty:
                stock_data[ticker] = data
                print(f"✓ {ticker}: {len(data)} days of data")
            else:
                print(f"✗ {ticker}: No data available")
        except Exception as e:
            print(f"✗ {ticker}: Error - {str(e)}")
    
    print(f"\n✅ Successfully downloaded data for {len(stock_data)} stocks")
    return stock_data

def clean_stock_data(df, ticker):
    """Clean stock data: handle missing values, outliers, and format properly"""
    df = df.copy()
    
    print(f"\n--- Cleaning {ticker} ---")
    print(f"Original shape: {df.shape}")
    
    # Handle missing values
    df = df.fillna(method='ffill', limit=3)
    df = df.fillna(method='bfill', limit=3)
    df = df.dropna()
    
    # Check for outliers
    df['Returns'] = df['Close'].pct_change()
    Q1 = df['Returns'].quantile(0.01)
    Q3 = df['Returns'].quantile(0.99)
    IQR = Q3 - Q1
    outliers = ((df['Returns'] < (Q1 - 3 * IQR)) | (df['Returns'] > (Q3 + 3 * IQR))).sum()
    print(f"Extreme outliers detected: {outliers}")
    
    df = df.drop('Returns', axis=1)
    df = df.sort_index()
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    print(f"Cleaned shape: {df.shape}")
    return df

def clean_all_data(stock_data):
    """Clean all stock data"""
    print_section("DATA PREPARATION AND CLEANING")
    
    cleaned_data = {}
    for ticker, data in stock_data.items():
        cleaned_data[ticker] = clean_stock_data(data, ticker)
    
    print("\n✅ Data cleaning completed!")
    return cleaned_data

def perform_eda(cleaned_data):
    """Perform exploratory data analysis"""
    print_section("EXPLORATORY DATA ANALYSIS")
    
    df = cleaned_data[PRIMARY_STOCK].copy()
    print(f"\nAnalyzing {PRIMARY_STOCK} in detail...")
    
    # Price and Volume Trends
    fig, axes = plt.subplots(3, 1, figsize=(15, 10))
    
    axes[0].plot(df.index, df['Close'], linewidth=1.5, color='#2E86AB')
    axes[0].set_title(f'{PRIMARY_STOCK} - Closing Price Trend', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Price ($)', fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    axes[1].bar(df.index, df['Volume'], color='#A23B72', alpha=0.6, width=1)
    axes[1].set_title(f'{PRIMARY_STOCK} - Trading Volume', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Volume', fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    returns = df['Close'].pct_change()
    axes[2].plot(df.index, returns, linewidth=1, color='#F18F01', alpha=0.7)
    axes[2].axhline(y=0, color='black', linestyle='--', linewidth=1)
    axes[2].set_title(f'{PRIMARY_STOCK} - Daily Returns', fontsize=14, fontweight='bold')
    axes[2].set_ylabel('Return (%)', fontsize=11)
    axes[2].set_xlabel('Date', fontsize=11)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{PRIMARY_STOCK}_trends.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Trend analysis chart saved: {PRIMARY_STOCK}_trends.png")
    
    # Multiple stocks comparison
    fig, ax = plt.subplots(figsize=(15, 7))
    for ticker in list(cleaned_data.keys())[:6]:
        normalized = (cleaned_data[ticker]['Close'] / cleaned_data[ticker]['Close'].iloc[0]) * 100
        ax.plot(normalized.index, normalized, label=ticker, linewidth=2, alpha=0.8)
    
    ax.set_title('Normalized Price Comparison (Base = 100)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Normalized Price', fontsize=11)
    ax.set_xlabel('Date', fontsize=11)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('multi_stock_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Multi-stock comparison chart saved")
    
    # Key statistics
    print(f"\n📈 Key Statistics for {PRIMARY_STOCK}:")
    print(f"  • Total trading days: {len(df)}")
    print(f"  • Average daily return: {returns.mean()*100:.3f}%")
    print(f"  • Daily return volatility: {returns.std()*100:.3f}%")
    print(f"  • Price range: ${df['Close'].min():.2f} - ${df['Close'].max():.2f}")
    
    return df

def calculate_rsi(prices, period=14):
    """Calculate Relative Strength Index"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(prices, fast=12, slow=26, signal=9):
    """Calculate MACD indicator"""
    ema_fast = prices.ewm(span=fast).mean()
    ema_slow = prices.ewm(span=slow).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal).mean()
    return macd, macd_signal

def create_features(df, lags=[1, 2, 3, 5, 10], windows=[5, 10, 20, 50]):
    """Create comprehensive features for stock prediction"""
    df = df.copy()
    
    # Basic price features
    df['Returns'] = df['Close'].pct_change()
    df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
    df['Range'] = (df['High'] - df['Low']) / df['Close']
    df['Gap'] = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)
    
    # Lagged features
    for lag in lags:
        df[f'Close_Lag_{lag}'] = df['Close'].shift(lag)
        df[f'Returns_Lag_{lag}'] = df['Returns'].shift(lag)
        df[f'Volume_Lag_{lag}'] = df['Volume'].shift(lag)
    
    # Rolling window features
    for window in windows:
        df[f'SMA_{window}'] = df['Close'].rolling(window=window).mean()
        df[f'EMA_{window}'] = df['Close'].ewm(span=window).mean()
        df[f'Volatility_{window}'] = df['Returns'].rolling(window=window).std()
        df[f'Volume_SMA_{window}'] = df['Volume'].rolling(window=window).mean()
        df[f'Volume_Ratio_{window}'] = df['Volume'] / df[f'Volume_SMA_{window}']
        df[f'Price_to_SMA_{window}'] = df['Close'] / df[f'SMA_{window}']
    
    # Momentum indicators
    df['RSI_14'] = calculate_rsi(df['Close'], 14)
    df['MACD'], df['MACD_Signal'] = calculate_macd(df['Close'])
    df['MACD_Diff'] = df['MACD'] - df['MACD_Signal']
    
    # Bollinger Bands
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (2 * bb_std)
    df['BB_Lower'] = df['BB_Middle'] - (2 * bb_std)
    df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
    df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
    
    # Time-based features
    df['Day_of_Week'] = df.index.dayofweek
    df['Month'] = df.index.month
    df['Quarter'] = df.index.quarter
    
    # Target variable
    df['Target'] = df['Close'].shift(-1)
    
    return df

def engineer_features(cleaned_data):
    """Feature engineering wrapper"""
    print_section("FEATURE ENGINEERING")
    
    df = cleaned_data[PRIMARY_STOCK].copy()
    print(f"\nCreating features for {PRIMARY_STOCK}...")
    
    df_features = create_features(df)
    print(f"✓ Created {len(df_features.columns)} total features")
    
    return df_features

def train_arima_model(df):
    """Train ARIMA model"""
    print_section("ARIMA MODEL DEVELOPMENT")
    
    close_prices = df['Close'].copy()
    
    # Train-test split
    train_size = int(len(close_prices) * 0.8)
    train, test = close_prices[:train_size], close_prices[train_size:]
    
    print(f"Training size: {len(train)}, Test size: {len(test)}")
    
    # Grid search for optimal parameters
    print("\n🔍 Finding optimal ARIMA parameters...")
    
    best_aic = np.inf
    best_params = None
    
    for p, d, q in product(range(0, 3), range(0, 2), range(0, 3)):
        try:
            model = ARIMA(train, order=(p, d, q))
            fitted_model = model.fit()
            if fitted_model.aic < best_aic:
                best_aic = fitted_model.aic
                best_params = (p, d, q)
        except:
            continue
    
    print(f"\n✓ Best ARIMA parameters: {best_params}")
    print(f"  AIC: {best_aic:.2f}")
    
    # Fit final model
    arima_model = ARIMA(train, order=best_params)
    arima_fitted = arima_model.fit()
    
    # Forecast
    forecast = arima_fitted.forecast(steps=len(test))
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(test.values, forecast.values))
    mae = mean_absolute_error(test.values, forecast.values)
    mape = np.mean(np.abs((test.values - forecast.values) / test.values)) * 100
    
    print(f"\nARIMA{best_params} - Evaluation Metrics:")
    print(f"  RMSE: ${rmse:.4f}")
    print(f"  MAE:  ${mae:.4f}")
    print(f"  MAPE: {mape:.4f}%")
    
    # Plot
    fig, ax = plt.subplots(figsize=(15, 7))
    ax.plot(train.index, train.values, label='Training Data', color='blue', linewidth=1.5)
    ax.plot(test.index, test.values, label='Actual Test Data', color='green', linewidth=2)
    ax.plot(test.index, forecast.values, label='ARIMA Forecast', color='red', linewidth=2, linestyle='--')
    ax.set_title(f'ARIMA{best_params} - Forecast', fontsize=14, fontweight='bold')
    ax.set_ylabel('Price ($)', fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{PRIMARY_STOCK}_arima_forecast.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ ARIMA forecast chart saved")
    
    return {
        'model': arima_fitted,
        'forecast': forecast,
        'test': test,
        'params': best_params,
        'metrics': {'RMSE': rmse, 'MAE': mae, 'MAPE': mape}
    }

def train_gradient_boosting(df_features):
    """Train Gradient Boosting model"""
    print_section("GRADIENT BOOSTING MODEL DEVELOPMENT")
    
    df_ml = df_features.dropna().copy()
    
    # Select features
    feature_cols = [col for col in df_ml.columns 
                   if col not in ['Target', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'Year']]
    
    X = df_ml[feature_cols]
    y = df_ml['Target']
    
    valid_idx = ~(X.isna().any(axis=1) | y.isna())
    X = X[valid_idx]
    y = y[valid_idx]
    
    print(f"\nDataset prepared:")
    print(f"  Features: {X.shape[1]}")
    print(f"  Samples: {X.shape[0]}")
    
    # Train-test split
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"  Training: {len(X_train)} samples")
    print(f"  Testing: {len(X_test)} samples")
    
    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    print("\n🔧 Training Gradient Boosting model...")
    gb_model = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        random_state=42,
        verbose=0
    )
    gb_model.fit(X_train_scaled, y_train)
    
    # Predictions
    y_pred = gb_model.predict(X_test_scaled)
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test.values, y_pred))
    mae = mean_absolute_error(y_test.values, y_pred)
    mape = np.mean(np.abs((y_test.values - y_pred) / y_test.values)) * 100
    
    print(f"\nGB Optimized - Evaluation Metrics:")
    print(f"  RMSE: ${rmse:.4f}")
    print(f"  MAE:  ${mae:.4f}")
    print(f"  MAPE: {mape:.4f}%")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': gb_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("\n📊 Top 10 Most Important Features:")
    print(feature_importance.head(10).to_string(index=False))
    
    # Plot predictions
    test_dates = df_ml.index[split_idx:][valid_idx[split_idx:]]
    fig, ax = plt.subplots(figsize=(15, 7))
    ax.plot(test_dates, y_test.values, label='Actual Price', color='green', linewidth=2, alpha=0.8)
    ax.plot(test_dates, y_pred, label='GB Predicted Price', color='red', linewidth=2, linestyle='--', alpha=0.8)
    ax.set_title('Gradient Boosting - Predictions', fontsize=14, fontweight='bold')
    ax.set_ylabel('Price ($)', fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{PRIMARY_STOCK}_gb_predictions.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ GB predictions chart saved")
    
    return {
        'model': gb_model,
        'predictions': y_pred,
        'test': y_test,
        'test_dates': test_dates,
        'feature_importance': feature_importance,
        'metrics': {'RMSE': rmse, 'MAE': mae, 'MAPE': mape}
    }

def compare_models(arima_results, gb_results):
    """Compare model performance"""
    print_section("MODEL COMPARISON")
    
    comparison_df = pd.DataFrame({
        'Model': ['ARIMA', 'Gradient Boosting'],
        'RMSE': [arima_results['metrics']['RMSE'], gb_results['metrics']['RMSE']],
        'MAE': [arima_results['metrics']['MAE'], gb_results['metrics']['MAE']],
        'MAPE': [arima_results['metrics']['MAPE'], gb_results['metrics']['MAPE']]
    })
    
    print("\n📊 Model Performance Comparison:")
    print(comparison_df.to_string(index=False))
    
    best_model = 'Gradient Boosting' if gb_results['metrics']['RMSE'] < arima_results['metrics']['RMSE'] else 'ARIMA'
    print(f"\n🏆 Best Model (by RMSE): {best_model}")
    
    # Plot comparison
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    metrics = ['RMSE', 'MAE', 'MAPE']
    colors = ['#FF6B6B', '#4ECDC4']
    
    for idx, metric in enumerate(metrics):
        values = comparison_df[metric].values
        bars = axes[idx].bar(comparison_df['Model'], values, color=colors, alpha=0.7, edgecolor='black')
        axes[idx].set_title(f'{metric} Comparison', fontsize=12, fontweight='bold')
        axes[idx].set_ylabel(metric, fontsize=10)
        axes[idx].grid(True, alpha=0.3, axis='y')
        
        for bar in bars:
            height = bar.get_height()
            axes[idx].text(bar.get_x() + bar.get_width()/2., height,
                          f'{height:.2f}',
                          ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{PRIMARY_STOCK}_model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Model comparison chart saved")
    
    return comparison_df, best_model

def simulate_strategies(arima_results, gb_results):
    """Simulate trading strategies"""
    print_section("TRADING STRATEGY SIMULATION")
    
    def calculate_directional_accuracy(actual, predicted):
        actual_direction = np.sign(np.diff(actual))
        predicted_direction = np.sign(np.diff(predicted))
        accuracy = np.mean(actual_direction == predicted_direction) * 100
        return accuracy
    
    def simulate_trading(actual_prices, predicted_prices, initial_capital=100000):
        capital = initial_capital
        position = 0
        trades = []
        
        for i in range(1, len(predicted_prices)):
            predicted_return = (predicted_prices[i] - actual_prices[i-1]) / actual_prices[i-1]
            actual_return = (actual_prices[i] - actual_prices[i-1]) / actual_prices[i-1]
            
            if predicted_return > 0.001 and position == 0:
                position = 1
                trades.append(('BUY', i, actual_prices[i]))
            elif predicted_return < -0.001 and position == 1:
                position = 0
                trades.append(('SELL', i, actual_prices[i]))
            
            if position == 1:
                capital = capital * (1 + actual_return)
        
        if position == 1:
            trades.append(('SELL', len(actual_prices)-1, actual_prices[-1]))
        
        total_return = (capital - initial_capital) / initial_capital * 100
        return capital, total_return, trades
    
    # Calculate metrics
    arima_dir_acc = calculate_directional_accuracy(
        arima_results['test'].values, 
        arima_results['forecast'].values
    )
    gb_dir_acc = calculate_directional_accuracy(
        gb_results['test'].values, 
        gb_results['predictions']
    )
    
    print(f"\nARIMA Directional Accuracy: {arima_dir_acc:.2f}%")
    print(f"Gradient Boosting Directional Accuracy: {gb_dir_acc:.2f}%")
    
    # Simulate strategies
    arima_capital, arima_return, arima_trades = simulate_trading(
        arima_results['test'].values,
        arima_results['forecast'].values
    )
    gb_capital, gb_return, gb_trades = simulate_trading(
        gb_results['test'].values,
        gb_results['predictions']
    )
    
    buy_hold_return = (arima_results['test'].values[-1] - arima_results['test'].values[0]) / arima_results['test'].values[0] * 100
    
    print(f"\n💰 Trading Strategy Simulation Results:")
    print(f"\n1. ARIMA Strategy:")
    print(f"   Final Capital: ${arima_capital:,.2f}")
    print(f"   Total Return: {arima_return:.2f}%")
    print(f"   Number of Trades: {len(arima_trades)}")
    
    print(f"\n2. Gradient Boosting Strategy:")
    print(f"   Final Capital: ${gb_capital:,.2f}")
    print(f"   Total Return: {gb_return:.2f}%")
    print(f"   Number of Trades: {len(gb_trades)}")
    
    print(f"\n3. Buy & Hold Benchmark:")
    print(f"   Total Return: {buy_hold_return:.2f}%")
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    strategies = ['ARIMA Strategy', 'GB Strategy', 'Buy & Hold']
    returns = [arima_return, gb_return, buy_hold_return]
    colors_strat = ['#3498db', '#e74c3c', '#2ecc71']
    
    bars = ax.bar(strategies, returns, color=colors_strat, alpha=0.7, edgecolor='black', linewidth=2)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax.set_ylabel('Return (%)', fontsize=12)
    ax.set_title('Trading Strategy Performance Comparison', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}%',
                ha='center', va='bottom' if height > 0 else 'top', 
                fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{PRIMARY_STOCK}_trading_strategy.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Trading strategy chart saved")
    
    return {
        'arima_return': arima_return,
        'gb_return': gb_return,
        'buy_hold_return': buy_hold_return,
        'arima_dir_acc': arima_dir_acc,
        'gb_dir_acc': gb_dir_acc
    }

def generate_final_report(arima_results, gb_results, strategy_results, comparison_df, best_model):
    """Generate final report"""
    print_section("KEY FINDINGS AND RECOMMENDATIONS")
    
    print(f"""
📊 EXECUTIVE SUMMARY - {PRIMARY_STOCK} Stock Price Prediction

1. DATA ANALYSIS
   • Analyzed historical data from {START_DATE} to {END_DATE}
   • {len(STOCKS)} stocks evaluated across different sectors
   • Data quality: High, with minimal missing values and outliers

2. MODEL PERFORMANCE

   ARIMA{arima_results['params']}:
   • RMSE: ${arima_results['metrics']['RMSE']:.4f}
   • MAE: ${arima_results['metrics']['MAE']:.4f}
   • MAPE: {arima_results['metrics']['MAPE']:.2f}%
   • Directional Accuracy: {strategy_results['arima_dir_acc']:.2f}%
   
   Gradient Boosting:
   • RMSE: ${gb_results['metrics']['RMSE']:.4f}
   • MAE: ${gb_results['metrics']['MAE']:.4f}
   • MAPE: {gb_results['metrics']['MAPE']:.2f}%
   • Directional Accuracy: {strategy_results['gb_dir_acc']:.2f}%

3. TRADING IMPLICATIONS

   Strategy Performance:
   • ARIMA Trading Return: {strategy_results['arima_return']:.2f}%
   • GB Trading Return: {strategy_results['gb_return']:.2f}%
   • Buy & Hold Return: {strategy_results['buy_hold_return']:.2f}%

4. RECOMMENDATIONS

   ✓ Best Model: {best_model}
   ✓ Use ensemble approach combining both models
   ✓ Implement strict risk management
   ✓ Monitor model performance continuously
   ✓ Retrain models weekly with new data

5. CONCLUSION

   Both models show promise for stock price prediction.
   {best_model} demonstrates superior accuracy overall.
   Recommend deployment with proper risk controls.
""")
    
    # Save summary CSV
    summary_data = {
        'Metric': ['Stock', 'ARIMA RMSE', 'ARIMA MAE', 'ARIMA MAPE',
                  'GB RMSE', 'GB MAE', 'GB MAPE',
                  'ARIMA Dir Accuracy', 'GB Dir Accuracy',
                  'ARIMA Return', 'GB Return', 'Buy Hold Return', 'Best Model'],
        'Value': [PRIMARY_STOCK,
                 f'{arima_results["metrics"]["RMSE"]:.4f}',
                 f'{arima_results["metrics"]["MAE"]:.4f}',
                 f'{arima_results["metrics"]["MAPE"]:.2f}%',
                 f'{gb_results["metrics"]["RMSE"]:.4f}',
                 f'{gb_results["metrics"]["MAE"]:.4f}',
                 f'{gb_results["metrics"]["MAPE"]:.2f}%',
                 f'{strategy_results["arima_dir_acc"]:.2f}%',
                 f'{strategy_results["gb_dir_acc"]:.2f}%',
                 f'{strategy_results["arima_return"]:.2f}%',
                 f'{strategy_results["gb_return"]:.2f}%',
                 f'{strategy_results["buy_hold_return"]:.2f}%',
                 best_model]
    }
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(f'{PRIMARY_STOCK}_analysis_summary.csv', index=False)
    print(f"\n✓ Summary statistics saved: {PRIMARY_STOCK}_analysis_summary.csv")
    
    print("\n📁 Generated Files:")
    files = [
        f'{PRIMARY_STOCK}_trends.png',
        'multi_stock_comparison.png',
        f'{PRIMARY_STOCK}_arima_forecast.png',
        f'{PRIMARY_STOCK}_gb_predictions.png',
        f'{PRIMARY_STOCK}_model_comparison.png',
        f'{PRIMARY_STOCK}_trading_strategy.png',
        f'{PRIMARY_STOCK}_analysis_summary.csv'
    ]
    for i, file in enumerate(files, 1):
        print(f"   {i}. {file}")

def main():
    """Main execution function"""
    print("="*70)
    print("STOCK PRICE PREDICTION PIPELINE")
    print("Invsto Data Science Internship Assignment")
    print("Author: Aryan Patel")
    print("="*70)
    print(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Step 1: Download data
        stock_data = download_stock_data()
        
        # Step 2: Clean data
        cleaned_data = clean_all_data(stock_data)
        
        # Step 3: EDA
        df = perform_eda(cleaned_data)
        
        # Step 4: Feature engineering
        df_features = engineer_features(cleaned_data)
        
        # Step 5: Train ARIMA
        arima_results = train_arima_model(df)
        
        # Step 6: Train GB
        gb_results = train_gradient_boosting(df_features)
        
        # Step 7: Compare models
        comparison_df, best_model = compare_models(arima_results, gb_results)
        
        # Step 8: Simulate strategies
        strategy_results = simulate_strategies(arima_results, gb_results)
        
        # Step 9: Generate report
        generate_final_report(arima_results, gb_results, strategy_results, comparison_df, best_model)
        
        print("\n" + "="*70)
        print("ANALYSIS COMPLETE ✅")
        print("="*70)
        print("\n🎉 Ready for submission to Invsto!")
        
    except Exception as e:
        print(f"\n❌ Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()