import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
import os

# Helper function to parse boolean environment variables
def str_to_bool(s):
    return str(s).lower() in ['true', '1', 'yes']
    
# Constants for configuration
RISK_MANAGEMENT_ENABLED = str_to_bool(os.environ.get('RISK_MANAGEMENT_ENABLED', 'true'))
TAKE_PROFIT_PERCENTAGE = float(os.environ.get('TAKE_PROFIT_PERCENTAGE', '10'))  # Percentage
TRADING_FREQUENCY = int(os.environ.get('TRADING_FREQUENCY', '15'))  # Minutes
ML_MODEL_UPDATE_FREQUENCY = int(os.environ.get('ML_MODEL_UPDATE_FREQUENCY', '24'))  # Hours
VOLATILITY_THRESHOLD = float(os.environ.get('VOLATILITY_THRESHOLD', '0'))  # Set to 0 to disable volatility filter
AUTO_TRADE_THRESHOLD = float(os.environ.get('AUTO_TRADE_THRESHOLD', '0.5'))  # Lowered threshold
PROFIT_REINVESTMENT_RATIO = float(os.environ.get('PROFIT_REINVESTMENT_RATIO', '0.5'))  # Ratio for reinvesting profits
DYNAMIC_POSITION_SIZING = str_to_bool(os.environ.get('DYNAMIC_POSITION_SIZING', 'true'))
TREND_FOLLOWING_STRENGTH = float(os.environ.get('TREND_FOLLOWING_STRENGTH', '0.0'))  # Reduced impact to zero
COUNTER_TREND_STRENGTH = str_to_bool(os.environ.get('COUNTER_TREND_STRENGTH', 'true'))  # Allow counter-trend trades
TRAILING_STOP_LOSS = str_to_bool(os.environ.get('TRAILING_STOP_LOSS', 'true'))
ADAPTIVE_STOP_LOSS = str_to_bool(os.environ.get('ADAPTIVE_STOP_LOSS', 'true'))
SENTIMENT_ANALYSIS_WEIGHT = float(os.environ.get('SENTIMENT_ANALYSIS_WEIGHT', '0.0'))  # Set to zero to remove impact

# Symbol to Name Mapping
SYMBOL_NAME_MAPPING = {
    # Commodities
    "GC=F": "Gold",
    "SI=F": "Silver",
    "NG=F": "Natural Gas",
    "KC=F": "Coffee",

    # Forex
    "EURUSD=X": "EUR/USD",
    "USDJPY=X": "USD/JPY",
    "GBPUSD=X": "GBP/USD",
    "AUDUSD=X": "AUD/USD",
    "USDCAD=X": "USD/CAD",

    # Crypto
    "BTC-USD": "Bitcoin",
    "ETH-USD": "Ethereum",
    "XRP-USD": "Ripple",
    "ADA-USD": "Cardano",
    "DOT-USD": "Polkadot",
    "LINK-USD": "Chainlink",
    "LTC-USD": "Litecoin",
}

# Constants
COMMODITIES = ["GC=F", "SI=F", "NG=F", "KC=F"]
FOREX_SYMBOLS = ["EURUSD=X", "USDJPY=X", "GBPUSD=X", "AUDUSD=X", "USDCAD=X"]
CRYPTO_SYMBOLS = ["BTC-USD", "ETH-USD", "XRP-USD",
                  "ADA-USD", "DOT-USD", "LINK-USD", "LTC-USD"]

# Initialize session state variables
if 'positions' not in st.session_state:
    st.session_state.positions = {}
if 'trade_history' not in st.session_state:
    st.session_state.trade_history = []
if 'stop_loss' not in st.session_state:
    st.session_state.stop_loss = {}
if 'take_profit' not in st.session_state:
    st.session_state.take_profit = {}
if 'last_model_update' not in st.session_state:
    st.session_state.last_model_update = None
if 'models' not in st.session_state:
    st.session_state.models = {}
if 'model_accuracies' not in st.session_state:
    st.session_state.model_accuracies = {}
if 'balance_history' not in st.session_state:
    st.session_state.balance_history = []
if 'asset_type' not in st.session_state:
    st.session_state.asset_type = "Commodities"
if 'TRADING_MODE' not in st.session_state:
    st.session_state.TRADING_MODE = "AUTO"

def get_data(symbol):
    try:
        data = yf.download(symbol, period='1y', interval='1d', progress=False)
        return data
    except Exception as e:
        st.warning(f"Failed to download data for {SYMBOL_NAME_MAPPING.get(symbol, symbol)}: {e}")
        return pd.DataFrame()

def calculate_signals(data):
    """Calculate technical indicators and generate signals."""
    if data.empty:
        return pd.DataFrame()

    data = data.copy()
    data.sort_index(inplace=True)

    # Add technical indicators
    # Moving Averages
    data['SMA20'] = data['Close'].rolling(window=20).mean()
    data['SMA50'] = data['Close'].rolling(window=50).mean()
    data['EMA20'] = data['Close'].ewm(span=20, adjust=False).mean()

    # RSI
    delta = data['Close'].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    up_avg = up.rolling(window=14).mean()
    down_avg = down.rolling(window=14).mean()
    epsilon = 1e-6  # Small value to prevent division by zero
    data['RSI'] = 100 - (100 / (1 + (up_avg / (down_avg + epsilon))))

    # MACD
    exp1 = data['Close'].ewm(span=12, adjust=False).mean()
    exp2 = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = exp1 - exp2
    data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()

    # Bollinger Bands
    data['BB_Mid'] = data['Close'].rolling(window=20).mean()
    data['BB_Std'] = data['Close'].rolling(window=20).std()
    data['BB_Upper'] = data['BB_Mid'] + (2 * data['BB_Std'])
    data['BB_Lower'] = data['BB_Mid'] - (2 * data['BB_Std'])

    # Feature Engineering
    data['Price_Change'] = data['Close'].pct_change()
    data['Volatility'] = data['Price_Change'].rolling(window=10).std()

    # Drop NaN values
    data.dropna(inplace=True)

    return data

def perform_sentiment_analysis(symbol):
    """Dummy sentiment analysis function."""
    np.random.seed(hash(symbol) % 123456789)  # Seed with symbol hash for consistency
    sentiment_score = np.random.uniform(-1, 1)
    return sentiment_score

def prepare_ml_data(data):
    """Prepare data for machine learning."""
    if data.empty:
        return None, None

    data = data.copy()
    data['Future_Close'] = data['Close'].shift(-1)
    data['Target'] = np.where(data['Future_Close'] > data['Close'], 1, 0)
    data.dropna(inplace=True)

    # Features excluding 'Close' to prevent data leakage
    features = ['SMA20', 'SMA50', 'EMA20', 'RSI', 'MACD',
                'Signal_Line', 'BB_Upper', 'BB_Lower', 'Volatility']
    if not all(feature in data.columns for feature in features):
        return None, None  # Missing features

    X = data[features]
    y = data['Target']

    return X, y

def train_ml_model(X, y):
    """Train the XGBoost model."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False)
    model = XGBClassifier(eval_metric='logloss', use_label_encoder=False)
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    accuracy = report['accuracy']

    return model, accuracy

def execute_trade(symbol, action, price, amount):
    """Execute buy or sell orders."""
    if st.session_state.TRADING_MODE == 'MANUAL':
        if st.sidebar.button(f"Approve {action} {SYMBOL_NAME_MAPPING.get(symbol, symbol)} at ${price:.2f}?"):
            approved = True
        else:
            st.sidebar.write(f"Awaiting approval to {action} {SYMBOL_NAME_MAPPING.get(symbol, symbol)} at ${price:.2f}")
            return False
    else:
        approved = True

    if approved:
        if action == 'Buy':
            total_cost = amount * price
            if st.session_state.balance >= total_cost:
                st.session_state.balance -= total_cost
                if symbol in st.session_state.positions:
                    previous_quantity = st.session_state.positions[symbol]['quantity']
                    previous_cost_basis = st.session_state.positions[symbol]['cost_basis']
                    new_quantity = previous_quantity + amount
                    new_cost_basis = ((previous_cost_basis * previous_quantity) + (price * amount)) / new_quantity
                    st.session_state.positions[symbol]['quantity'] = new_quantity
                    st.session_state.positions[symbol]['cost_basis'] = new_cost_basis
                else:
                    st.session_state.positions[symbol] = {
                        'quantity': amount, 'cost_basis': price}
                st.session_state.trade_history.append({
                    'Date': datetime.now(),
                    'Symbol': SYMBOL_NAME_MAPPING.get(symbol, symbol),
                    'Action': action,
                    'Price': price,
                    'Amount': amount
                })

                # Risk management: Set stop-loss and take-profit
                if RISK_MANAGEMENT_ENABLED:
                    if ADAPTIVE_STOP_LOSS:
                        # Adjust stop-loss based on volatility
                        volatility = st.session_state.latest_volatility.get(symbol, 0.02)
                        st.session_state.stop_loss[symbol] = price * (1 - volatility)
                    else:
                        st.session_state.stop_loss[symbol] = price * \
                            (1 - st.session_state.user_stop_loss_pct)
                    st.session_state.take_profit[symbol] = price * \
                        (1 + TAKE_PROFIT_PERCENTAGE / 100)

                st.session_state.balance_history.append(
                    {'Date': datetime.now(), 'Balance': st.session_state.balance})
                st.success(f"Bought {amount:.4f} of {SYMBOL_NAME_MAPPING.get(symbol, symbol)} at ${price:.2f}")
                return True
            else:
                st.warning(f"Insufficient balance to execute trade for {SYMBOL_NAME_MAPPING.get(symbol, symbol)}.")
                return False
        elif action == 'Sell':
            if symbol in st.session_state.positions:
                available_quantity = st.session_state.positions[symbol]['quantity']
                amount_to_sell = min(amount, available_quantity)
                st.session_state.balance += amount_to_sell * price
                st.session_state.positions[symbol]['quantity'] -= amount_to_sell
                if st.session_state.positions[symbol]['quantity'] == 0:
                    del st.session_state.positions[symbol]
                    if symbol in st.session_state.stop_loss:
                        del st.session_state.stop_loss[symbol]
                    if symbol in st.session_state.take_profit:
                        del st.session_state.take_profit[symbol]
                st.session_state.trade_history.append({
                    'Date': datetime.now(),
                    'Symbol': SYMBOL_NAME_MAPPING.get(symbol, symbol),
                    'Action': action,
                    'Price': price,
                    'Amount': amount_to_sell
                })
                st.session_state.balance_history.append(
                    {'Date': datetime.now(), 'Balance': st.session_state.balance})
                st.success(f"Sold {amount_to_sell:.4f} of {SYMBOL_NAME_MAPPING.get(symbol, symbol)} at ${price:.2f}")
                return True
            else:
                st.warning(f"No holdings to sell for {SYMBOL_NAME_MAPPING.get(symbol, symbol)}.")
                return False

def check_stop_loss(symbol, current_price):
    """Check if the stop-loss or take-profit condition is met."""
    if symbol in st.session_state.stop_loss and current_price <= st.session_state.stop_loss[symbol]:
        amount = st.session_state.positions[symbol]['quantity']
        if execute_trade(symbol, 'Sell', current_price, amount):
            st.warning(
                f"Stop-loss triggered for {SYMBOL_NAME_MAPPING.get(symbol, symbol)}. Sold {amount:.4f} at ${current_price:.2f}")
    elif symbol in st.session_state.take_profit and current_price >= st.session_state.take_profit[symbol]:
        amount = st.session_state.positions[symbol]['quantity']
        if execute_trade(symbol, 'Sell', current_price, amount):
            st.success(
                f"Take-profit reached for {SYMBOL_NAME_MAPPING.get(symbol, symbol)}. Sold {amount:.4f} at ${current_price:.2f}")
    # Trailing Stop-Loss Adjustment
    if TRAILING_STOP_LOSS and symbol in st.session_state.positions:
        last_stop = st.session_state.stop_loss.get(symbol, 0)
        potential_stop = current_price * (1 - st.session_state.user_stop_loss_pct)
        if potential_stop > last_stop:
            st.session_state.stop_loss[symbol] = potential_stop

def calculate_portfolio_metrics():
    """Calculate ROI and Max Drawdown."""
    total_value = st.session_state.balance
    for symbol, position in st.session_state.positions.items():
        data = yf.download(symbol, period='1d', progress=False)
        if not data.empty:
            current_price = data['Close'].iloc[-1]
            total_value += position['quantity'] * current_price

    initial_balance = st.session_state.initial_balance
    roi = (total_value - initial_balance) / initial_balance * 100

    portfolio_series = pd.Series([entry['Balance']
                                  for entry in st.session_state.balance_history])
    cumulative_max = portfolio_series.cummax()
    drawdowns = (cumulative_max - portfolio_series) / cumulative_max
    max_drawdown = drawdowns.max() * 100

    return roi, max_drawdown

def calculate_profit_loss(symbol, current_price):
    """Calculate profit or loss for a position."""
    if symbol in st.session_state.positions:
        position = st.session_state.positions[symbol]
        quantity = position['quantity']
        cost_basis = position['cost_basis']
        profit_loss = (current_price - cost_basis) * quantity
        return profit_loss
    else:
        return 0.0

def plot_balance_history():
    """Plot the account balance over time."""
    df = pd.DataFrame(st.session_state.balance_history)
    if df.empty:
        return None
    fig = px.line(df, x='Date', y='Balance', title='Account Balance History')
    return fig

def main():
    # Set page configuration
    st.set_page_config(
        page_title="Advanced Multi-Asset Trading Bot",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom CSS for styling
    st.markdown(
        """
        <style>
        /* Background gradient */
        .reportview-container {
            background: linear-gradient(to bottom right, #f0f2f6, #c0c4c7);
        }
        /* Sidebar styling */
        .sidebar .sidebar-content {
            background: linear-gradient(to bottom, #4b79a1, #283e51);
            color: white;
        }
        /* Header styling */
        .main .block-container {
            padding-top: 0rem;
        }
        .stButton>button {
            color: white !important;
            background-color: #4CAF50 !important;
        }
        /* Dataframe styling */
        .css-1d391kg edgvbvh3 {
            font-size: 20px;
        }
        /* Content styling */
        h2 {
            color: #4B79A1;
            font-size: 28px;
        }
        h3 {
            color: #283E51;
            font-size: 24px;
        }
        .stDataFrame th, .stDataFrame td {
            font-size: 16px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # App title with logo
    st.markdown(
        """
        <div style='text-align: center;'>
            <h1 style='color: #4B79A1;'>Advanced Multi-Asset Trading Bot</h1>
            <p style='font-size:20px; color: #283E51;'>Empowered by Machine Learning and Technical Analysis</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.sidebar.title("Bot Settings")

    # Configurable initial balance
    if 'initial_balance' not in st.session_state:
        st.sidebar.subheader("Initial Balance")
        initial_balance = st.sidebar.number_input(
            "Set Initial Balance", min_value=1000, value=100000, step=1000)
        st.session_state.initial_balance = initial_balance
        st.session_state.balance = initial_balance
        st.session_state.balance_history = [
            {'Date': datetime.now(), 'Balance': st.session_state.balance}]
    else:
        st.sidebar.subheader("Initial Balance")
        st.sidebar.write(f"${st.session_state.initial_balance:,.2f}")

    # Risk management settings
    st.sidebar.subheader("Risk Management Settings")
    st.session_state.user_stop_loss_pct = st.sidebar.slider(
        "Stop-Loss Percentage", min_value=0.01, max_value=0.2, value=0.05, step=0.01)

    # Trading mode selection as tabs
    st.sidebar.subheader("Trading Mode")
    trading_modes = ["AUTO", "MANUAL"]
    st.session_state.TRADING_MODE = st.sidebar.radio("", trading_modes, index=trading_modes.index(st.session_state.TRADING_MODE))

    # Asset selection as tabs
    st.sidebar.subheader("Asset Selection")
    asset_types = ["Commodities", "Forex", "Crypto"]
    st.session_state.asset_type = st.sidebar.radio("", asset_types, index=asset_types.index(st.session_state.asset_type))

    if st.session_state.asset_type == "Commodities":
        symbols = COMMODITIES
    elif st.session_state.asset_type == "Forex":
        symbols = FOREX_SYMBOLS
    else:
        symbols = CRYPTO_SYMBOLS

    # Trade amount
    st.sidebar.subheader("Trade Amount Settings")
    if DYNAMIC_POSITION_SIZING:
        base_trade_amount = st.sidebar.number_input(
            "Base Trade Amount per Asset", min_value=0.01, value=1.0, step=0.01)
    else:
        trade_amount = st.sidebar.number_input(
            "Trade Amount per Asset", min_value=0.01, value=1.0, step=0.01)

    # Automatically run analysis on selection change
    run_analysis = True  # Set to True to always run analysis

    if run_analysis:
        all_data = {}
        signals_list = []
        model_accuracies = []
        st.session_state.latest_volatility = {}

        progress_bar = st.progress(0)
        total_symbols = len(symbols)

        # Check if need to update ML model
        current_time = datetime.now()
        if st.session_state.last_model_update is None or \
           (current_time - st.session_state.last_model_update).total_seconds() >= ML_MODEL_UPDATE_FREQUENCY * 3600:
            update_ml_model = True
            st.session_state.last_model_update = current_time
            # Clear cached models
            st.session_state.models = {}
            st.session_state.model_accuracies = {}
        else:
            update_ml_model = False

        for idx, symbol in enumerate(symbols):
            try:
                data = get_data(symbol)
                if data.empty:
                    st.warning(f"No data retrieved for {SYMBOL_NAME_MAPPING.get(symbol, symbol)}. Skipping.")
                    continue
                data = calculate_signals(data)
                if data.empty:
                    st.warning(f"No data after signal calculation for {SYMBOL_NAME_MAPPING.get(symbol, symbol)}. Skipping.")
                    continue

                # Ensure that data has enough rows
                if len(data) < 50:
                    st.warning(f"Not enough data for {SYMBOL_NAME_MAPPING.get(symbol, symbol)}. Skipping.")
                    continue

                st.session_state.latest_volatility[symbol] = data['Volatility'].iloc[-1]

                X, y = prepare_ml_data(data)
                if X is None or y is None or X.empty:
                    st.warning(
                        f"Skipping {SYMBOL_NAME_MAPPING.get(symbol, symbol)} due to insufficient data or missing features.")
                    continue

                if update_ml_model or symbol not in st.session_state.models:
                    model, accuracy = train_ml_model(X, y)
                    st.session_state.models[symbol] = model
                    st.session_state.model_accuracies[symbol] = accuracy
                else:
                    model = st.session_state.models[symbol]
                    accuracy = st.session_state.model_accuracies[symbol]

                model_accuracies.append(accuracy)

                latest_data = data.iloc[-1]
                features = X.columns.tolist()
                X_latest = latest_data[features].values.reshape(1, -1)
                pred_prob = model.predict_proba(X_latest)[0]
                prediction = model.predict(X_latest)[0]
                prediction_confidence = pred_prob[prediction]

                current_price = latest_data['Close']

                # Log prediction confidence
                st.write(f"{SYMBOL_NAME_MAPPING.get(symbol, symbol)} - Prediction Confidence: {prediction_confidence:.2f}")

                # Sentiment Analysis (Impact set to zero)
                sentiment_score = perform_sentiment_analysis(symbol)
                # Adjust prediction confidence based on sentiment analysis
                adjusted_prediction_confidence = prediction_confidence + (sentiment_score * SENTIMENT_ANALYSIS_WEIGHT)
                adjusted_prediction_confidence = min(max(adjusted_prediction_confidence, 0), 1)

                # Determine recommended action based on prediction and thresholds
                if adjusted_prediction_confidence >= AUTO_TRADE_THRESHOLD:
                    action = 'Buy' if prediction == 1 else 'Sell'
                else:
                    action = 'Hold'

                # Volatility filter (Disabled)
                if VOLATILITY_THRESHOLD and st.session_state.latest_volatility[symbol] < VOLATILITY_THRESHOLD:
                    action = 'Hold'

                # Trend following adjustments (Disabled)
                if TREND_FOLLOWING_STRENGTH > 0:
                    # Simple trend following: compare EMA and SMA
                    if latest_data['EMA20'] > latest_data['SMA50']:
                        trend = 'Uptrend'
                    else:
                        trend = 'Downtrend'
                    if trend == 'Uptrend' and action == 'Sell' and not COUNTER_TREND_STRENGTH:
                        action = 'Hold'
                    if trend == 'Downtrend' and action == 'Buy' and not COUNTER_TREND_STRENGTH:
                        action = 'Hold'

                # Dynamic position sizing
                if DYNAMIC_POSITION_SIZING:
                    # Adjust amount based on prediction confidence and profit reinvestment
                    amount = base_trade_amount * adjusted_prediction_confidence
                    profit = calculate_profit_loss(symbol, current_price)
                    if profit > 0:
                        reinvestment = profit * PROFIT_REINVESTMENT_RATIO / current_price
                        amount += reinvestment
                else:
                    amount = trade_amount

                # Limit amount to positive values
                if amount <= 0:
                    amount = base_trade_amount if DYNAMIC_POSITION_SIZING else trade_amount

                # Ensure amount is a reasonable number
                amount = max(amount, 0.01)

                # Execute trade
                if action in ['Buy', 'Sell'] and (adjusted_prediction_confidence >= AUTO_TRADE_THRESHOLD):
                    executed = execute_trade(symbol, action, current_price, amount)
                    if not executed:
                        action = 'Hold'

                # Calculate profit/loss
                profit_loss = calculate_profit_loss(symbol, current_price)

                # Calculate signal prices
                buy_price = current_price if action == 'Buy' else None
                sell_price = current_price if action == 'Sell' else None
                stop_loss_price = st.session_state.stop_loss.get(symbol, None)

                signals_list.append({
                    'Symbol': SYMBOL_NAME_MAPPING.get(symbol, symbol),
                    'Current Price': f"${current_price:.2f}",
                    'Buy Price': f"${buy_price:.2f}" if buy_price else '-',
                    'Sell Price': f"${sell_price:.2f}" if sell_price else '-',
                    'Stop-Loss': f"${stop_loss_price:.2f}" if stop_loss_price else '-',
                    'Recommended Action': action,
                    'Profit/Loss': f"${profit_loss:.2f}"
                })

                # Risk management checks
                if RISK_MANAGEMENT_ENABLED and symbol in st.session_state.positions:
                    check_stop_loss(symbol, current_price)

            except Exception as e:
                st.error(f"An error occurred while processing {SYMBOL_NAME_MAPPING.get(symbol, symbol)}: {e}")

            progress_bar.progress((idx + 1) / total_symbols)

        # Remove progress bar after completion
        progress_bar.empty()

        # Display signals in a table
        signals_df = pd.DataFrame(signals_list)
        if not signals_df.empty:
            signals_df = signals_df.sort_values(
                by='Symbol', ascending=True)

            # Apply custom styling to the DataFrame
            def highlight_recommended_action(val):
                color = ''
                if val == 'Buy':
                    color = 'green'
                elif val == 'Sell':
                    color = 'red'
                return f'color: {color}; font-weight: bold; font-size: 16px;'

            st.markdown("<h2 style='color: #4B79A1;'>Trade Signals</h2>", unsafe_allow_html=True)
            st.dataframe(signals_df.style.applymap(
                highlight_recommended_action,
                subset=['Recommended Action']
            ))
        else:
            st.write("No signals to display.")

        # Display current positions
        st.markdown("<h2 style='color: #4B79A1;'>Current Positions</h2>", unsafe_allow_html=True)
        positions_list = []
        for symbol, position in st.session_state.positions.items():
            data = yf.download(symbol, period='1d', progress=False)
            if not data.empty:
                current_price = data['Close'].iloc[-1]
                quantity = position['quantity']
                cost_basis = position['cost_basis']
                market_value = quantity * current_price
                profit_loss = calculate_profit_loss(symbol, current_price)
                positions_list.append({
                    'Symbol': SYMBOL_NAME_MAPPING.get(symbol, symbol),
                    'Quantity': quantity,
                    'Cost Basis': f"${cost_basis:.2f}",
                    'Current Price': f"${current_price:.2f}",
                    'Market Value': f"${market_value:.2f}",
                    'Profit/Loss': f"${profit_loss:.2f}"
                })
        if positions_list:
            positions_df = pd.DataFrame(positions_list)
            st.dataframe(positions_df)
        else:
            st.write("No positions currently held.")

        # Display trade history
        st.markdown("<h2 style='color: #4B79A1;'>Trade History</h2>", unsafe_allow_html=True)
        if st.session_state.trade_history:
            history_df = pd.DataFrame(st.session_state.trade_history)
            history_df['Date'] = history_df['Date'].dt.strftime('%Y-%m-%d %H:%M:%S')
            st.dataframe(history_df)
        else:
            st.write("No trades executed yet.")

        # Display account information
        st.markdown("<h2 style='color: #4B79A1;'>Account Information</h2>", unsafe_allow_html=True)
        st.write(f"**Current Balance:** ${st.session_state.balance:.2f}")
        roi, max_drawdown = calculate_portfolio_metrics()
        st.write(f"**ROI:** {roi:.2f}%")
        st.write(f"**Max Drawdown:** {max_drawdown:.2f}%")
        if model_accuracies:
            avg_accuracy = sum(model_accuracies) / len(model_accuracies)
            st.write(f"**Average Model Accuracy:** {avg_accuracy*100:.2f}%")

        # Display balance history chart
        st.markdown("<h2 style='color: #4B79A1;'>Account Balance History</h2>", unsafe_allow_html=True)
        fig = plot_balance_history()
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.write("No balance history to display.")

    # Note: The "Next analysis will run in..." message is no longer applicable without a timer.

if __name__ == "__main__":
    main()