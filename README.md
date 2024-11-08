# Advanced Multi-Asset Trading Bot

A sophisticated trading bot platform that enables automated trading across multiple asset classes including Commodities, Forex, and Cryptocurrencies, powered by machine learning and technical analysis.

## Project Overview

- Real-time market data analysis using yfinance
- Machine learning predictions using XGBoost
- Interactive web interface built with Streamlit
- Multiple asset class support:
  - Commodities (Gold, Silver, Natural Gas, Coffee)
  - Forex (EUR/USD, USD/JPY, GBP/USD, AUD/USD, USD/CAD)
  - Cryptocurrencies (BTC, ETH, XRP, ADA, DOT, LINK, LTC)

## Tech Stack

- Python 3.10.12
- Streamlit 1.25.0
- pandas 1.5.3
- numpy 1.24.3
- yfinance 0.2.43
- plotly 5.15.0
- scikit-learn 1.3.0
- xgboost 1.7.6

## Project Structure

```
RoboTrade/
├── add.py              # Main application file with trading logic
├── requirements.txt    # Python dependencies
├── runtime.txt        # Python runtime specification
├── setup.sh           # Streamlit configuration
├── Procfile           # Deployment configuration
├── .gitignore         # Git ignore patterns
├── .slugignore        # Deployment ignore patterns
└── README.md          # Project documentation
```

## Features

- Advanced technical analysis with multiple indicators
- Machine learning-based predictions
- Risk management system
- Dynamic position sizing
- Real-time market data analysis
- Interactive charts and performance metrics
- Portfolio tracking and management
- Trade history and performance analytics
- Manual and automated trading modes

## Installation

1. Clone the repository:
```bash
git clone https://github.com/ituhub/RoboTrade.git
cd RoboTrade
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the application:
```bash
streamlit run add.py
```

## Trading Features

- Multi-asset trading support
- Real-time market data analysis
- Technical indicators:
  - Moving Averages (SMA, EMA)
  - RSI
  - MACD
  - Bollinger Bands
- Machine learning predictions
- Risk management with stop-loss and take-profit
- Portfolio performance tracking
- Trade history logging

## Configuration

The application uses environment variables for configuration. Key settings:

- RISK_MANAGEMENT_ENABLED
- TAKE_PROFIT_PERCENTAGE
- TRADING_FREQUENCY
- ML_MODEL_UPDATE_FREQUENCY
- AUTO_TRADE_THRESHOLD

## Usage

1. Select asset class (Commodities, Forex, or Crypto)
2. Choose trading mode (AUTO or MANUAL)
3. Set initial balance and risk parameters
4. Monitor trade signals and portfolio performance
5. Review trade history and analytics

## Deployment

The application is configured for deployment with the following files:
- `Procfile`: Web server configuration
- `setup.sh`: Streamlit setup
- `runtime.txt`: Python version specification
- `.slugignore`: Deployment file exclusions

## License

This project is available for use under standard open-source terms.

## Disclaimer

This trading bot is for educational and research purposes only. Always conduct your own research and risk assessment before trading. Past performance does not guarantee future results.

## Support

For issues, questions, or contributions, please open an issue in the GitHub repository.
