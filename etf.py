from pathlib import Path
import appdirs as ad
CACHE_DIR = ".cache"
# Force appdirs to say that the cache dir is .cache
ad.user_cache_dir = lambda *args: CACHE_DIR
# Create the cache dir if it doesn't exist
Path(CACHE_DIR).mkdir(exist_ok=True)
import yfinance as yf
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from cvxopt import matrix, solvers

# Set the Streamlit app title
st.title("ETF Allocation Optimization")

# Define the list of symbols (stocks or assets)
symbols = ['VTI', 'ITOT', 'VEA', 'SCHF', 'VWO', 'IEMG', 'VIG', 'DGRO', 'VTEB', 'MUB', 'BND', 'BIV', 'VCIT', 'SCHP', 'VTIP']

# Define the start and end dates for historical data
start_date = "2016-01-01"
end_date = datetime.today().strftime('%Y-%m-%d')

# Create an empty DataFrame to store the data
df = pd.DataFrame()

# Fetch the data for each symbol and add it to the DataFrame
for symbol in symbols:
    data = yf.download(symbol, start=start_date, end=end_date)
    df[symbol] = data["Adj Close"]

# Calculate daily returns, expected returns, and covariance matrix
daily_returns = df.pct_change().dropna()
exp_rtn = daily_returns.mean() * 252
cov_mat = daily_returns.cov() * 252

# User input for risk aversion
risk_aversion = st.slider("Select Risk Aversion Coefficient (lambda):", min_value=1, max_value=10, value=1)

# Define the variables for optimization
num_assets = len(exp_rtn)
cov_mat_numpy = cov_mat.values
P = risk_aversion * matrix(cov_mat_numpy)
q = -matrix(exp_rtn)

# Constraint matrices
G = matrix(-np.eye(num_assets))  # Negative identity matrix for long-only constraint
h = matrix(np.zeros(num_assets))

# Sum of weights must equal 1
A = matrix(1.0, (1, num_assets))
b = matrix(1.0)

# Solve the quadratic programming problem
solvers.options['show_progress'] = False
optimal_weights = solvers.qp(P, q, G, h, A, b)

# Extract the optimal asset allocation weights
optimal_weights = np.array(optimal_weights['x'])

# Display the optimal weights
st.write("Optimal ETF Allocation Weights:")
for symbol, weight in zip(symbols, optimal_weights):
    if weight[0] != 0:
        st.write(f"{symbol}: {weight[0]:.4f}")




# Create a list of dictionaries with asset information
assets_data = [
    {
        "Attributes": "US Stocks",
        "About": "US Stocks represent an ownership share in US-based corporations. The US has the largest economy and stock market in the world. Although the US economy was hit hard in the 2008-2009 Financial Crisis, it is still one of the most resilient and active in the world because it is powered by a remarkable innovation engine.",
        "Ticker": "VTI",
        "Total Name": "Vanguard Total Stock Market ETF"
    },
    {
        "Attributes": "US Stocks",
        "About": "US Stocks represent an ownership share in US-based corporations. The US has the largest economy and stock market in the world. Although the US economy was hit hard in the 2008-2009 Financial Crisis, it is still one of the most resilient and active in the world because it is powered by a remarkable innovation engine.",
        "Ticker": "ITOT",
        "Total Name": "iShares Core S&P Total U.S. Stock Market ETF"
    },
    {
        "Attributes": "Foreign Stocks",
        "About": "Foreign Developed Market Stocks represent an ownership share in companies headquartered in developed economies like Europe, Australia, and Japan. Although the economies of Europe and Japan have experienced some struggles in the last few decades, Foreign Developed Markets represent a significant part of the world economy and provide diversification from US Stocks.",
        "Ticker": "VEA",
        "Total Name": "Vanguard FTSE Developed Markets ETF"
    },
    {
        "Attributes": "Foreign Stocks",
        "About": "Foreign Developed Market Stocks represent an ownership share in companies headquartered in developed economies like Europe, Australia, and Japan. Although the economies of Europe and Japan have experienced some struggles in the last few decades, Foreign Developed Markets represent a significant part of the world economy and provide diversification from US Stocks.",
        "Ticker": "SCHF",
        "Total Name": "Schwab International Equity ETF"
    },
    {
        "Attributes": "Emerging Markets",
        "About": "Emerging Market Stocks represent an ownership share in foreign companies in developing economies such as Brazil, China, India, South Africa, and Taiwan. Compared with developed countries, developing countries have younger demographics, expanding middle classes and faster economic growth. They account for half of world GDP, and that portion is likely to increase as the Emerging Markets develop. Emerging Market Stocks are more volatile, but have historically delivered higher returns than US Stocks and Foreign Developed Markets Stocks.",
        "Ticker": "VWO",
        "Total Name": "Vanguard FTSE Emerging Markets ETF"
    },
    {
        "Attributes": "Emerging Markets",
        "About": "Emerging Market Stocks represent an ownership share in foreign companies in developing economies such as Brazil, China, India, South Africa, and Taiwan. Compared with developed countries, developing countries have younger demographics, expanding middle classes and faster economic growth. They account for half of world GDP, and that portion is likely to increase as the Emerging Markets develop. Emerging Market Stocks are more volatile, but have historically delivered higher returns than US Stocks and Foreign Developed Markets Stocks.",
        "Ticker": "IEMG",
        "Total Name": "iShares Core MSCI Emerging Markets ETF"
    },
    {
        "Attributes": "Dividend Stocks",
        "About": "Dividend Growth Stocks represent an ownership share in US companies that have increased their dividend payout each year for the last ten or more consecutive years. They tend to be large-cap well-run companies in less cyclical industries and thus are less volatile than stocks more generally. Many companies in this asset class have higher dividend yields than their corporate bond yields and the yields on US government bonds.",
        "Ticker": "VIG",
        "Total Name": "Vanguard Dividend Appreciation ETF"
    },
    {
        "Attributes": "Dividend Stocks",
        "About": "Dividend Growth Stocks represent an ownership share in US companies that have increased their dividend payout each year for the last ten or more consecutive years. They tend to be large-cap well-run companies in less cyclical industries and thus are less volatile than stocks more generally. Many companies in this asset class have higher dividend yields than their corporate bond yields and the yields on US government bonds.",
        "Ticker": "DGRO",
        "Total Name": "iShares Core Dividend Growth ETF"
    },
    {
        "Attributes": "Municipal Bonds",
        "About": "Municipal Bonds are debt issued by U.S. state and local governments. Unlike most other bonds, Municipal Bonds’ interest is exempt from federal income taxes. They provide individual investors in high tax brackets a tax efficient way to obtain income, low historical volatility, and diversification.",
        "Ticker": "VTEB",
        "Total Name": "Vanguard Tax-Exempt Bond ETF"
    },
    {
        "Attributes": "Municipal Bonds",
        "About": "Municipal Bonds are debt issued by U.S. state and local governments. Unlike most other bonds, Municipal Bonds’ interest is exempt from federal income taxes. They provide individual investors in high tax brackets a tax efficient way to obtain income, low historical volatility, and diversification.",
        "Ticker": "MUB",
        "Total Name": "iShares National Muni Bond ETF"
    },
    {
        "Attributes": "US Bonds",
        "About": "US Bonds are high-quality debt issued by the US Treasury, government agencies, and US corporations. US Bonds provide steady income, low historical volatility and low correlation with stocks.",
        "Ticker": "BND",
        "Total Name": "Vanguard Total Bond Market ETF"
    },
    {
        "Attributes": "US Bonds",
        "About": "US Bonds are high-quality debt issued by the US Treasury, government agencies, and US corporations. US Bonds provide steady income, low historical volatility and low correlation with stocks.",
        "Ticker": "BIV",
        "Total Name": "Vanguard Intermediate-Term Bond ETF"
    },
    {
        "Attributes": "Corporate Bonds",
        "About": "Corporate Bonds are debt issued by US corporations with investment-grade credit ratings to fund business activities. Compared to US Bonds, which contain large amounts of bonds issued by the US government and government agencies, corporate bonds offer higher yields due to higher credit risk, illiquidity, and callability.",
        "Ticker": "VCIT",
        "Total Name": "Vanguard Intermediate-Term Corporate Bond ETF"
    },
    {
        "Attributes": "TIPS",
        "About": "Treasury Inflation-Protected Securities (TIPS) are inflation-indexed bonds issued by the U.S. federal government. Unlike nominal bonds, TIPS’ principal and coupons are adjusted periodically based on the Consumer Price Index (CPI). Their inflation-indexed feature and low volatility makes them the only asset class that can provide income generation and inflation protection to risk averse investors.",
        "Ticker": "SCHP",
        "Total Name": "Schwab U.S. TIPS ETF"
    },
    {
        "Attributes": "TIPS",
        "About": "Treasury Inflation-Protected Securities (TIPS) are inflation-indexed bonds issued by the U.S. federal government. Unlike nominal bonds, TIPS’ principal and coupons are adjusted periodically based on the Consumer Price Index (CPI). Their inflation-indexed feature and low volatility makes them the only asset class that can provide income generation and inflation protection to risk averse investors.",
        "Ticker": "VTIP",
        "Total Name": "Vanguard Short-Term Inflation-Protected Securities ETF"
    }
]

# Create a Pandas DataFrame from the list of dictionaries
assets_df = pd.DataFrame(assets_data)

# Set the Streamlit app title
st.title("Asset Information Table")

# Display the asset information table using Streamlit
st.table(assets_df)

