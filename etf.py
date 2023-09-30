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
