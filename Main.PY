import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(layout="wide")

# Sidebar: Stock universe and rules
st.sidebar.header("Stock Universe")
stocks = st.sidebar.text_area("Add stock tickers (comma separated):", value="AAPL, MSFT, GOOG")
stock_list = [s.strip().upper() for s in stocks.split(',') if s.strip()]

st.sidebar.header("Trading Rules")
num_rules = st.sidebar.number_input("Number of Rules", min_value=1, max_value=10, value=1, step=1)

rules = []
for i in range(num_rules):
    st.sidebar.subheader(f"Rule {i+1}")
    action = st.sidebar.selectbox(f"Action {i+1}", ["BUY", "SELL"], key=f"action_{i}")
    target = st.sidebar.selectbox(f"Target {i+1}", ["Individual Stock", "All Stocks"], key=f"target_{i}")
    metric = st.sidebar.selectbox(f"Metric {i+1}", ["Price", "Volume", "Moving Average"], key=f"metric_{i}")
    comparator = st.sidebar.selectbox(f"Comparator {i+1}", [">", "<"], key=f"comp_{i}")
    threshold = st.sidebar.number_input(f"Threshold {i+1}", key=f"thresh_{i}")
    rules.append((action, target, metric, comparator, threshold))

# Main: Simulated data and rule testing
st.title("High Frequency Trading Algorithm Playground")
st.write("Use the sidebar to configure your stock universe and trading rules.")

# Simulate stock data
data = {ticker: np.random.normal(loc=100, scale=5, size=100) for ticker in stock_list}
df = pd.DataFrame(data)
df.index.name = "Time"

st.subheader("Simulated Stock Data")
st.line_chart(df)

st.subheader("Rule Evaluation")
results = []
for rule in rules:
    action, target, metric, comparator, threshold = rule
    for stock in stock_list:
        series = df[stock]
        latest_value = series.iloc[-1] if metric == "Price" else series.mean() if metric == "Moving Average" else np.random.randint(1000, 10000)
        passed = (latest_value > threshold) if comparator == ">" else (latest_value < threshold)
        if target == "All Stocks" or stock in stock_list:
            if passed:
                results.append(f"{action} signal for {stock} on metric {metric} ({latest_value:.2f} {comparator} {threshold})")

if results:
    for res in results:
        st.success(res)
else:
    st.info("No trading signals triggered based on current rules.")
