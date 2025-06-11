
pip install -r requirements.txt
import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta




# Helper: get metric from ticker info or price
def get_metric_value(ticker, metric, data, date):
    if metric == "Price":
        try:
            return data[ticker].loc[date]["Close"]
        except:
            return None
    info = yf.Ticker(ticker).info
    if metric == "PE Ratio":
        return info.get("trailingPE", None)
    if metric == "Market Cap":
        return info.get("marketCap", None)
    if metric == "EPS":
        return info.get("trailingEps", None)
    if metric == "Dividend Yield":
        return info.get("dividendYield", None)
    if metric == "ROE":
        return info.get("returnOnEquity", None)
    return None

# Condition evaluation
def check_condition(val, operator, target):
    if val is None:
        return False
    if operator == "<":
        return val < target
    if operator == ">":
        return val > target
    if operator == "=":
        return abs(val - target) < 1e-6
    return False

st.title("Trading Strategy Dashboard")

# --- Stock Universe Input ---
st.header("Stock Universe")
tickers = st.text_input("Enter ticker symbols separated by commas (e.g. AAPL,MSFT,TSLA)").upper()
stock_universe = [t.strip() for t in tickers.split(",") if t.strip()]

# --- Rules Creation ---
st.header("Create Rules")

if 'rules' not in st.session_state:
    st.session_state.rules = []

metric_options = ["Price", "PE Ratio", "Market Cap", "EPS", "Dividend Yield", "ROE"]
action_options = ["buy", "sell"]
operator_options = ["<", ">", "="]
logic_options = ["", "and", "or"]
allocation_type_options = ["percent", "fixed"]

with st.form("rule_form", clear_on_submit=True):
    conditions = []
    n_conditions = st.number_input("Number of conditions", min_value=1, max_value=5, value=1, step=1)
    
    for i in range(n_conditions):
        st.markdown(f"**Condition {i+1}**")
        cols = st.columns([1,1,1,1,1])
        action = cols[0].selectbox("Action", action_options, key=f"action_{i}")
        metric = cols[1].selectbox("Metric", metric_options, key=f"metric_{i}")
        operator = cols[2].selectbox("Operator", operator_options, key=f"operator_{i}")
        value = cols[3].number_input("Value", key=f"value_{i}")
        logic = cols[4].selectbox("Logic", logic_options, key=f"logic_{i}")
        conditions.append({
            "action": action,
            "metric": metric,
            "operator": operator,
            "value": value,
            "logic": logic
        })
    
    st.markdown("**Allocation**")
    alloc_type = st.selectbox("Allocation Type", allocation_type_options, key="alloc_type")
    alloc_value = st.number_input("Allocation Value", min_value=0.0, key="alloc_value")

    submitted = st.form_submit_button("Add Rule")
    if submitted:
        # Validate conditions
        for c in conditions:
            if c["value"] is None:
                st.warning("Please enter valid values for all conditions.")
                st.stop()
        if alloc_value <= 0:
            st.warning("Allocation value must be positive.")
            st.stop()
        st.session_state.rules.append({"conditions": conditions, "allocation": {"type": alloc_type, "value": alloc_value}})
        st.success("Rule added!")

# Display rules
st.header("Defined Rules")
for i, rule in enumerate(st.session_state.rules):
    cond_texts = []
    for cond in rule["conditions"]:
        s = f"{cond['action'].upper()} if {cond['metric']} {cond['operator']} {cond['value']}"
        if cond["logic"]:
            s += f" {cond['logic'].upper()}"
        cond_texts.append(s)
    alloc = rule["allocation"]
    alloc_str = f"Allocate {alloc['value']} ({alloc['type']})"
    st.write(f"Rule {i+1}: " + " ".join(cond_texts) + f" with {alloc_str}")

if st.button("Clear Rules"):
    st.session_state.rules = []

# --- Backtest Setup ---
st.header("Backtesting Parameters")

capital = st.number_input("Starting Capital ($)", min_value=1000.0, value=10000.0, step=1000.0)
weeks = st.number_input("Backtest Length (weeks)", min_value=1, max_value=520, value=26, step=1)
fee_per_trade = st.number_input("Broker Fee ($ per trade)", min_value=0.0, value=5.0, step=0.1)

if st.button("Run Backtest"):
    if not stock_universe:
        st.error("Add tickers first")
        st.stop()
    if not st.session_state.rules:
        st.error("Add rules first")
        st.stop()

    days = weeks * 7
    end_date = datetime.today()
    start_date = end_date - timedelta(days=days)
    data = yf.download(stock_universe, start=start_date.strftime("%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d"), group_by='ticker', progress=False)
    if data.empty:
        st.error("No data fetched for selected period/tickers.")
        st.stop()

    portfolio = {ticker: 0 for ticker in stock_universe}
    cash = capital
    portfolio_value_history = []
    dates = sorted(data.index.unique())
    trades = []

    for date in dates:
        date_str = date.strftime("%Y-%m-%d")
        for rule in st.session_state.rules:
            conditions = rule["conditions"]
            # evaluate all conditions - for now treat them with AND logic simply
            passed = True
            for cond in conditions:
                metric_val = None
                # get metric for any ticker, condition is evaluated over universe - simplify to any ticker satisfying
                satisfied = False
                for ticker in stock_universe:
                    metric_val = get_metric_value(ticker, cond["metric"], data, date)
                    if check_condition(metric_val, cond["operator"], cond["value"]):
                        satisfied = True
                        break
                if not satisfied:
                    passed = False
                    break

            if not passed:
                continue

            allocation = rule["allocation"]
            action = conditions[0]["action"]

            for ticker in stock_universe:
                try:
                    price = data[ticker].loc[date]["Close"]
                except:
                    continue
                if pd.isna(price):
                    continue

                if action == "buy" and cash > 0:
                    amount = cash * (allocation["value"] / 100) if allocation["type"] == "percent" else min(allocation["value"], cash)
                    shares = int(amount / price)
                    cost = shares * price + fee_per_trade
                    if shares > 0 and cost <= cash:
                        portfolio[ticker] += shares
                        cash -= cost
                        trades.append({"date": date, "ticker": ticker, "action": "buy", "price": price, "shares": shares})

                elif action == "sell" and portfolio[ticker] > 0:
                    shares = portfolio[ticker]
                    revenue = shares * price - fee_per_trade
                    portfolio[ticker] = 0
                    cash += revenue
                    trades.append({"date": date, "ticker": ticker, "action": "sell", "price": price, "shares": shares})

        total_val = cash + sum(portfolio[t] * data[t].loc[date]["Close"] if date in data[t].index else 0 for t in stock_universe)
        portfolio_value_history.append(total_val)

    # Compute stats
    total_trades = len(trades) // 2
    wins = 0
    money_made = 0.0
    total_spread = 0.0
    trade_pairs = {}

    for trade in trades:
        key = trade["ticker"]
        if trade["action"] == "buy":
            trade_pairs[key] = trade
        elif trade["action"] == "sell" and key in trade_pairs:
            buy_trade = trade_pairs.pop(key)
            profit = (trade["price"] - buy_trade["price"]) * min(buy_trade["shares"], trade["shares"]) - 2*fee_per_trade
            if profit > 0:
                wins += 1
            money_made += profit
            total_spread += abs(trade["price"] - buy_trade["price"])

    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0.0

    # Plot portfolio value
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=portfolio_value_history, mode='lines', name='Portfolio Value'))

    # Add trades markers
    for trade in trades:
        val_idx = dates.index(trade["date"])
        val = portfolio_value_history[val_idx]
        color = "green" if trade["action"] == "buy" else "red"
        symbol = "triangle-up" if trade["action"] == "buy" else "triangle-down"
        fig.add_trace(go.Scatter(
            x=[trade["date"]],
            y=[val],
            mode='markers',
            marker=dict(color=color, size=12, symbol=symbol),
            name=f"{trade['action'].capitalize()} {trade['ticker']}"
        ))

    fig.update_layout(
        title="Backtest Portfolio Value",
        xaxis_title="Date",
        yaxis_title="Value ($)",
        legend=dict(font=dict(size=10))
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown(f"""
    **Stats:**
    - Total Trades (round-trips): {total_trades}
    - Win Rate: {win_rate:.2f}%
    - Total Money Made: ${money_made:.2f}
    - Total Spread: ${total_spread:.2f}
    - Broker Fee per Trade: ${fee_per_trade:.2f}
    - Backtest Length: {weeks} weeks ({days} days)
    """)
