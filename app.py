import subprocess
subprocess.run(["pip", "install", "pandas", "numpy", "scikit-learn", "openai", "streamlit", "python-dotenv", "yfinance", "matplotlib"])

import os
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from dotenv import load_dotenv
import openai
import streamlit as st
from datetime import datetime, timedelta
import matplotlib.dates as mdates
import random

# Load API keys
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Streamlit app title
st.title("Stock Market Prediction & AI Insights")

# Function to get stock ticker using OpenAI
def get_stock_ticker(company_name):
    prompt = f"What is the stock ticker for {company_name} in NSE India? Provide only the ticker symbol."
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip() + ".NS"

# Store the selected company in session state
if "selected_company" not in st.session_state:
    st.session_state.selected_company = "Reliance Industries"
if "selected_ticker" not in st.session_state:
    st.session_state.selected_ticker = "RELIANCE.NS"

# User input for company name
company_name = st.text_input("Enter company name:", st.session_state.selected_company)

# Button to fetch stock ticker
if st.button("Find Stock Ticker"):
    ticker = get_stock_ticker(company_name)
    st.session_state.selected_company = company_name  # Store new company name
    st.session_state.selected_ticker = ticker  # Store new ticker
    # ✅ Force Streamlit to refresh the UI with new data
    st.rerun()

    ticker = st.session_state.selected_ticker  # ✅ Ensure the latest ticker is used
    st.write(f"✅ Identified Stock Ticker: {ticker}")

ticker = st.session_state.selected_ticker  # Ensure ticker updates correctly

# Fetch latest stock data from Yahoo Finance
@st.cache_data(ttl=0)  # Set cache timeout to 0 to force a fresh fetch
def fetch_stock_data(ticker):
    end_date = datetime.today().strftime('%Y-%m-%d')  
    start_date = (datetime.today() - timedelta(days=365)).strftime('%Y-%m-%d')  

    try:
        df = yf.download(ticker, start=start_date, end=end_date, interval="1d", progress=False, auto_adjust=True)
        
        if df.empty:
            st.error("⚠️ No stock data found. Please try another company.")
            return None

        df.reset_index(inplace=True)
        df.rename(columns={"Date": "Date", "Close": "Close", "Open": "Open", "High": "High", "Low": "Low", "Volume": "Volume"}, inplace=True)
        df["Date"] = df["Date"].dt.date
        df.set_index("Date", inplace=True)  # ✅ Set Date as index

        # Ensure enough data exists
        if len(df) < 30:
            st.error("⚠️ Not enough historical data to make accurate predictions. Try another company.")
            return None

        # Add a random column to force Streamlit to recognize a change
        df["force_update"] = random.random()  # ✅ Trick Streamlit into refreshing
        
        return df

    except Exception as e:
        st.error(f"⚠️ Error fetching stock data: {e}")
        return None

df = fetch_stock_data(ticker)

# Function to compute technical indicators
def compute_technical_indicators(df):
    df["SMA_20"] = df["Close"].rolling(window=20, min_periods=1).mean()
    df["EMA_20"] = df["Close"].ewm(span=20, adjust=False).mean()

    # RSI Calculation
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=14, min_periods=1).mean()
    avg_loss = loss.rolling(window=14, min_periods=1).mean()
    rs = avg_gain / avg_loss
    df["RSI_14"] = 100 - (100 / (1 + rs))

    # MACD Calculation
    df["EMA_12"] = df["Close"].ewm(span=12, adjust=False).mean()
    df["EMA_26"] = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = df["EMA_12"] - df["EMA_26"]

    # Bollinger Bands
    std_dev = df["Close"].rolling(window=20, min_periods=1).std()  # Ensure this is a Series
    df["Bollinger_Upper"] = df["SMA_20"].astype(float).squeeze() + (std_dev.astype(float).squeeze() * 2)
    df["Bollinger_Lower"] = df["SMA_20"].astype(float).squeeze() - (std_dev.astype(float).squeeze() * 2)

    return df

if df is not None:
    df = compute_technical_indicators(df)  # Compute indicators
    df.dropna(inplace=True)  # Drop NaN values due to indicator calculations
    st.write(f"📊 Latest Stock Data for {st.session_state.selected_company} ({ticker}):", df.drop(columns=["force_update"]).tail().reset_index())

     # Plot stock price trend
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df.index, df["Close"], label="Closing Price", color='blue', linewidth=2)
    ax.plot(df.index, df["High"], label="Highest Price", color='green', linestyle='dashed', linewidth=1.5)
    ax.plot(df.index, df["Low"], label="Lowest Price", color='red', linestyle='dashed', linewidth=1.5)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.set_title(f"Stock Price Trend for {st.session_state.selected_company}")
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))  
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=30))  
    plt.xticks(rotation=45)
    # Reduce Y-axis interval (set to 20, but adjust as needed)
    min_price = df[["Close", "High", "Low"]].min().min()
    max_price = df[["Close", "High", "Low"]].max().max()
    y_interval = 20  # Change this value to reduce or increase interval

    ax.set_yticks(np.arange(min_price, max_price, y_interval))
    # Add grid lines
    ax.grid(True, linestyle='--', alpha=0.6)
    st.pyplot(fig)

    # Function to generate AI Investment Insight
    def generate_market_insight(df, company_name, ticker):
        last_n_days = df.tail(10)  # Use the last 10 days

        prompt = f"""
Analyze the stock trends for {company_name} ({ticker}) based on the past 10 days of data:

Date, Open, High, Low, Close, Volume, SMA_20, EMA_20, RSI_14, MACD, Bollinger_Upper, Bollinger_Lower
{last_n_days.to_string(index=False)}

Provide an investment recommendation based on trends, RSI, MACD, and Bollinger Bands.
"""

        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )

        return response.choices[0].message.content

    if st.button("Get AI Investment Insight"):
        insight = generate_market_insight(df, st.session_state.selected_company, ticker)
        st.write(f"📊 AI Investment Insight for {st.session_state.selected_company}:", insight)

    # Auto-refresh option
    if st.button("Refresh Data"):
        st.rerun()
