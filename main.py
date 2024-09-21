import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import time
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
from datetime import date

# Set page title and icon
st.set_page_config(
    page_title="Stock Predictor",
    page_icon="chart_with_upwards_trend",
)

# Add header to the app
st.header("Stock Predictor")
st.divider()

# Define start and end dates for stock data
start_date = "2017-01-01"
today_date = date.today().strftime("%Y-%m-%d")

# Define a list of stock tickers for selection
stocks = ["MSFT", "NVDA", "GOOG", "AAPL", "TSLA"]

# Sidebar for user inputs and model components
with st.sidebar:
    st.subheader("Model Components", divider=True)
    select_stock = st.selectbox("Select a Stock:", stocks)
    select_years = st.slider("Yers of Prediction:", 1,5)
    st.divider()
    
    st.info("""
    Hi, I'm Sakiful Ahmed Saiyan! I am a data and business analyst with a strong foundation in both academic and practical knowledge.
    """)

    # Social Media Links with Icons
    st.markdown("""
    ### Connect with Me:
    """)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            '<a href="https://www.sakiful.com/" target="_blank"><img src="https://cdn-icons-png.flaticon.com/256/3059/3059997.png" width="30"></a>',
            unsafe_allow_html=True
        )
    with col2:
        st.markdown(
            '<a href="https://www.linkedin.com/in/sakiful/" target="_blank"><img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" width="30"></a>',
            unsafe_allow_html=True
        )
    with col3:
        st.markdown(
            '<a href="https://github.com/sakiful" target="_blank"><img src="https://cdn-icons-png.flaticon.com/512/25/25231.png" width="30"></a>',
            unsafe_allow_html=True
        )
    with col4:
        st.markdown(
            '<a href="https://twitter.com/sakiful" target="_blank"><img src="https://cdn-icons-png.flaticon.com/512/733/733579.png" width="30"></a>',
            unsafe_allow_html=True
        )

# Convert years to prediction period in days
periods = select_years*365

# Function to load stock data from Yahoo Financ
@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, start_date, today_date, interval="1d")
    data.reset_index(inplace=True)
    return data

# Show a progress bar while loading the data
progress_text = "Loading data model. Please wait."
my_bar = st.progress(0, text=progress_text)

# Simulate progress
for percent_complete in range(100):
    time.sleep(0.01)
    my_bar.progress(percent_complete + 1, text=progress_text)
time.sleep(0.05)
my_bar.empty()

st.toast('Your model is ready!', icon='🎉')

# Load the stock data for the selected stock
df = load_data(select_stock)

st.subheader("Raw Data")
st.dataframe(df.iloc[::-1], hide_index=True)

# Function to plot the raw stock data (Open and Close prices)
def plot_raw():
    fig = px.line(df, x='Date', y=['Open', 'Close'])
    fig.update_xaxes(
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="YTD", step="year", stepmode="todate"),
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(step="all")
            ])
        ),
        rangeslider=dict(
                visible=True,
                bgcolor="#1f77b4",
                thickness=0.05
            ),
            type="date"
    )

    st.plotly_chart(fig)

st.subheader("Current Data Time Series")
plot_raw()
st.divider()

#Forecasting

# Prepare the data for Prophet model (rename columns for compatibility)
training_df = df[['Date', 'Close']]
training_df = training_df.rename(columns={'Date':'ds', 'Close':'y'})

# Train a Prophet model on the stock data
m = Prophet().fit(training_df)

# Create a future dataframe with specified prediction period
future = m.make_future_dataframe(periods)
forecast = m.predict(future)

# Display the forecasted data
st.subheader('Forecast Data')
st.dataframe(forecast.iloc[::-1], hide_index=True)

# Plot the forecast data using Prophet's Plotly integration
st.subheader("Forecast Data Time Series")
forecast_fig = plot_plotly(m,forecast)
st.plotly_chart(forecast_fig)

# Additional analysis using Prophet's component plots
with st.expander('Additional Analysis'):
    st.write(m.plot_components(forecast))