# Stock Predictor App 📈

This **Stock Predictor** app is built using **Streamlit** and leverages the **Facebook Prophet** forecasting model to predict future stock prices. It allows users to select popular stocks like MSFT, NVDA, GOOG, AAPL, and TSLA, and forecast their future performance over a defined period of up to 5 years. It also includes additional analysis of the forecast components and interactive visualizations powered by **Plotly**.

## Features

- **Stock Data Visualization**: Fetches historical stock data from **Yahoo Finance** and provides an interactive plot of stock prices (Open & Close).
- **Future Predictions**: Uses **Facebook Prophet** to predict stock prices for up to 5 years.
- **Forecast Components**: Breaks down forecast into trend and seasonality components for deeper analysis.
- **Social Links**: Connect with the app creator via LinkedIn, GitHub, and Twitter.
- **Custom Time Range**: View stock performance with predefined time periods like 1 month, 6 months, YTD, or 1 year.

## How to Run Locally

### Prerequisites

Make sure you have the following installed:

- Python 3.x
- pip (Python package manager)

### Clone the Repository

```bash
git clone https://github.com/SaiyanAhmed/Streamlit-FB-Prophet.git
cd stock-predictor-app
```

### Install the Dependencies

```bash
pip install -r requirements.txt
```

### Run the App

```bash
streamlit run app.py
```

### Once the app starts, open your browser and navigate to:
```arduino
http://localhost:8501
```

## How the App Works

1. **Stock Selection**  
   Choose from a list of popular stocks like Microsoft (MSFT), Nvidia (NVDA), Google (GOOG), Apple (AAPL), and Tesla (TSLA).

2. **Prediction Period**  
   Select the number of years (1-5) for which you want to predict the stock prices.

3. **Data Loading and Visualization**  
   The app downloads historical data for the selected stock using Yahoo Finance and displays it in a time-series chart.

4. **Forecasting**  
   The Prophet model is trained on the historical data, and the forecasted stock prices for the selected period are displayed in a second time-series chart.

5. **Additional Forecast Analysis**  
   You can expand the section to view additional breakdowns of the forecast including trend and seasonal components.

## Connect with the me

If you would like to connect with me, feel free to check out my website or follow me on social media:

- [Website](https://www.sakiful.com/)
- [LinkedIn](https://www.linkedin.com/in/sakiful/)
- [GitHub](https://github.com/sakiful)
- [Twitter](https://twitter.com/sakiful)

## License
This project is licensed under the MIT License. See the LICENSE file for details.



