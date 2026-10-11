# ==================================================
# 📈 Stock Price Prediction Web App using LSTM
# ==================================================

# -------------------- 1. Import Libraries --------------------
import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error
import math


# -------------------- 2. Page Configuration --------------------
st.set_page_config(page_title="Stock Price Predictor", layout="wide")

st.title("📊 Stock Price Prediction using LSTM Model")
st.write("Enter a stock symbol to download data, train the model, and predict prices.")


# -------------------- 3. User Input --------------------
stock_name = st.text_input(
    "Enter Stock Symbol (Example: AAPL, TCS.NS, INFY.NS)",
    ""
)


# -------------------- 4. Main Logic --------------------
if stock_name:

    st.info("Downloading stock data...")

    # Download 5 years data
    df = yf.download(stock_name, period="5y")

    if df.empty:
        st.error("Invalid stock symbol. Please try again.")
    else:
        st.success("Data downloaded successfully!")

        # -------------------- 5. Data Preparation --------------------
        close_prices = df["Close"].values.reshape(-1, 1)

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(close_prices)

        training_size = int(len(scaled_data) * 0.8)

        train_data = scaled_data[:training_size]
        test_data = scaled_data[training_size:]


        # -------------------- 6. Create Time Sequences --------------------
        def create_sequences(dataset, time_step=60):
            X, y = [], []
            for i in range(time_step, len(dataset)):
                X.append(dataset[i-time_step:i, 0])
                y.append(dataset[i, 0])
            return np.array(X), np.array(y)

        time_step = 60

        X_train, y_train = create_sequences(train_data, time_step)
        X_test, y_test = create_sequences(test_data, time_step)

        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)


        # -------------------- 7. Build LSTM Model --------------------
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(60, 1)),
            LSTM(50),
            Dense(1)
        ])

        model.compile(optimizer="adam", loss="mean_squared_error")

        st.info("Training model... Please wait ⏳")

        model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0)


        # -------------------- 8. Prediction --------------------
        predictions = model.predict(X_test)

        predictions = scaler.inverse_transform(predictions)
        y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

        rmse = math.sqrt(mean_squared_error(y_test, predictions))

        st.subheader("📌 Model Evaluation")
        st.write(f"**RMSE:** {rmse:.4f}")


        # -------------------- 9. Interactive Plot --------------------
        st.subheader("📈 Prediction Graph (Hover to See Values)")

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=df.index[-len(y_test):],
            y=y_test.flatten(),
            mode='lines',
            name='Actual Price'
        ))

        fig.add_trace(go.Scatter(
            x=df.index[-len(predictions):],
            y=predictions.flatten(),
            mode='lines',
            name='Predicted Price'
        ))

        fig.update_layout(
            title=f"{stock_name} Stock Price Prediction",
            xaxis_title="Date",
            yaxis_title="Stock Price",
            hovermode="x unified",
            template="plotly_white"
        )

        st.plotly_chart(fig, use_container_width=True)

        st.success("✅ Prediction Completed Successfully!")
