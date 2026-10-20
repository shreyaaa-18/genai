
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, SimpleRNN, LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

# 1. Prepare data
LOOKBACK = 12

values = df["Passengers"].values.reshape(-1, 1).astype("float32")
train_size = int(len(values) * 0.8)

scaler = MinMaxScaler()
scaler.fit(values[:train_size])
scaled = scaler.transform(values)


# 2. Create sequences
def make_sequences(data, start, end, lookback):
    X, y = [], []

    for i in range(start, end):
        X.append(data[i-lookback:i])
        y.append(data[i])

    return np.array(X), np.array(y)


X_train, y_train = make_sequences(
    scaled, LOOKBACK, train_size, LOOKBACK
)

X_test, y_test = make_sequences(
    scaled, train_size, len(scaled), LOOKBACK
)


# 3. Build Simple RNN
rnn = Sequential([
    Input(shape=(LOOKBACK, 1)),
    SimpleRNN(32, activation="tanh"),
    Dense(1)
])

rnn.compile(optimizer="adam", loss="mse")


# 4. Build LSTM
lstm = Sequential([
    Input(shape=(LOOKBACK, 1)),
    LSTM(32),
    Dense(1)
])

lstm.compile(optimizer="adam", loss="mse")


# 5. Early stopping
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)


# 6. Train both models
rnn_history = rnn.fit(
    X_train, y_train,
    epochs=100,
    batch_size=8,
    validation_split=0.15,
    shuffle=False,
    callbacks=[early_stop],
    verbose=0
)

lstm_history = lstm.fit(
    X_train, y_train,
    epochs=100,
    batch_size=8,
    validation_split=0.15,
    shuffle=False,
    callbacks=[early_stop],
    verbose=0
)

print("Training completed.")


# 7. Predict on test data
rnn_pred = rnn.predict(X_test, verbose=0)
lstm_pred = lstm.predict(X_test, verbose=0)

actual = scaler.inverse_transform(y_test)
rnn_pred = scaler.inverse_transform(rnn_pred)
lstm_pred = scaler.inverse_transform(lstm_pred)


# 8. Evaluate and compare models
print("RNN MSE:", mean_squared_error(actual, rnn_pred))
print("RNN MAE:", mean_absolute_error(actual, rnn_pred))

print("LSTM MSE:", mean_squared_error(actual, lstm_pred))
print("LSTM MAE:", mean_absolute_error(actual, lstm_pred))


# 9. Plot validation loss comparison
plt.figure(figsize=(8, 4))
plt.plot(rnn_history.history["val_loss"], label="RNN")
plt.plot(lstm_history.history["val_loss"], label="LSTM")
plt.title("Validation Loss Comparison")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.legend()
plt.show()


# 10. Plot actual vs predicted values
dates = df["Month"].iloc[train_size:]

plt.figure(figsize=(10, 4))
plt.plot(dates, actual.flatten(), label="Actual")
plt.plot(dates, rnn_pred.flatten(), label="RNN Prediction")
plt.plot(dates, lstm_pred.flatten(), label="LSTM Prediction")
plt.title("Actual vs Predicted Passenger Counts")
plt.xlabel("Month")
plt.ylabel("Passengers")
plt.legend()
plt.show()