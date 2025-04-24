# daily_quantity_forecast.py
"""
Linear script : prévision de la quantité quotidienne de commandes

1. Connexion MySQL → extraction agrégée au jour
2. Analyse rapide (plots) – facultatif en production
3. Train / test split (dernier mois en test)
4. Trois modèles : SARIMA, Holt‑Winters, LSTM
5. Évaluation MAE / RMSE → sélection du meilleur
6. Sauvegarde du modèle + export CSV prévisions futur 30 jours

Pour exécuter :
$ python -m venv venv && source venv/bin/activate  # (Linux/Mac)
venv\Scripts\activate                               # (Windows)
$ pip install -r requirements.txt
$ python daily_quantity_forecast.py

requirements.txt minimal :
mysql-connector-python
SQLAlchemy
pandas
numpy
matplotlib
statsmodels
scikit-learn
tensorflow~=2.10  # ou adapté à ton GPU
joblib
"""

import warnings
warnings.filterwarnings("ignore")

import os
import math
import pickle
from datetime import timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler

from sqlalchemy import create_engine
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.callbacks import EarlyStopping

# ---------------------------------------------------------------------------
# 1 ) Connexion MySQL et extraction jour
# ---------------------------------------------------------------------------
DB_CONFIG = {
    "user": "root",          
    "password": "",          
    "host": "localhost",
    "port": 3306,
    "database": "ecommerce_2019",
}

url = (
    f"mysql+mysqlconnector://{DB_CONFIG['user']}:{DB_CONFIG['password']}@"
    f"{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
)

engine = create_engine(url)

query = """
    SELECT DATE(order_date)   AS order_day,
           SUM(quantity_ordered) AS qty
    FROM   orders_2019
    GROUP  BY order_day
    ORDER  BY order_day;
"""

df = pd.read_sql(query, engine, parse_dates=["order_day"])
df.set_index("order_day", inplace=True)
print(f"{len(df)} jours importés : {df.index.min().date()} → {df.index.max().date()}")

# ---------------------------------------------------------------------------
# 2 ) Analyse rapide
# ---------------------------------------------------------------------------
plt.figure(figsize=(10, 4))
plt.plot(df.index, df["qty"], label="Quantité / jour")
plt.title("Quantité quotidienne 2019")
plt.tight_layout()
plt.savefig("serie_journaliere.png")

# ---------------------------------------------------------------------------
# 3 ) Train / Test split
# ---------------------------------------------------------------------------
train_end = df.index.max() - timedelta(days=30)
train = df.loc[:train_end]
test = df.loc[train_end + timedelta(days=1):]

print(f"Train : {train.index.min().date()} - {train.index.max().date()} ({len(train)} j)\n"
      f"Test  : {test.index.min().date()} - {test.index.max().date()} ({len(test)} j)")

# ---------------------------------------------------------------------------
# 4 ) Modèle SARIMA
# ---------------------------------------------------------------------------
order = (1, 1, 1)
seasonal_order = (1, 1, 1, 7)  # saisonnalité hebdo (7) simple
sarima = SARIMAX(train["qty"], order=order, seasonal_order=seasonal_order,
                 enforce_stationarity=False, enforce_invertibility=False)
res_sarima = sarima.fit(disp=False)

pred_sarima = res_sarima.forecast(steps=len(test))
mae_sarima = mean_absolute_error(test["qty"], pred_sarima)
rmse_sarima = math.sqrt(mean_squared_error(test["qty"], pred_sarima))

# ---------------------------------------------------------------------------
# 5 ) Holt‑Winters Additif
# ---------------------------------------------------------------------------
hw = ExponentialSmoothing(train["qty"], trend="add", seasonal="add", seasonal_periods=7)
res_hw = hw.fit()
pred_hw = res_hw.forecast(len(test))
mae_hw = mean_absolute_error(test["qty"], pred_hw)
rmse_hw = math.sqrt(mean_squared_error(test["qty"], pred_hw))

# ---------------------------------------------------------------------------
# 6 ) LSTM
# ---------------------------------------------------------------------------
# Préparation : scaler + séquences
scaler = MinMaxScaler(feature_range=(0, 1))
train_scaled = scaler.fit_transform(train)

def create_sequences(data, look_back=7):
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i : i + look_back, 0])
        y.append(data[i + look_back, 0])
    return np.array(X), np.array(y)

look_back = 7
X_train, y_train = create_sequences(train_scaled, look_back)

X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))

model = Sequential([
    LSTM(50, input_shape=(look_back, 1)),
    Dense(1)
])
model.compile(optimizer="adam", loss="mse")
model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=16,
    verbose=0,
    callbacks=[EarlyStopping(patience=5, restore_best_weights=True)],
)

# Prévisions LSTM
test_inputs = pd.concat([train.tail(look_back), test])["qty"].values.reshape(-1, 1)
inputs_scaled = scaler.transform(test_inputs)
X_test = []
for i in range(look_back, len(inputs_scaled)):
    X_test.append(inputs_scaled[i - look_back : i, 0])
X_test = np.array(X_test).reshape(-1, look_back, 1)

pred_lstm_scaled = model.predict(X_test)
pred_lstm = scaler.inverse_transform(pred_lstm_scaled)

mae_lstm = mean_absolute_error(test["qty"].values, pred_lstm)
rmse_lstm = math.sqrt(mean_squared_error(test["qty"].values, pred_lstm))

# ---------------------------------------------------------------------------
# 7 ) Comparaison métriques
# ---------------------------------------------------------------------------
results = pd.DataFrame(
    {
        "model": ["SARIMA", "HoltWinters", "LSTM"],
        "MAE": [mae_sarima, mae_hw, mae_lstm],
        "RMSE": [rmse_sarima, rmse_hw, rmse_lstm],
    }
)
print("\nMétriques sur la période de test :\n", results)

best_row = results.loc[results["RMSE"].idxmin()]
best_model = best_row["model"]
print(f"\n→ Modèle retenu : {best_model}")

# ---------------------------------------------------------------------------
# 8 ) Sauvegarde du meilleur modèle & prévisions futures (30 jours)
# ---------------------------------------------------------------------------
forecast_horizon = 30
future_dates = pd.date_range(df.index.max() + timedelta(days=1), periods=forecast_horizon)

if best_model == "SARIMA":
    final_model = res_sarima  # déjà entraîné sur train
    # Re‑entraîne sur tout le jeu de données :
    final_model = SARIMAX(df["qty"], order=order, seasonal_order=seasonal_order,
                          enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
    forecast = final_model.forecast(steps=forecast_horizon)
    with open("best_model_sarima.pkl", "wb") as f:
        pickle.dump(final_model, f)
elif best_model == "HoltWinters":
    final_model = ExponentialSmoothing(df["qty"], trend="add", seasonal="add", seasonal_periods=7).fit()
    forecast = final_model.forecast(forecast_horizon)
    with open("best_model_holtwinters.pkl", "wb") as f:
        pickle.dump(final_model, f)
else:  # LSTM
    # Re‑entraîne LSTM sur tout l’historique
    full_scaled = scaler.fit_transform(df)
    X_full, y_full = create_sequences(full_scaled, look_back)
    X_full = X_full.reshape((X_full.shape[0], look_back, 1))
    model.fit(X_full, y_full, epochs=50, batch_size=16, verbose=0)
    model.save("best_model_lstm.h5")

    # Génère forecast auto‑récursif
    inputs = full_scaled[-look_back:].tolist()
    preds = []
    for _ in range(forecast_horizon):
        X = np.array(inputs[-look_back:]).reshape(1, look_back, 1)
        pred_scaled = model.predict(X, verbose=0)[0][0]
        preds.append(pred_scaled)
        inputs.append([pred_scaled])
    forecast = scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()

# DataFrame prédictions
forecast_df = pd.DataFrame({"date": future_dates, "qty_pred": forecast})
forecast_df.to_csv("daily_quantity_forecast.csv", index=False)
print("\nPrévisions sur 30 jours enregistrées dans daily_quantity_forecast.csv")
