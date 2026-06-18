"""
================================================================================
DM Predictions Only — Minimal Run for Diebold-Mariano Test
================================================================================
Runs only seed 0 of LSTM, MLP, LR, Persistence on Kelmarsh and Penmanshiel
to generate predictions for Diebold-Mariano statistical test.

This is FASTER than running full pipeline because:
  - Only 1 seed (vs 5)
  - Only model comparison (no ablation)
  - Saves predictions only

Total time: ~30 minutes (vs 8-10 hours for full pipeline)

USAGE:
  python3 generate_dm_predictions.py
================================================================================
"""

import pandas as pd
import numpy as np
import math
import os
import sys
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Settings (must match main pipeline)
LOOK_BACK = 48
EPOCHS = 50
BATCH_SIZE = 32
LR = 0.001
TEST_HOURS = 1000
SEED = 0  # Only seed 0

DATASETS = {
    'kelmarsh': {
        'filepath': 'data/Kelmarsh_T1_2018_hourly.csv',
        'rated_power_kw': 2050,
        'label': 'kelmarsh_full',
    },
    'penmanshiel': {
        'filepath': 'data/Penmanshiel_T01_2018_hourly.csv',
        'rated_power_kw': 2050,
        'label': 'penmanshiel_full',
    },
}

TARGET_COL = 'ActivePower_kW'
ALL_FEATURES = [
    'WindSpeed_m_s', 'Temperature_C', 'ActivePower_lag1',
    'Hour_sin', 'Hour_cos', 'Month_sin', 'Month_cos', 'WindSpeed_Rolling3h'
]


def load_and_prepare(filepath):
    df = pd.read_csv(filepath)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df.set_index('Timestamp', inplace=True)
    df['Hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
    df['Hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
    df['Month_sin'] = np.sin(2 * np.pi * df.index.month / 12)
    df['Month_cos'] = np.cos(2 * np.pi * df.index.month / 12)
    df['WindSpeed_Rolling3h'] = df['WindSpeed_m_s'].rolling(window=3).mean()
    df['ActivePower_lag1'] = df[TARGET_COL].shift(1)
    return df.dropna()


def build_lstm(input_shape, seed=SEED):
    tf.random.set_seed(seed)
    model = Sequential([
        Input(shape=input_shape),
        LSTM(128, return_sequences=True), Dropout(0.05),
        LSTM(64, return_sequences=False), Dropout(0.05),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=LR), loss='mse')
    return model


def build_mlp(input_dim, seed=SEED):
    tf.random.set_seed(seed)
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(128, activation='relu'), Dropout(0.05),
        Dense(64, activation='relu'), Dropout(0.05),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=LR), loss='mse')
    return model


def run_dataset(name, config):
    print(f"\n{'#' * 60}")
    print(f"# {name.upper()}")
    print(f"{'#' * 60}")
    
    df = load_and_prepare(config['filepath'])
    print(f"  Loaded {len(df)} rows")
    
    X_data = df[ALL_FEATURES].values
    y_data = df[TARGET_COL].values.reshape(-1, 1)
    
    fs = MinMaxScaler()
    ts = MinMaxScaler()
    X_scaled = fs.fit_transform(X_data)
    y_scaled = ts.fit_transform(y_data).flatten()
    
    # Sequences
    X_seq, y_seq = [], []
    for i in range(LOOK_BACK, len(X_scaled)):
        X_seq.append(X_scaled[i - LOOK_BACK:i])
        y_seq.append(y_scaled[i])
    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)
    
    split = len(X_seq) - TEST_HOURS
    X_train_seq, X_test_seq = X_seq[:split], X_seq[split:]
    y_train, y_test = y_seq[:split], y_seq[split:]
    X_train_flat = X_train_seq[:, -1, :]
    X_test_flat = X_test_seq[:, -1, :]
    
    y_test_real = ts.inverse_transform(y_test.reshape(-1, 1)).flatten()
    
    # Persistence (shift by 1)
    persistence_pred = np.concatenate([[y_test_real[0]], y_test_real[:-1]])
    
    # Linear Regression
    print("  Training LR...")
    lr = LinearRegression()
    lr.fit(X_train_flat, y_train)
    lr_pred = lr.predict(X_test_flat)
    lr_pred_real = ts.inverse_transform(lr_pred.reshape(-1, 1)).flatten()
    
    # MLP
    print("  Training MLP (seed 0)...")
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    mlp = build_mlp(X_train_flat.shape[1])
    es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    mlp.fit(X_train_flat, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
            validation_split=0.1, callbacks=[es], verbose=0)
    mlp_pred = mlp.predict(X_test_flat, verbose=0)
    mlp_pred_real = ts.inverse_transform(mlp_pred).flatten()
    
    # LSTM
    print("  Training LSTM (seed 0)...")
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    lstm = build_lstm((X_train_seq.shape[1], X_train_seq.shape[2]))
    lstm.fit(X_train_seq, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
             validation_split=0.1, callbacks=[es], verbose=0)
    lstm_pred = lstm.predict(X_test_seq, verbose=0)
    lstm_pred_real = ts.inverse_transform(lstm_pred).flatten()
    
    # Save
    os.makedirs('results', exist_ok=True)
    pred_df = pd.DataFrame({
        'actual': y_test_real,
        'lstm_pred': lstm_pred_real,
        'mlp_pred': mlp_pred_real,
        'lr_pred': lr_pred_real,
        'persistence_pred': persistence_pred,
    })
    out = f"results/{config['label']}_predictions_seed0.csv"
    pred_df.to_csv(out, index=False)
    
    # Quick metrics
    rated = config['rated_power_kw']
    print(f"\n  Quick metrics (seed 0):")
    for col, name_str in [('persistence_pred', 'Persistence'),
                          ('lr_pred', 'LR'),
                          ('mlp_pred', 'MLP'),
                          ('lstm_pred', 'LSTM')]:
        rmse = math.sqrt(mean_squared_error(y_test_real, pred_df[col]))
        nrmse = (rmse / rated) * 100
        print(f"    {name_str:<12} nRMSE={nrmse:.2f}%, RMSE={rmse:.1f} kW")
    
    print(f"\n  Saved: {out}")


def main():
    print(f"{'=' * 60}")
    print(f"DM Predictions Generator — Seed 0 Only")
    print(f"{'=' * 60}")
    
    for name, config in DATASETS.items():
        if not os.path.exists(config['filepath']):
            print(f"\nSkipping {name}: {config['filepath']} not found")
            continue
        run_dataset(name, config)
    
    print(f"\n{'=' * 60}")
    print(f"Done. Now run: python3 diebold_mariano_test.py")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
