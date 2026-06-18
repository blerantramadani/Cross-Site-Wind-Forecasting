"""
================================================================================
Diebold-Mariano Test on 5-Seed Averaged Predictions
================================================================================
Generates predictions from 5 seeds for both Kelmarsh and Penmanshiel,
averages them, and runs DM tests on the seed-averaged forecasts.

This addresses Reviewer 1's concern that DM on a single seed is statistically
weak — instead, we test on the mean prediction across 5 seeds.

USAGE:
  python3 dm_test_5seeds.py

OUTPUT:
  results/dm_test_5seeds_kelmarsh.csv
  results/dm_test_5seeds_penmanshiel.csv
  results/dm_test_5seeds_summary.csv

Total time: ~50 minutes (5 seeds × 2 datasets × 5 min each)
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
from scipy import stats

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

LOOK_BACK = 48
EPOCHS = 50
BATCH_SIZE = 32
LR = 0.001
TEST_HOURS = 1000
N_SEEDS = 5

DATASETS = {
    'kelmarsh': {'filepath': 'data/Kelmarsh_T1_2018_hourly.csv', 'rated_power_kw': 2050},
    'penmanshiel': {'filepath': 'data/Penmanshiel_T01_2018_hourly.csv', 'rated_power_kw': 2050},
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


def build_lstm(input_shape, seed):
    tf.random.set_seed(seed)
    model = Sequential([
        Input(shape=input_shape),
        LSTM(128, return_sequences=True), Dropout(0.05),
        LSTM(64, return_sequences=False), Dropout(0.05),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=LR), loss='mse')
    return model


def build_mlp(input_dim, seed):
    tf.random.set_seed(seed)
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(128, activation='relu'), Dropout(0.05),
        Dense(64, activation='relu'), Dropout(0.05),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=LR), loss='mse')
    return model


def run_seeds(name, config):
    """Train both architectures with 5 seeds and average their predictions."""
    print(f"\n{'#' * 60}")
    print(f"# {name.upper()} — 5 SEEDS")
    print(f"{'#' * 60}")
    
    df = load_and_prepare(config['filepath'])
    print(f"  Loaded {len(df)} rows")
    
    X_data = df[ALL_FEATURES].values
    y_data = df[TARGET_COL].values.reshape(-1, 1)
    
    fs = MinMaxScaler()
    ts = MinMaxScaler()
    X_scaled = fs.fit_transform(X_data)
    y_scaled = ts.fit_transform(y_data).flatten()
    
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
    
    # Persistence (deterministic)
    persistence_pred = np.concatenate([[y_test_real[0]], y_test_real[:-1]])
    
    # Linear Regression (deterministic)
    print("  Training LR...")
    lr = LinearRegression()
    lr.fit(X_train_flat, y_train)
    lr_pred = lr.predict(X_test_flat)
    lr_pred_real = ts.inverse_transform(lr_pred.reshape(-1, 1)).flatten()
    
    # Train 5 seeds for MLP and LSTM
    mlp_preds = []
    lstm_preds = []
    
    for seed in range(N_SEEDS):
        print(f"\n  Seed {seed}: MLP...")
        np.random.seed(seed)
        tf.random.set_seed(seed)
        mlp = build_mlp(X_train_flat.shape[1], seed)
        es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        mlp.fit(X_train_flat, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
                validation_split=0.1, callbacks=[es], verbose=0)
        mlp_pred = mlp.predict(X_test_flat, verbose=0)
        mlp_pred_real = ts.inverse_transform(mlp_pred).flatten()
        mlp_preds.append(mlp_pred_real)
        
        print(f"  Seed {seed}: LSTM...")
        np.random.seed(seed)
        tf.random.set_seed(seed)
        lstm = build_lstm((X_train_seq.shape[1], X_train_seq.shape[2]), seed)
        lstm.fit(X_train_seq, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
                 validation_split=0.1, callbacks=[es], verbose=0)
        lstm_pred = lstm.predict(X_test_seq, verbose=0)
        lstm_pred_real = ts.inverse_transform(lstm_pred).flatten()
        lstm_preds.append(lstm_pred_real)
    
    # Average across seeds
    mlp_avg = np.mean(mlp_preds, axis=0)
    lstm_avg = np.mean(lstm_preds, axis=0)
    
    # Save individual seeds for transparency
    os.makedirs('results', exist_ok=True)
    seeds_df = pd.DataFrame({
        'actual': y_test_real,
        'persistence': persistence_pred,
        'lr': lr_pred_real,
        'mlp_avg': mlp_avg,
        'lstm_avg': lstm_avg,
        **{f'mlp_seed{i}': mlp_preds[i] for i in range(N_SEEDS)},
        **{f'lstm_seed{i}': lstm_preds[i] for i in range(N_SEEDS)},
    })
    out = f"results/{name}_predictions_5seeds.csv"
    seeds_df.to_csv(out, index=False)
    print(f"\n  Saved: {out}")
    
    return y_test_real, persistence_pred, lr_pred_real, mlp_avg, lstm_avg


def diebold_mariano(actual, pred1, pred2, h=1):
    """DM test with HLN correction, using Student-t distribution."""
    actual = np.asarray(actual)
    pred1 = np.asarray(pred1)
    pred2 = np.asarray(pred2)
    
    loss1 = (actual - pred1) ** 2
    loss2 = (actual - pred2) ** 2
    d = loss1 - loss2
    n = len(d)
    d_mean = np.mean(d)
    d_var = np.var(d, ddof=1)
    
    dm_stat = d_mean / np.sqrt(d_var / n)
    correction = np.sqrt((n + 1 - 2*h + h*(h-1)/n) / n)
    dm_corrected = dm_stat * correction
    p_value = 2 * (1 - stats.t.cdf(np.abs(dm_corrected), df=n - 1))
    
    return dm_corrected, p_value


def main():
    print(f"{'=' * 60}")
    print(f"DM Test on 5-Seed Averaged Predictions")
    print(f"{'=' * 60}")
    
    all_results = []
    
    for name, config in DATASETS.items():
        if not os.path.exists(config['filepath']):
            print(f"\nSkipping {name}: {config['filepath']} not found")
            continue
        
        actual, persistence, lr, mlp_avg, lstm_avg = run_seeds(name, config)
        rated = config['rated_power_kw']
        
        print(f"\n{'-' * 60}")
        print(f"DM TESTS — {name.upper()} (5-seed averaged)")
        print(f"{'-' * 60}")
        
        # Quick metrics
        for label, pred in [('Persistence', persistence), ('LR', lr),
                             ('MLP (5-seed avg)', mlp_avg), ('LSTM (5-seed avg)', lstm_avg)]:
            rmse = math.sqrt(np.mean((actual - pred) ** 2))
            nrmse = (rmse / rated) * 100
            print(f"  {label:<20} nRMSE={nrmse:.2f}%, RMSE={rmse:.1f} kW")
        
        # DM tests
        pairs = [
            ('LSTM_avg_vs_MLP_avg', lstm_avg, mlp_avg),
            ('LSTM_avg_vs_Persistence', lstm_avg, persistence),
            ('MLP_avg_vs_Persistence', mlp_avg, persistence),
        ]
        
        for comp_name, p1, p2 in pairs:
            dm, pval = diebold_mariano(actual, p1, p2)
            sig = "significant" if pval < 0.05 else "not significant"
            
            rmse_p1 = math.sqrt(np.mean((actual - p1) ** 2))
            rmse_p2 = math.sqrt(np.mean((actual - p2) ** 2))
            
            print(f"\n  {comp_name}:")
            print(f"    DM (HLN-corrected) = {dm:.3f}")
            print(f"    p-value = {pval:.4f} ({sig})")
            print(f"    RMSE diff = {rmse_p1 - rmse_p2:+.2f} kW")
            
            all_results.append({
                'dataset': name,
                'comparison': comp_name,
                'DM_corrected': round(dm, 3),
                'p_value': round(pval, 4),
                'significant_5pct': pval < 0.05,
                'rmse_diff_kW': round(rmse_p1 - rmse_p2, 2),
                'n': len(actual)
            })
    
    if all_results:
        df_out = pd.DataFrame(all_results)
        out = 'results/dm_test_5seeds_summary.csv'
        df_out.to_csv(out, index=False)
        
        print(f"\n{'#' * 60}")
        print(f"# AGGREGATED RESULTS")
        print(f"{'#' * 60}")
        print(df_out.to_string(index=False))
        print(f"\nSaved: {out}")
        
        print(f"\n{'#' * 60}")
        print(f"# COMPARISON: 5-seed avg vs seed-0 (existing)")
        print(f"{'#' * 60}")
        if os.path.exists('results/diebold_mariano_results.csv'):
            old = pd.read_csv('results/diebold_mariano_results.csv')
            print("\nOld (seed 0):")
            print(old[['dataset', 'comparison', 'DM_corrected', 'p_value']].to_string(index=False))
            print("\nNew (5-seed avg):")
            print(df_out[['dataset', 'comparison', 'DM_corrected', 'p_value']].to_string(index=False))


if __name__ == "__main__":
    main()
