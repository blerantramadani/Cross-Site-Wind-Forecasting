"""
================================================================================
Forecasting Pipeline v2.3 — DEFINITIVE VERSION
================================================================================
Physics-Informed Digital Twin for Wind Power Forecasting

DESIGN PRINCIPLES:
  1. ONE codebase, multiple datasets (synthetic, Kelmarsh, Bogdanci)
  2. Dataset-specific RATED_POWER configurable via DATASET_CONFIG
  3. Target column always SEPARATE from features (no leakage)
  4. All results saved with dataset-specific prefixes for traceability
  5. Reproducible: fixed seeds, documented versions

DATASETS SUPPORTED:
  - synthetic_ou:  Our Digital Twin output (2300 kW rated, Bogdanci class)
  - kelmarsh:      UK wind farm SCADA data (2050 kW rated, Senvion MM92)
  - bogdanci:      Real SCADA from PVE Bogdanci (2300 kW rated) — when available
  - la_haute_borne: French wind farm SCADA (2050 kW rated, future)

USAGE:
  python3 forecasting_pipeline_v2_3.py              # runs synthetic (default)
  python3 forecasting_pipeline_v2_3.py kelmarsh     # runs Kelmarsh
  python3 forecasting_pipeline_v2_3.py bogdanci     # runs Bogdanci (when data ready)

Authors: Blerant Ramadani, Vangel Fustic (IEEE Senior Member)
Repository: github.com/blerantramadani/Wind-Power-Digital-Twin
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
from sklearn.metrics import mean_squared_error, mean_absolute_error

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

import matplotlib.pyplot as plt


# ================================================================================
# DATASET REGISTRY — central configuration for all datasets
# ================================================================================
DATASET_CONFIG = {
    'synthetic_ou': {
        'filepath': 'data/Synthetic_Bogdanci_OU.csv',
        'rated_power_kw': 2300,
        'description': 'Digital Twin synthetic data (Ornstein-Uhlenbeck + cubic power)',
        'label_prefix': 'synthetic',
    },
    'synthetic_noise': {
        'filepath': 'data/Synthetic_Bogdanci_Noise.csv',
        'rated_power_kw': 2300,
        'description': 'Uncorrelated noise dataset (noise floor baseline)',
        'label_prefix': 'noise_floor',
    },
    'kelmarsh': {
        'filepath': 'data/Kelmarsh_T1_2018_hourly.csv',
        'rated_power_kw': 2050,
        'description': 'Kelmarsh Wind Farm UK (Senvion MM92, 2018)',
        'label_prefix': 'kelmarsh',
    },
    'penmanshiel': {
        'filepath': 'data/Penmanshiel_T01_2018_hourly.csv',
        'rated_power_kw': 2050,
        'description': 'Penmanshiel Wind Farm UK (Senvion MM82, 2018)',
        'label_prefix': 'penmanshiel',
    },
    'bogdanci': {
        'filepath': 'data/Bogdanci_SCADA.csv',
        'rated_power_kw': 2300,
        'description': 'PVE Bogdanci real SCADA data (when available from ESM)',
        'label_prefix': 'bogdanci',
    },
}


# ================================================================================
# GLOBAL HYPERPARAMETERS (same across all experiments for fair comparison)
# ================================================================================
LOOK_BACK = 48           # 48 hours = 2 full diurnal cycles
EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 0.001
N_SEEDS = 5              # statistical rigor
TEST_HOURS = 1000        # test window

TARGET_COL = 'ActivePower_kW'

# Feature set — target is NEVER here
ALL_FEATURES = [
    'WindSpeed_m_s',
    'Temperature_C',
    'ActivePower_lag1',     # P(t-1) as input feature (NOT the target)
    'Hour_sin', 'Hour_cos',
    'Month_sin', 'Month_cos',
    'WindSpeed_Rolling3h'
]

os.makedirs('results', exist_ok=True)


# ================================================================================
# DATA PREPARATION
# ================================================================================
def load_and_prepare_data(filepath):
    """Load CSV and engineer features. Target remains separate."""
    df = pd.read_csv(filepath)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df.set_index('Timestamp', inplace=True)

    # Cyclic time encodings
    df['Hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
    df['Hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
    df['Month_sin'] = np.sin(2 * np.pi * df.index.month / 12)
    df['Month_cos'] = np.cos(2 * np.pi * df.index.month / 12)

    # Rolling mean on wind speed (low-pass filter)
    df['WindSpeed_Rolling3h'] = df['WindSpeed_m_s'].rolling(window=3).mean()

    # Explicit lagged power feature (P(t-1) as INPUT, distinct from target)
    df['ActivePower_lag1'] = df[TARGET_COL].shift(1)

    df = df.dropna()
    return df


def create_sequences(X_scaled, y_scaled, look_back=LOOK_BACK):
    """Create sliding window sequences."""
    X_seq, y_seq = [], []
    for i in range(look_back, len(X_scaled)):
        X_seq.append(X_scaled[i - look_back:i])
        y_seq.append(y_scaled[i])
    return np.array(X_seq), np.array(y_seq)


def compute_metrics(actual, predicted, rated_power):
    """Compute RMSE, nRMSE, and MAE with dataset-specific normalization."""
    rmse = math.sqrt(mean_squared_error(actual, predicted))
    nrmse = (rmse / rated_power) * 100
    mae = mean_absolute_error(actual, predicted)
    return {
        'RMSE_kW': round(rmse, 2),
        'nRMSE_pct': round(nrmse, 2),
        'MAE_kW': round(mae, 2)
    }


# ================================================================================
# MODEL ARCHITECTURES
# ================================================================================
def build_lstm(input_shape, seed=42):
    tf.random.set_seed(seed)
    model = Sequential([
        Input(shape=input_shape),
        LSTM(128, return_sequences=True),
        Dropout(0.05),
        LSTM(64, return_sequences=False),
        Dropout(0.05),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss='mse')
    return model


def build_mlp(input_dim, seed=42):
    tf.random.set_seed(seed)
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(128, activation='relu'),
        Dropout(0.05),
        Dense(64, activation='relu'),
        Dropout(0.05),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss='mse')
    return model


# ================================================================================
# CORE EXPERIMENT FUNCTION
# ================================================================================
def run_full_experiment(dataset_name, feature_list=None, seeds=None,
                        run_baselines=True, experiment_suffix=''):
    """
    Run the complete experiment on a specified dataset.

    Parameters
    ----------
    dataset_name : str
        Key from DATASET_CONFIG (e.g., 'synthetic_ou', 'kelmarsh').
    feature_list : list
        Features to use. Defaults to ALL_FEATURES.
    seeds : list
        Random seeds. Defaults to range(N_SEEDS).
    run_baselines : bool
        Whether to run Persistence, LR, MLP baselines.
    experiment_suffix : str
        Optional suffix for experiment name (e.g., 'ablation_no_temp').
    """
    if dataset_name not in DATASET_CONFIG:
        raise ValueError(f"Unknown dataset '{dataset_name}'. "
                         f"Choose from: {list(DATASET_CONFIG.keys())}")

    config = DATASET_CONFIG[dataset_name]
    filepath = config['filepath']
    rated_power = config['rated_power_kw']
    label = config['label_prefix']

    if experiment_suffix:
        exp_name = f"{label}_{experiment_suffix}"
    else:
        exp_name = label

    if feature_list is None:
        feature_list = ALL_FEATURES
    if seeds is None:
        seeds = list(range(N_SEEDS))

    print(f"\n{'=' * 70}")
    print(f"EXPERIMENT: {exp_name}")
    print(f"Dataset: {config['description']}")
    print(f"Rated power: {rated_power} kW")
    print(f"Features ({len(feature_list)}): {feature_list}")
    print(f"Target: {TARGET_COL} (separated from features)")
    print(f"Seeds: {seeds}")
    print(f"{'=' * 70}")

    # Safety check — target must never be in features
    if TARGET_COL in feature_list:
        raise ValueError(f"FATAL: '{TARGET_COL}' must not be in feature list!")

    # Load data
    df = load_and_prepare_data(filepath)
    print(f"Loaded {len(df)} rows after feature engineering.")

    # Extract features and target separately
    X_data = df[feature_list].values
    y_data = df[TARGET_COL].values.reshape(-1, 1)

    # Two separate scalers
    feature_scaler = MinMaxScaler(feature_range=(0, 1))
    target_scaler = MinMaxScaler(feature_range=(0, 1))

    X_scaled = feature_scaler.fit_transform(X_data)
    y_scaled = target_scaler.fit_transform(y_data).flatten()

    # Sequences
    X_seq, y_seq = create_sequences(X_scaled, y_scaled, LOOK_BACK)

    # Split
    split_idx = len(X_seq) - TEST_HOURS
    X_train_seq = X_seq[:split_idx]
    X_test_seq = X_seq[split_idx:]
    y_train = y_seq[:split_idx]
    y_test = y_seq[split_idx:]

    # Flat versions (current timestep only, no lookback)
    X_train_flat = X_train_seq[:, -1, :]
    X_test_flat = X_test_seq[:, -1, :]

    # Real-scale test targets
    y_test_real = target_scaler.inverse_transform(
        y_test.reshape(-1, 1)).flatten()

    results = {
        'LSTM': [], 'MLP': [], 'LinearRegression': [], 'Persistence': [],
        '_rated_power': rated_power,
        '_dataset': dataset_name,
    }

    # --- Persistence (no seed required) ---
    if run_baselines:
        print("\n  Persistence model...")
        pers_pred = y_test_real[:-1]
        pers_actual = y_test_real[1:]
        pers_metrics = compute_metrics(pers_actual, pers_pred, rated_power)
        results['Persistence'].append(pers_metrics)
        print(f"    nRMSE: {pers_metrics['nRMSE_pct']:.2f}%")

    # --- Multi-seed runs ---
    for i, seed in enumerate(seeds):
        print(f"\n  Seed {i + 1}/{len(seeds)} (seed={seed})")

        # Linear Regression (seed 0 only)
        if run_baselines and i == 0:
            print("    Linear Regression...")
            lr = LinearRegression()
            lr.fit(X_train_flat, y_train)
            lr_pred = lr.predict(X_test_flat)
            lr_pred_real = target_scaler.inverse_transform(
                lr_pred.reshape(-1, 1)).flatten()
            lr_metrics = compute_metrics(y_test_real, lr_pred_real, rated_power)
            results['LinearRegression'].append(lr_metrics)
            print(f"    LR nRMSE: {lr_metrics['nRMSE_pct']:.2f}%")

        # MLP
        if run_baselines:
            print("    MLP...")
            mlp = build_mlp(X_train_flat.shape[1], seed=seed)
            es = EarlyStopping(monitor='val_loss', patience=5,
                               restore_best_weights=True)
            mlp.fit(X_train_flat, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
                    validation_split=0.1, callbacks=[es], verbose=0)
            mlp_pred = mlp.predict(X_test_flat, verbose=0)
            mlp_pred_real = target_scaler.inverse_transform(mlp_pred).flatten()
            mlp_metrics = compute_metrics(y_test_real, mlp_pred_real, rated_power)
            results['MLP'].append(mlp_metrics)
            print(f"    MLP nRMSE: {mlp_metrics['nRMSE_pct']:.2f}%")

        # LSTM
        print("    LSTM...")
        np.random.seed(seed)
        tf.random.set_seed(seed)
        lstm = build_lstm((X_train_seq.shape[1], X_train_seq.shape[2]), seed=seed)
        es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        history = lstm.fit(X_train_seq, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
                           validation_split=0.1, callbacks=[es], verbose=0)
        lstm_pred = lstm.predict(X_test_seq, verbose=0)
        lstm_pred_real = target_scaler.inverse_transform(lstm_pred).flatten()
        lstm_metrics = compute_metrics(y_test_real, lstm_pred_real, rated_power)
        results['LSTM'].append(lstm_metrics)
        print(f"    LSTM nRMSE: {lstm_metrics['nRMSE_pct']:.2f}%")

        # Keep artifacts from seed 0 for plotting
        if i == 0:
            results['_predictions'] = lstm_pred_real
            results['_actual'] = y_test_real
            results['_history'] = history
            
            # Save predictions for Diebold-Mariano test (seed 0 only)
            if run_baselines:
                try:
                    pred_df = pd.DataFrame({
                        'actual': y_test_real,
                        'lstm_pred': lstm_pred_real,
                        'mlp_pred': mlp_pred_real,
                        'lr_pred': lr_pred_real,
                    })
                    pred_csv_path = f'results/{exp_name}_predictions_seed0.csv'
                    pred_df.to_csv(pred_csv_path, index=False)
                    print(f"    Saved predictions for DM test: {pred_csv_path}")
                except Exception as e:
                    print(f"    Warning: Could not save predictions: {e}")

    return results


# ================================================================================
# SUMMARIZATION AND EXPORT
# ================================================================================
def summarize_results(results, exp_name):
    """Aggregate multi-seed runs and save to CSV."""
    rows = []
    for model_name in ['Persistence', 'LinearRegression', 'MLP', 'LSTM']:
        runs = results.get(model_name, [])
        if not runs:
            continue
        nrmses = [r['nRMSE_pct'] for r in runs]
        rmses = [r['RMSE_kW'] for r in runs]
        maes = [r['MAE_kW'] for r in runs]

        rows.append({
            'Model': model_name,
            'nRMSE_mean': round(np.mean(nrmses), 2),
            'nRMSE_std': round(np.std(nrmses), 2),
            'RMSE_mean': round(np.mean(rmses), 2),
            'RMSE_std': round(np.std(rmses), 2),
            'MAE_mean': round(np.mean(maes), 2),
            'MAE_std': round(np.std(maes), 2),
            'n_runs': len(runs),
            'rated_power_kW': results['_rated_power'],
        })

    df_results = pd.DataFrame(rows)
    print(f"\n{'=' * 70}\nSUMMARY: {exp_name}\n{'=' * 70}")
    print(df_results.to_string(index=False))

    csv_path = f'results/{exp_name}_summary.csv'
    df_results.to_csv(csv_path, index=False)
    print(f"\nSaved to {csv_path}")
    return df_results


def plot_predictions(results, exp_name, hours=200):
    if '_predictions' not in results:
        return
    pred = results['_predictions'][:hours]
    actual = results['_actual'][:hours]

    plt.figure(figsize=(14, 5))
    plt.plot(actual, label='Actual Power', color='#1E293B', linewidth=1)
    plt.plot(pred, label='Predicted (LSTM)', color='#0D9488',
             linewidth=1, linestyle='--')
    plt.title(f'{exp_name} — Actual vs Predicted ({hours}h)', fontsize=13)
    plt.ylabel('Active Power (kW)')
    plt.xlabel('Test Sample (hours)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out = f'results/{exp_name}_prediction.png'
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved {out}")


def plot_training_loss(results, exp_name):
    if '_history' not in results:
        return
    history = results['_history']
    plt.figure(figsize=(8, 4))
    plt.plot(history.history['loss'], label='Training Loss', color='#1E293B')
    plt.plot(history.history['val_loss'], label='Validation Loss', color='#0D9488')
    plt.title(f'{exp_name} — LSTM Loss Curves', fontsize=13)
    plt.ylabel('MSE Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out = f'results/{exp_name}_loss.png'
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved {out}")


# ================================================================================
# ABLATION STUDY
# ================================================================================
def run_ablation_study(dataset_name, seeds=None):
    """Run feature ablation on the specified dataset."""
    if seeds is None:
        seeds = list(range(N_SEEDS))

    configurations = {
        'Full': ALL_FEATURES,
        'No_rolling_mean': [f for f in ALL_FEATURES if f != 'WindSpeed_Rolling3h'],
        'No_hour_encoding': [f for f in ALL_FEATURES if f not in ['Hour_sin', 'Hour_cos']],
        'No_month_encoding': [f for f in ALL_FEATURES if f not in ['Month_sin', 'Month_cos']],
        'No_temperature': [f for f in ALL_FEATURES if f != 'Temperature_C'],
        'No_lagged_power': [f for f in ALL_FEATURES if f != 'ActivePower_lag1'],
        'Wind_only': ['WindSpeed_m_s'],
    }

    ablation_results = []
    for config_name, features in configurations.items():
        print(f"\n--- Ablation: {config_name} ({len(features)} features) ---")
        res = run_full_experiment(
            dataset_name=dataset_name,
            feature_list=features,
            seeds=seeds,
            run_baselines=False,
            experiment_suffix=f'ablation_{config_name}'
        )
        nrmses = [r['nRMSE_pct'] for r in res['LSTM']]
        ablation_results.append({
            'Configuration': config_name,
            'n_features': len(features),
            'nRMSE_mean': round(np.mean(nrmses), 2),
            'nRMSE_std': round(np.std(nrmses), 2),
        })

    df = pd.DataFrame(ablation_results)
    label = DATASET_CONFIG[dataset_name]['label_prefix']
    out = f'results/{label}_ablation_study.csv'
    df.to_csv(out, index=False)
    print(f"\n{'=' * 70}\nABLATION RESULTS — {label}\n{'=' * 70}")
    print(df.to_string(index=False))
    print(f"\nSaved to {out}")
    return df


# ================================================================================
# MAIN ENTRY POINTS
# ================================================================================
def run_synthetic_validation(include_ablation=True):
    """Run all experiments on synthetic data (Digital Twin)."""
    print("\n" + "#" * 70)
    print("# PART A: SYNTHETIC VALIDATION (DIGITAL TWIN)")
    print("#" * 70)

    # 1. Noise floor
    print("\n--- Experiment 1: Noise Floor ---")
    noise_results = run_full_experiment('synthetic_noise', run_baselines=False)
    summarize_results(noise_results, 'noise_floor')

    # 2. Physics-consistent full comparison
    print("\n--- Experiment 2: Physics-Consistent (OU) ---")
    physics_results = run_full_experiment('synthetic_ou', run_baselines=True)
    summarize_results(physics_results, 'synthetic_full')
    plot_predictions(physics_results, 'synthetic_full')
    plot_training_loss(physics_results, 'synthetic_full')

    # 3. Ablation
    if include_ablation:
        print("\n--- Experiment 3: Feature Ablation ---")
        run_ablation_study('synthetic_ou')


def run_kelmarsh_validation(include_ablation=True):
    """Run all experiments on Kelmarsh SCADA data."""
    print("\n" + "#" * 70)
    print("# PART B: KELMARSH SCADA VALIDATION (UK)")
    print("#" * 70)

    kelmarsh_results = run_full_experiment('kelmarsh', run_baselines=True)
    summarize_results(kelmarsh_results, 'kelmarsh_full')
    plot_predictions(kelmarsh_results, 'kelmarsh_full')
    plot_training_loss(kelmarsh_results, 'kelmarsh_full')

    if include_ablation:
        print("\n--- Kelmarsh Feature Ablation ---")
        run_ablation_study('kelmarsh')


def run_penmanshiel_validation(include_ablation=True):
    """Run all experiments on Penmanshiel SCADA data."""
    print("\n" + "#" * 70)
    print("# PART C: PENMANSHIEL SCADA VALIDATION (UK)")
    print("#" * 70)

    penmanshiel_results = run_full_experiment('penmanshiel', run_baselines=True)
    summarize_results(penmanshiel_results, 'penmanshiel_full')
    plot_predictions(penmanshiel_results, 'penmanshiel_full')
    plot_training_loss(penmanshiel_results, 'penmanshiel_full')

    if include_ablation:
        print("\n--- Penmanshiel Feature Ablation ---")
        run_ablation_study('penmanshiel')


def run_bogdanci_validation(include_ablation=True):
    """Run all experiments on Bogdanci SCADA data (when available)."""
    print("\n" + "#" * 70)
    print("# PART C: BOGDANCI SCADA VALIDATION (North Macedonia)")
    print("#" * 70)

    bogdanci_results = run_full_experiment('bogdanci', run_baselines=True)
    summarize_results(bogdanci_results, 'bogdanci_full')
    plot_predictions(bogdanci_results, 'bogdanci_full')
    plot_training_loss(bogdanci_results, 'bogdanci_full')

    if include_ablation:
        run_ablation_study('bogdanci')


# ================================================================================
# MAIN
# ================================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("Forecasting Pipeline v2.3 — DEFINITIVE")
    print("=" * 70)

    # Parse argument
    mode = sys.argv[1] if len(sys.argv) > 1 else 'synthetic'

    if mode == 'synthetic':
        run_synthetic_validation(include_ablation=True)
    elif mode == 'kelmarsh':
        run_kelmarsh_validation(include_ablation=True)
    elif mode == 'penmanshiel':
        run_penmanshiel_validation(include_ablation=True)
    elif mode == 'bogdanci':
        run_bogdanci_validation(include_ablation=True)
    elif mode == 'all':
        run_synthetic_validation(include_ablation=True)
        run_kelmarsh_validation(include_ablation=True)
        run_penmanshiel_validation(include_ablation=True)
    elif mode == 'synthetic_only':
        run_synthetic_validation(include_ablation=False)
    elif mode == 'kelmarsh_only':
        run_kelmarsh_validation(include_ablation=False)
    elif mode == 'penmanshiel_only':
        run_penmanshiel_validation(include_ablation=False)
    else:
        print(f"Unknown mode: {mode}")
        print("Usage: python3 forecasting_pipeline_v2_3.py [synthetic|kelmarsh|penmanshiel|bogdanci|all]")
        sys.exit(1)

    print("\n" + "=" * 70)
    print("COMPLETE. All results saved to /results/")
    print("=" * 70)
