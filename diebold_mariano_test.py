"""
================================================================================
Diebold-Mariano Test for Forecast Accuracy Comparison
================================================================================
Implements the Diebold-Mariano (1995) test with Harvey-Leybourne-Newbold (1997)
small-sample correction for comparing predictive accuracy of competing forecasts.

REFERENCE:
  Diebold, F.X. and Mariano, R.S. (1995). "Comparing predictive accuracy."
  J. of Business & Economic Statistics, 13(3), 253-263.
  
  Harvey, D., Leybourne, S., and Newbold, P. (1997). "Testing the equality of
  prediction mean squared errors." Int. J. of Forecasting, 13(2), 281-291.

USAGE (after running pipeline with prediction-saving enabled):

  python3 diebold_mariano_test.py
  
This automatically reads all *_predictions_seed0.csv files in results/
and produces DM tests for LSTM-vs-MLP, LSTM-vs-Persistence, MLP-vs-LR
on each available dataset.
================================================================================
"""

import numpy as np
import pandas as pd
from scipy import stats
import os
import glob
import sys


def diebold_mariano(actual, pred1, pred2, h=1, loss='squared'):
    """
    Diebold-Mariano test with Harvey small-sample correction.
    
    Returns: dict with statistic, p-value (Harvey-corrected), interpretation
    """
    actual = np.asarray(actual)
    pred1 = np.asarray(pred1)
    pred2 = np.asarray(pred2)
    
    if loss == 'squared':
        loss1 = (actual - pred1) ** 2
        loss2 = (actual - pred2) ** 2
    elif loss == 'absolute':
        loss1 = np.abs(actual - pred1)
        loss2 = np.abs(actual - pred2)
    else:
        raise ValueError("loss must be 'squared' or 'absolute'")
    
    d = loss1 - loss2
    n = len(d)
    d_mean = np.mean(d)
    d_var = np.var(d, ddof=1)
    
    # Standard DM statistic
    dm_stat = d_mean / np.sqrt(d_var / n)
    
    # Harvey-Leybourne-Newbold (1997) small-sample correction
    correction = np.sqrt((n + 1 - 2*h + h*(h-1)/n) / n)
    dm_corrected = dm_stat * correction
    
    # Use Student-t distribution with n-1 degrees of freedom (Harvey recommendation)
    p_value = 2 * (1 - stats.t.cdf(np.abs(dm_corrected), df=n - 1))
    
    if p_value < 0.001:
        sig = "highly significant (p < 0.001)"
    elif p_value < 0.01:
        sig = "significant (p < 0.01)"
    elif p_value < 0.05:
        sig = "significant (p < 0.05)"
    elif p_value < 0.10:
        sig = "marginally significant (p < 0.10)"
    else:
        sig = "not significant (p >= 0.10)"
    
    return {
        'DM_statistic': dm_stat,
        'DM_corrected': dm_corrected,
        'p_value': p_value,
        'significance': sig,
        'mean_loss_diff': d_mean,
        'n': n
    }


def run_pairwise_tests(actual, predictions, model_names, dataset_name):
    """Run all pairwise DM tests for a dataset."""
    print(f"\n{'=' * 75}")
    print(f"DATASET: {dataset_name.upper()}")
    print(f"{'=' * 75}")
    print(f"Test set size: {len(actual)} hourly observations\n")
    
    # Compute RMSEs for context
    print(f"  {'Model':<25} {'RMSE (kW)':>12}    {'nRMSE (%)':>10}")
    print(f"  {'-' * 25} {'-' * 12}    {'-' * 10}")
    
    # Determine rated power from dataset name
    if 'kelmarsh' in dataset_name.lower() or 'penmanshiel' in dataset_name.lower():
        rated_power = 2050
    else:
        rated_power = 2300
    
    rmses = {}
    for name, pred in zip(model_names, predictions):
        rmse = np.sqrt(np.mean((actual - pred) ** 2))
        nrmse = (rmse / rated_power) * 100
        rmses[name] = rmse
        print(f"  {name:<25} {rmse:>10.2f}      {nrmse:>10.2f}")
    
    print()
    
    # Define pairs to test
    pairs = []
    if 'LSTM' in model_names and 'MLP' in model_names:
        pairs.append(('LSTM', 'MLP'))
    if 'LSTM' in model_names and 'Persistence' in model_names:
        pairs.append(('LSTM', 'Persistence'))
    if 'MLP' in model_names and 'Persistence' in model_names:
        pairs.append(('MLP', 'Persistence'))
    
    results_list = []
    
    for model1_name, model2_name in pairs:
        idx1 = model_names.index(model1_name)
        idx2 = model_names.index(model2_name)
        
        print(f"--- {model1_name} vs {model2_name} ---")
        
        # Squared loss
        result = diebold_mariano(actual, predictions[idx1], predictions[idx2],
                                  h=1, loss='squared')
        
        if result['DM_corrected'] < 0:
            direction = f"{model1_name} has LOWER errors"
        else:
            direction = f"{model2_name} has LOWER errors"
        
        diff_rmse = rmses[model1_name] - rmses[model2_name]
        
        print(f"  Squared-loss DM (Harvey-corrected): {result['DM_corrected']:.3f}")
        print(f"  p-value: {result['p_value']:.4f} -- {result['significance']}")
        print(f"  Direction: {direction}")
        print(f"  RMSE difference: {diff_rmse:+.2f} kW")
        print()
        
        results_list.append({
            'dataset': dataset_name,
            'comparison': f'{model1_name}_vs_{model2_name}',
            'DM_corrected': round(result['DM_corrected'], 3),
            'p_value': round(result['p_value'], 4),
            'significant_at_5pct': result['p_value'] < 0.05,
            'rmse_diff_kW': round(diff_rmse, 2),
            'direction': direction,
            'n': result['n']
        })
    
    return results_list


def main():
    results_dir = 'results' if os.path.exists('results') else '.'
    
    # Find all prediction files
    pattern = os.path.join(results_dir, '*_predictions_seed0.csv')
    pred_files = sorted(glob.glob(pattern))
    
    if not pred_files:
        print(f"\nNo prediction files found matching: {pattern}")
        print(f"\nFirst run pipeline with prediction-saving enabled:")
        print(f"  python3 forecasting_pipeline_v2_3.py kelmarsh")
        print(f"  python3 forecasting_pipeline_v2_3.py penmanshiel")
        print(f"  python3 forecasting_pipeline_v2_3.py synthetic\n")
        return 1
    
    print(f"\n{'#' * 75}")
    print(f"# Diebold-Mariano Test Suite")
    print(f"# Found {len(pred_files)} prediction file(s)")
    print(f"{'#' * 75}")
    
    all_results = []
    
    for pred_file in pred_files:
        # Extract dataset name from filename (e.g., kelmarsh_full_predictions_seed0.csv -> kelmarsh)
        basename = os.path.basename(pred_file)
        dataset_name = basename.replace('_predictions_seed0.csv', '').replace('_full', '')
        
        df = pd.read_csv(pred_file)
        
        # Need actual + lstm_pred + mlp_pred at minimum
        if 'actual' not in df.columns or 'lstm_pred' not in df.columns:
            print(f"\nSkipping {basename}: missing required columns")
            continue
        
        actual = df['actual'].values
        
        predictions = [df['lstm_pred'].values]
        model_names = ['LSTM']
        
        if 'mlp_pred' in df.columns:
            predictions.append(df['mlp_pred'].values)
            model_names.append('MLP')
        
        if 'lr_pred' in df.columns:
            predictions.append(df['lr_pred'].values)
            model_names.append('LinearRegression')
        
        # Add persistence (P(t+1) = P(t)), shift by one
        # The 'actual' values are already the targets P(t); persistence forecast is P(t-1)
        # But since we have hourly aligned data, persistence = actual shifted by 1
        persistence_pred = np.concatenate([[actual[0]], actual[:-1]])
        predictions.append(persistence_pred)
        model_names.append('Persistence')
        
        results = run_pairwise_tests(actual, predictions, model_names, dataset_name)
        all_results.extend(results)
    
    # Save aggregated results
    if all_results:
        df_out = pd.DataFrame(all_results)
        out_path = os.path.join(results_dir, 'diebold_mariano_results.csv')
        df_out.to_csv(out_path, index=False)
        
        print(f"\n{'#' * 75}")
        print(f"# AGGREGATED RESULTS")
        print(f"{'#' * 75}")
        print(df_out.to_string(index=False))
        print(f"\nSaved: {out_path}")
        
        # Generate paper text
        print(f"\n{'#' * 75}")
        print(f"# TEXT FOR PAPER (Section V):")
        print(f"{'#' * 75}\n")
        
        for _, row in df_out.iterrows():
            comp = row['comparison'].replace('_vs_', ' vs ')
            sig_text = "statistically significant" if row['significant_at_5pct'] else "not statistically significant"
            print(f"  [{row['dataset'].title()}] {comp}: DM = {row['DM_corrected']} (p = {row['p_value']:.3f}), {sig_text}.")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
