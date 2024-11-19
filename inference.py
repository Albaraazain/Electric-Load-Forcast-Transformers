import argparse
import pickle
from pathlib import Path

import torch
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error

from models.tranformers.transformer import TimeSeriesTransformer
from pipeline import Pipeline, ModelType, TARGET_VARIABLE, UTC_TIMESTAMP
from data_loading.time_series import TimeSeriesDataframeLoader, TimeInterval
from data_loading.transformer_dataset import TransformerDataset
from training.tranformer_trainer import create_mask

def create_separate_horizon_plots(df, save_dir="results/plots"):
    """Create separate plots for each prediction horizon"""
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # Set style for better visualization
    plt.style.use('seaborn')

    # 1. 1-hour ahead predictions
    plt.figure(figsize=(15, 6))
    plt.plot(df.index, df['actual'], label='Actual', color='black', alpha=0.7)
    plt.plot(df.index, df['prediction_1h'], label='1h Prediction', color='blue', alpha=0.7)
    plt.title('1-Hour Ahead Load Forecast vs Actual')
    plt.xlabel('Time')
    plt.ylabel('Load (MW)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/1h_prediction.png")
    plt.close()

    # 2. 24-hour ahead predictions
    plt.figure(figsize=(15, 6))
    plt.plot(df.index, df['actual'], label='Actual', color='black', alpha=0.7)
    plt.plot(df.index, df['prediction_24h'], label='24h Prediction', color='orange', alpha=0.7)
    plt.title('24-Hour Ahead Load Forecast vs Actual')
    plt.xlabel('Time')
    plt.ylabel('Load (MW)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/24h_prediction.png")
    plt.close()

    # 3. 48-hour ahead predictions
    plt.figure(figsize=(15, 6))
    plt.plot(df.index, df['actual'], label='Actual', color='black', alpha=0.7)
    plt.plot(df.index, df['prediction_48h'], label='48h Prediction', color='green', alpha=0.7)
    plt.title('48-Hour Ahead Load Forecast vs Actual')
    plt.xlabel('Time')
    plt.ylabel('Load (MW)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/48h_prediction.png")
    plt.close()

    # 4. 72-hour ahead predictions
    plt.figure(figsize=(15, 6))
    plt.plot(df.index, df['actual'], label='Actual', color='black', alpha=0.7)
    plt.plot(df.index, df['prediction_72h'], label='72h Prediction', color='red', alpha=0.7)
    plt.title('72-Hour Ahead Load Forecast vs Actual')
    plt.xlabel('Time')
    plt.ylabel('Load (MW)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/72h_prediction.png")
    plt.close()

    # 5. 96-hour ahead predictions
    plt.figure(figsize=(15, 6))
    plt.plot(df.index, df['actual'], label='Actual', color='black', alpha=0.7)
    plt.plot(df.index, df['prediction_96h'], label='96h Prediction', color='purple', alpha=0.7)
    plt.title('96-Hour Ahead Load Forecast vs Actual')
    plt.xlabel('Time')
    plt.ylabel('Load (MW)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/96h_prediction.png")
    plt.close()

    # 6. Error Analysis
    plt.figure(figsize=(15, 6))
    horizons = ['1h', '24h', '48h', '72h', '96h']
    error_means = []
    error_stds = []

    for horizon in horizons:
        error = df[f'prediction_{horizon}'] - df['actual']
        error_means.append(np.mean(np.abs(error)))
        error_stds.append(np.std(error))

    plt.bar(horizons, error_means, yerr=error_stds, alpha=0.7)
    plt.title('Mean Absolute Error by Prediction Horizon')
    plt.xlabel('Prediction Horizon')
    plt.ylabel('Mean Absolute Error (MW)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/error_analysis.png")
    plt.close()

    # 7. Weekly Pattern Analysis
    weekly_data = df.copy()
    weekly_data['hour'] = weekly_data.index.hour
    weekly_data['day'] = weekly_data.index.dayofweek

    plt.figure(figsize=(15, 8))
    pivot_table = weekly_data.pivot_table(
        values=['actual', 'prediction_24h'],
        index='hour',
        columns='day',
        aggfunc='mean'
    )

    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    for day_num in range(7):
        plt.subplot(2, 4, day_num+1)
        plt.plot(pivot_table['actual'][day_num], label='Actual', color='black', alpha=0.7)
        plt.plot(pivot_table['prediction_24h'][day_num], label='Predicted', color='orange', alpha=0.7)
        plt.title(days[day_num])
        plt.xlabel('Hour')
        plt.ylabel('Load (MW)')
        if day_num == 0:
            plt.legend()

    plt.tight_layout()
    plt.savefig(f"{save_dir}/weekly_patterns.png")
    plt.close()


def load_model_and_scaler(model_path: str):
    """Load the saved model and scaler"""
    print("Loading model and scaler...")
    # Add TimeSeriesTransformer to safe globals
    torch.serialization.add_safe_globals([TimeSeriesTransformer])
    model = torch.load(model_path, weights_only=False)

    with open(model_path + ".scaler", "rb") as f:
        scaler = pickle.load(f)
    return model, scaler


def evaluate_predictions(actual, predicted, horizon):
    """Calculate error metrics"""
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    return {
        f'MAE_{horizon}h': mae,
        f'RMSE_{horizon}h': rmse,
        f'MAPE_{horizon}h': mape
    }


def plot_predictions_vs_actual(timestamps, actual, predictions, save_dir="results/plots"):
    """Create comparison plots"""
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # 1. Actual vs Predicted Load
    plt.figure(figsize=(15, 8))
    plt.plot(timestamps, actual, label='Actual Load', alpha=0.7)
    plt.plot(timestamps, predictions['prediction_1h'], label='1h Prediction', alpha=0.7)
    plt.plot(timestamps, predictions['prediction_24h'], label='24h Prediction', alpha=0.7)
    plt.plot(timestamps, predictions['prediction_96h'], label='96h Prediction', alpha=0.7)
    plt.title('Actual vs Predicted Load')
    plt.xlabel('Time')
    plt.ylabel('Load (MW)')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/actual_vs_predicted.png")
    plt.close()

    # 2. Error Distribution
    plt.figure(figsize=(15, 8))
    for horizon in ['1h', '24h', '96h']:
        error = predictions[f'prediction_{horizon}'] - actual
        sns.kdeplot(error, label=f'{horizon} Prediction Error', alpha=0.7)
    plt.title('Prediction Error Distribution')
    plt.xlabel('Error (MW)')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{save_dir}/error_distribution.png")
    plt.close()

    # 3. Daily Pattern Analysis
    daily_actual = pd.DataFrame({'timestamp': timestamps, 'actual': actual})
    daily_actual['hour'] = daily_actual['timestamp'].dt.hour
    hourly_pattern = daily_actual.groupby('hour')['actual'].mean()

    plt.figure(figsize=(12, 6))
    plt.plot(hourly_pattern.index, hourly_pattern.values, marker='o')
    plt.title('Average Daily Load Pattern')
    plt.xlabel('Hour of Day')
    plt.ylabel('Average Load (MW)')
    plt.grid(True)
    plt.savefig(f"{save_dir}/daily_pattern.png")
    plt.close()


def run_inference(model_path: str, sample_data_path: str):
    # Load model and data
    model, scaler = load_model_and_scaler(model_path)

    # Set up arguments
    args = argparse.Namespace()
    args.forecasting_horizon = 96
    args.predict_single_value = False
    args.time_series_window = 168
    args.include_time_context = True
    args.transformer_labels_count = 24

    # Load data
    loader = TimeSeriesDataframeLoader(
        Path(sample_data_path),
        UTC_TIMESTAMP,
        TARGET_VARIABLE
    )

    # Create time interval
    sample_interval = TimeInterval(
        min_date=datetime(2019, 1, 1).date(),
        max_date=datetime(2019, 3, 31).date()
    )
    # Get sample dataset
    sample_df = loader.extract_dataframe_by_year_filter(sample_interval)
    print(f"Loaded dataframe with shape: {sample_df.shape}")

    # Create dataset
    sample_dataset = TransformerDataset(
        df=sample_df,
        time_variable=UTC_TIMESTAMP,
        target_variable=TARGET_VARIABLE,
        time_series_window_in_hours=args.time_series_window,
        forecasting_horizon_in_hours=args.forecasting_horizon,
        labels_count=args.transformer_labels_count,
        is_single_time_point_prediction=args.predict_single_value,
        include_time_information=args.include_time_context,
        time_series_scaler=scaler,
        is_training_set=False
    )

    # Run inference
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    model.eval()

    predictions = []
    actual_values = []
    timestamps = []

    with torch.no_grad():
        for i in range(len(sample_dataset)):
            encoder_input, decoder_input = sample_dataset[i]

            # Get actual values
            actual = decoder_input[-args.forecasting_horizon:, 0].cpu().numpy()
            actual = scaler.inverse_transform(actual.reshape(-1, 1)).flatten()

            # Make predictions
            encoder_input = encoder_input.unsqueeze(0).to(device)
            decoder_input = decoder_input.unsqueeze(0).to(device)

            mask = create_mask(args.transformer_labels_count + args.forecasting_horizon).to(device)
            prediction = model(encoder_input, decoder_input, tgt_mask=mask)
            prediction = prediction[0, -args.forecasting_horizon:, 0]

            prediction = scaler.inverse_transform(prediction.cpu().numpy().reshape(-1, 1)).flatten()

            predictions.append(prediction)
            actual_values.append(actual)
            timestamps.append(sample_dataset.time_labels[i])

            if i % 100 == 0:
                print(f"Completed prediction {i + 1}/{len(sample_dataset)}")

    # Convert to DataFrame
    results_df = pd.DataFrame({
        'timestamp': timestamps,
        'actual': [a[0] for a in actual_values],
        'prediction_1h': [p[0] for p in predictions],
        'prediction_24h': [p[23] for p in predictions],
        'prediction_48h': [p[47] for p in predictions],
        'prediction_72h': [p[71] for p in predictions],
        'prediction_96h': [p[-1] for p in predictions]
    })



    # Calculate error metrics
    metrics = {}
    for horizon in ['1h', '24h', '48h', '72h', '96h']:
        metrics.update(evaluate_predictions(
            results_df['actual'],
            results_df[f'prediction_{horizon}'],
            horizon
        ))

    # Create visualizations
    plot_predictions_vs_actual(timestamps,
                               results_df['actual'],
                               results_df)

    # Save results
    results_df.to_csv("results/predictions_with_actual.csv", index=False)

    # After getting predictions
    results_df['timestamp'] = pd.to_datetime(results_df['timestamp'])
    results_df.set_index('timestamp', inplace=True)

    # Create visualizations
    create_separate_horizon_plots(results_df)

    return results_df, metrics


if __name__ == "__main__":
    try:
        model_path = "data/models/TimeSeriesTransformer"
        sample_data_path = "data/opsd-time_series-2020-10-06/time_series_15min_singleindex.csv"

        results_df, metrics = run_inference(model_path, sample_data_path)

        print("\n=== Model Performance Metrics ===")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.2f}")



        print("\nResults have been saved to 'results' directory")
        print("\nSample of predictions (first 5 rows):")
        print(results_df.head())



    except Exception as e:
        print(f"Error occurred: {str(e)}")
        import traceback

        traceback.print_exc()
