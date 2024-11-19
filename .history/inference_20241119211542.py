import torch
import pandas as pd
import datetime
from pipeline import Pipeline, ModelType
from data_loading.standard_dataset import StandardDataset
from data_loading.time_series import TimeSeriesDataframeLoader, TimeInterval

def run_inference(model_path: str, sample_data_path: str):
    # 1. Load the saved model
    pipeline = Pipeline.load_from_file(model_path)

    # 2. Load and prepare sample data
    time_variable = 'utc_timestamp'
    target_variable = 'DE_load_actual_entsoe_transparency'

    # Load data
    loader = TimeSeriesDataframeLoader(
        path_to_csv=sample_data_path,
        time_variable=time_variable,
        target_variable=target_variable
    )

    # Create time interval for your sample data
    sample_interval = TimeInterval(
        min_date=datetime.date(2023, 1, 1),  # Adjust these dates based on your sample data
        max_date=datetime.date(2023, 1, 7)
    )

    # Get sample dataset
    sample_df = loader.extract_dataframe_by_year_filter(sample_interval)

    # 3. Create dataset (using the same parameters as training)
    sample_dataset = StandardDataset(
        df=sample_df,
        time_variable=time_variable,
        target_variable=target_variable,
        time_series_window_in_hours=168,  # Based on your training parameters
        forecasting_horizon_in_hours=96,
        is_single_time_point_prediction=False,
        include_time_information=True,
        time_series_scaler=pipeline.scaler,  # Use the same scaler as training
        is_training_set=False,
        one_hot_time_variables=False
    )

    # 4. Run inference
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pipeline.model.to(device)
    pipeline.model.eval()

    with torch.no_grad():
        for i in range(len(sample_dataset)):
            input_data, _ = sample_dataset[i]
            input_data = input_data.unsqueeze(0).to(device)
            prediction = pipeline.model(input_data)

            # Convert prediction back to original scale if scaler was used
            if pipeline.scaler:
                prediction = pipeline.scaler.inverse_transform(prediction.cpu().numpy())

            print(f"Prediction for timestep {i}:")
            print(prediction)

if __name__ == "__main__":
    model_path = "SimpleNeuralNet2024_11_19_20_51_16_968366"  # Your saved model
    sample_data_path = "path_to_your_sample_data.csv"  # Your sample data CSV

    run_inference(model_path, sample_data_path)