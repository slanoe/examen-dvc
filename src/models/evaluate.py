import pandas as pd
from pathlib import Path
import click
import logging
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pickle
import os
import json


@click.command()
@click.argument('input_filepath', type=click.Path(exists=False), required=0)
@click.argument('output_filepath', type=click.Path(exists=False), required=0)
def main(input_filepath, output_filepath):
    logger = logging.getLogger(__name__)
    logger.info('training model')

    input_filepath_model = click.prompt('Enter the file path for model', type=click.Path(exists=True))
    input_filepath_model = f"{input_filepath_model}/gbr_model.pkl"
    input_filepath = click.prompt('Enter the file path for the input data', type=click.Path(exists=True))
    input_filepath_X_test_scaled = f"{input_filepath}/X_test_scaled.csv"
    input_filepath_y_test = f"{input_filepath}/y_test.csv"
    output_folderpath_predictions = click.prompt('Enter the file path for predictions', type=click.Path())
    output_folderpath_metrics = click.prompt('Enter the file path for metrics', type=click.Path())

    evaluate(input_filepath_model, input_filepath_X_test_scaled, input_filepath_y_test, output_folderpath_predictions, output_folderpath_metrics)

def evaluate(input_filepath_model, input_filepath_X_test_scaled, input_filepath_y_test, output_folderpath_predictions, output_folderpath_metrics):
    # Load model
    with open(input_filepath_model, 'rb') as f:
        model = pickle.load(f)

    # Import datasets
    X_test_scaled = import_dataset(input_filepath_X_test_scaled, sep=",", header=0, encoding='utf-8')
    y_test = import_dataset(input_filepath_y_test, sep=",", header=0, encoding='utf-8')

    # Make predictions
    y_pred = model.predict(X_test_scaled)

    # Save predictions to CSV
    output_filepath_predictions = os.path.join(output_folderpath_predictions, 'prediction.csv')
    pd.DataFrame(y_pred, columns=['prediction']).to_csv(output_filepath_predictions, index=False)

    # Calculate metrics
    scores = {
        'r2_score': r2_score(y_test, y_pred),
        'mse': mean_squared_error(y_test, y_pred),
        'mae': mean_absolute_error(y_test, y_pred)
    }

    # Save metrics to JSON
    output_filepath_metrics = os.path.join(output_folderpath_metrics, 'scores.json')
    with open(output_filepath_metrics, 'w') as f:
        json.dump(scores, f, indent=4)

def import_dataset(file_path, **kwargs):
    return pd.read_csv(file_path, **kwargs)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    main()