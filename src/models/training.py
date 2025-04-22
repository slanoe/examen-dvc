import pandas as pd
from pathlib import Path
import click
import logging
from sklearn.ensemble import RandomForestRegressor
import pickle
import os


@click.command()
@click.argument('input_filepath', type=click.Path(exists=False), required=0)
@click.argument('output_filepath', type=click.Path(exists=False), required=0)
def main(input_filepath, output_filepath):
    logger = logging.getLogger(__name__)
    logger.info('training model')

    input_filepath_best_params = click.prompt('Enter the file path for best params', type=click.Path(exists=True))
    input_filepath_best_params = f"{input_filepath_best_params}/best_params.pkl"
    input_filepath = click.prompt('Enter the file path for the input data', type=click.Path(exists=True))
    input_filepath_X_train_scaled = f"{input_filepath}/X_train_scaled.csv"
    input_filepath_y_train = f"{input_filepath}/y_train.csv"
    output_folderpath = click.prompt('Enter the file path for model', type=click.Path())

    training(input_filepath_best_params, input_filepath_X_train_scaled, input_filepath_y_train, output_folderpath)

def training(input_filepath_best_params, input_filepath_X_train_scaled, input_filepath_y_train, output_folderpath):

    # Load best parameters
    with open(input_filepath_best_params, 'rb') as f:
        best_params = pickle.load(f)

    # Import datasets
    X_train_scaled = import_dataset(input_filepath_X_train_scaled, sep=",", header=0, encoding='utf-8')
    y_train = import_dataset(input_filepath_y_train, sep=",", header=0, encoding='utf-8')

    # Create and train the model with best parameters
    rf_model = RandomForestRegressor(**best_params, random_state=42)
    rf_model.fit(X_train_scaled, y_train.values.ravel())

    # Save the trained model
    output_filepath = os.path.join(output_folderpath, 'gbr_model.pkl')
    with open(output_filepath, 'wb') as f:
        pickle.dump(rf_model, f)

def import_dataset(file_path, **kwargs):
    return pd.read_csv(file_path, **kwargs)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    main()