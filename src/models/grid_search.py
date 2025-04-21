import pandas as pd
from pathlib import Path
import click
import logging
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
import pickle
import os

@click.command()
@click.argument('input_filepath', type=click.Path(exists=False), required=0)
@click.argument('output_filepath', type=click.Path(exists=False), required=0)
def main(input_filepath, output_filepath):
    logger = logging.getLogger(__name__)
    logger.info('normalize data')

    input_filepath = click.prompt('Enter the file path for the input data', type=click.Path(exists=True))
    input_filepath_X_train_scaled = f"{input_filepath}/X_train_scaled.csv"
    input_filepath_y_train = f"{input_filepath}/y_train.csv"
    output_filepath = click.prompt('Enter the file path for best params (e.g., models/best_params.pkl)', type=click.Path())

    grid_search(input_filepath_X_train_scaled, input_filepath_y_train, output_filepath)

def grid_search(input_filepath_X_train_scaled, input_filepath_y_train, output_folderpath):
    # Import datasets
    X_train_scaled = import_dataset(input_filepath_X_train_scaled, sep=",", header=0, encoding='utf-8')
    y_train = import_dataset(input_filepath_y_train, sep=",", header=0, encoding='utf-8')

    # Define parameter grid
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    # Initialize the model
    rf = RandomForestRegressor(random_state=42)

    # Create GridSearchCV object
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=5,
        n_jobs=-1,
        scoring='r2'
    )

    # Fit the grid search
    grid_search.fit(X_train_scaled, y_train.values.ravel())

    # Save best parameters
    output_filepath = os.path.join(output_folderpath, 'best_params.pkl')
    with open(output_filepath, 'wb') as f:
        pickle.dump(grid_search.best_params_, f)

def import_dataset(file_path, **kwargs):
    return pd.read_csv(file_path, **kwargs)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    main()