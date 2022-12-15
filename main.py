import os

from src.generate_data import import_data, format_data, generate_area_and_global, split_dataset
from src.models.catboost_model import Catboost_Model
from src.submission import output_submission

DATA_DIR = os.path.abspath('data')

if __name__ == "__main__":
    # Define the targets & features
    targets = ["Available", "Charging", "Passive", "Other"]

    station_features = ['Station',  'tod', 'dow', 'area', 'trend', 'Latitude',
                        'Longitude']  # temporal and spatial inputs
    area_features = ['area', 'tod', 'dow', 'trend', 'Latitude', 'Longitude']  # temporal and spatial inputs
    global_features = ['tod', 'dow', 'trend']  # temporal inputs

    # Import the training dataset
    train_station_raw = import_data("train", DATA_DIR)

    # Format the training dataset
    train_station = format_data(train_station_raw)

    # Generate area and global datasets from the training data at station level
    train_area, train_global = generate_area_and_global(train_station)

    # Split training dataset to create a testing dataset for catboost
    train_station, early_stop_station = split_dataset(train_station, threshold=0.8, keep_initial_dataset=True)
    train_area, early_stop_area = split_dataset(train_area, threshold=0.8, keep_initial_dataset=True)
    train_global, early_stop_global = split_dataset(train_global, threshold=0.8, keep_initial_dataset=True)

    # Instantiate the models
    model_station = Catboost_Model(train_data=train_station,
                                   early_stop=early_stop_station,
                                   features=station_features,
                                   cat_features=[0, 1, 2, 3],
                                   targets=targets,
                                   learning_rate=0.1,
                                   iterations=200,
                                   depth=5,
                                   level_col="Station",
                                   expo_loss=True)

    model_area = Catboost_Model(train_data=train_area,
                                early_stop=early_stop_area,
                                features=area_features,
                                cat_features=[0, 1, 2],
                                targets=targets,
                                learning_rate=0.1,
                                iterations=200,
                                depth=5,
                                level_col="area",
                                expo_loss=True)

    model_global = Catboost_Model(train_data=train_global,
                                  early_stop=early_stop_global,
                                  features=global_features,
                                  cat_features=[0, 1],
                                  targets=targets,
                                  learning_rate=0.1,
                                  iterations=200,
                                  depth=5,
                                  level_col="global",
                                  expo_loss=True)

    # Train the models
    model_station.train()
    model_area.train()
    model_global.train()

    # Import the testing dataset
    test_station_raw = import_data("test", DATA_DIR)

    # Format the test dataset columns to the expected types
    test_station = format_data(test_station_raw)

    # Generate area and global datasets from the testing data at station level
    test_area, test_global = generate_area_and_global(test_station, is_test=True)

    # Make predictions on testing dataset
    prediction_station = model_station.predict(test_station)
    prediction_area = model_area.predict(test_area)
    prediction_global = model_global.predict(test_global)

    # Generate the csv files containing the predictions
    output_submission(prediction_station, prediction_area, prediction_global, targets)

    print("Completed!\n", "Check the results in the output folder.")
