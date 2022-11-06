import pandas as pd

from starting_kit.catboost_model import Catboost_Model
from starting_kit.utils.my_metric import overall_metric
from starting_kit.utils.submit_submission import submit_submission
import datetime
import os
from pathlib import Path

from starting_kit.mean import Mean

# Constants
DATE_LIMIT_VALIDATION = "2021-01-01 00:00:00"
EXPORT = False
INPUT_PATH = Path(__file__).resolve().parents[1] / "Data"


def mkdir(d):
    if not os.path.exists(d):
        os.makedirs(d)


def import_data(name):
    return pd.read_csv(INPUT_PATH / (name + ".csv"), sep=",")


def format_data(station_raw):
    station_raw['date'] = pd.to_datetime(station_raw['date'])
    station_raw['Postcode'] = station_raw['Postcode'].astype(str)
    return station_raw


def generate_area_and_global(data, is_test=False):
    if is_test:
        area_data = test_station.groupby(['date', 'area']).agg({
            'tod': 'max',
            'dow': 'max',
            'Latitude': 'mean',
            'Longitude': 'mean',
            'trend': 'max'}).reset_index()

        global_data = test_station.groupby('date').agg({
            'tod': 'max',
            'dow': 'max',
            'trend': 'max'}).reset_index()

    else:
        area_data = train_station.groupby(['date', 'area']).agg({'Available': 'sum',
                                                                 'Charging': 'sum',
                                                                 'Passive': 'sum',
                                                                 'Other': 'sum',
                                                                 'tod': 'max',
                                                                 'dow': 'max',
                                                                 'Latitude': 'mean',
                                                                 'Longitude': 'mean',
                                                                 'trend': 'max'}).reset_index()

        global_data = train_station.groupby('date').agg({'Available': 'sum',
                                                         'Charging': 'sum',
                                                         'Passive': 'sum',
                                                         'Other': 'sum',
                                                         'tod': 'max',
                                                         'dow': 'max',
                                                         'trend': 'max'}).reset_index()
    return area_data, global_data


def visualisation(train_global, targets):
    mkdir("visualisation")
    train_select = train_global[train_global.tod == 65]
    train_select[['date'] +
                 targets].to_csv("visualisation/train_global.csv", index=False)


def filter_by_date(data_station, data_area, data_global, above_date_limit=True):
    date_limit = datetime.datetime.strptime(DATE_LIMIT_VALIDATION, "%Y-%m-%d %H:%M:%S")
    datasets = (data_station, data_area, data_global)
    if above_date_limit:
        return tuple([data[data.date >= date_limit].reset_index(drop=True) for data in datasets])
    else:
        return tuple([data[data.date < date_limit].reset_index(drop=True) for data in datasets])


def format_validation_filtered(validation_station, validation_area, validation_global, targets):
    validation_datasets = (validation_station, validation_area, validation_global)
    return tuple([data.drop(targets, axis=1) for data in validation_datasets])


def split_dataset(data_station, data_area, data_global, threshold):
    datasets = (data_station, data_area, data_global)
    validation_datasets = []
    for dataset in datasets:
        n = len(dataset)
        n_train = threshold * n
        validation_dataset = dataset.loc[n_train:n, ]
        validation_datasets.append(validation_dataset)
    return validation_datasets


if __name__ == "__main__":
    targets = ["Available", "Charging", "Passive", "Other"]
    station_features = ['Station', 'tod', 'dow', 'area'] + \
                       ['trend', 'Latitude', 'Longitude']  # temporal and spatial inputs
    area_features = ['area', 'tod', 'dow'] + ['trend',
                                              'Latitude', 'Longitude']  # temporal and spatial inputs
    global_features = ['tod', 'dow'] + ['trend']  # temporal input

    train_station_raw = import_data("train")
    train_station = format_data(train_station_raw)

    train_area, train_global = generate_area_and_global(train_station)

    validation_station, validation_area, validation_global = split_dataset(train_station, train_area,
                                                                           train_global, threshold=0.8)

    model_station = Catboost_Model(train_data=train_station,
                                   test=validation_station,
                                   features=station_features,
                                   cat_features=[0, 1, 2, 3],
                                   targets=targets,
                                   learning_rate=0.1,
                                   level_col="Station")

    model_area = Catboost_Model(train_data=train_area,
                                test=validation_area,
                                features=area_features,
                                cat_features=[0, 1, 2],
                                targets=targets,
                                learning_rate=0.1,
                                level_col="area")

    model_global = Catboost_Model(train_data=train_global,
                                  test=validation_global,
                                  features=global_features,
                                  cat_features=[0, 1],
                                  targets=targets,
                                  learning_rate=0.1,
                                  level_col="global")

    if EXPORT:
        # Instantiate the models
        # model_station = Mean(train_station, type="station")
        # model_area = Mean(train_area, type="area")
        # model_global = Mean(train_global, type="global")

        # Train the models
        model_station.train()
        model_area.train()
        model_global.train()

        # Import and process test dataset
        test_station_raw = import_data("test")
        test_station = format_data(test_station_raw)
        test_area, test_global = generate_area_and_global(test_station, is_test=True)

        # Run predictions on test dataset
        prediction_station = model_station.predict(test_station)
        prediction_area = model_area.predict(test_area)
        prediction_global = model_global.predict(test_global)

        # Format predictions before submitting it
        submit_submission(prediction_station, prediction_area, prediction_global, targets)
    else:
        # validation_station, validation_area, validation_global = filter_by_date(train_station, train_area,
        #                                                                         train_global, above_date_limit=True)
        # validation_station_filtered, validation_area_filtered, validation_global_filtered = format_validation_filtered(
        #     validation_station, validation_area, validation_global, targets)
        #
        # train_station, train_area, train_global = filter_by_date(train_station, train_area,
        #                                                          train_global, above_date_limit=False)
        # Instantiate the models
        # model_station = Mean(train_station, type="station")
        # model_area = Mean(train_area, type="area")
        # model_global = Mean(train_global, type="global")

        # train_station, train_area, train_global = filter_by_date(train_station, train_area,
        #                                                          train_global, above_date_limit=False)

        # Train the models
        model_station.train()
        model_area.train()
        model_global.train()

        validation_station_filtered, validation_area_filtered, validation_global_filtered = format_validation_filtered(
            validation_station, validation_area, validation_global, targets)

        # Run predictions on test dataset
        prediction_station = model_station.predict(validation_station_filtered)
        prediction_area = model_area.predict(validation_area_filtered)
        prediction_global = model_global.predict(validation_global_filtered)

        metric = overall_metric(validation_station, validation_area, validation_global, prediction_station,
                                prediction_area,
                                prediction_global)

        print("The metric is: " + str(metric))

    print("Completed!")
