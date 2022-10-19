import pandas as pd
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


if __name__ == "__main__":
    targets = ["Available", "Charging", "Passive", "Other"]

    train_station_raw = import_data("train")
    train_station = format_data(train_station_raw)

    train_area, train_global = generate_area_and_global(train_station)

    if EXPORT:
        model = Mean(train_station, train_area, train_global)
        model.train()
        test_station_raw = import_data("test")
        test_station = format_data(test_station_raw)
        test_area, test_global = generate_area_and_global(test_station, is_test=True)
        prediction_station, prediction_area, prediction_global = model.predict(test_station, test_area, test_global)
        submit_submission(prediction_station, prediction_area, prediction_global, targets)
    else:
        validation_station, validation_area, validation_global = filter_by_date(train_station, train_area,
                                                                                train_global, above_date_limit=True)
        validation_station_filtered, validation_area_filtered, validation_global_filtered = format_validation_filtered(
            validation_station, validation_area, validation_global, targets)

        train_station, train_area, train_global = filter_by_date(train_station, train_area,
                                                                 train_global, above_date_limit=False)

        model = Mean(train_station, train_area, train_global)
        model.train()
        prediction_station, prediction_area, prediction_global = model.predict(validation_station_filtered, validation_area_filtered, validation_global_filtered)
        metric = overall_metric(validation_station, validation_area, validation_global, prediction_station, prediction_area,
                       prediction_global)

        print("The metric is: " + str(metric))

    print("Completed!")
