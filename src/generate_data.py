from pathlib import Path

import pandas as pd

# Define static parameter
INPUT_PATH = "../data/"
DATE_LIMIT_VALIDATION = "2021-01-01 00:00:00"


def import_data(name, data_path):
    """Import files from the data folder"""
    try:
        return pd.read_csv(Path(data_path).joinpath(name + ".csv"), sep=",")
    except Exception as e:
        print(e)
        print("The datasets can be downloaded here: https://codalab.lisn.upsaclay.fr/my/datasets/download/4b524c74-e8b9-4630-b3f9-4eb841677538\n",
              "You should place the files in the data folder.")


def format_data(station_raw):
    """Format dataset column types"""
    station_raw['date'] = pd.to_datetime(station_raw['date'])
    station_raw['Postcode'] = station_raw['Postcode'].astype(str)
    return station_raw


def generate_area_and_global(data, is_test=False):
    """Generate dataframes at the area and global level"""
    if is_test:
        area_data = data.groupby(['date', 'area']).agg(
            {'tod': 'max',
             'dow': 'max',
             'Latitude': 'mean',
             'Longitude': 'mean',
             'trend': 'max'}
        ).reset_index()

        global_data = data.groupby('date').agg(
            {'tod': 'max',
             'dow': 'max',
             'trend': 'max'}
        ).reset_index()

    else:
        area_data = data.groupby(['date', 'area']).agg(
            {'Available': 'sum',
             'Charging': 'sum',
             'Passive': 'sum',
             'Other': 'sum',
             'tod': 'max',
             'dow': 'max',
             'Latitude': 'mean',
             'Longitude': 'mean',
             'trend': 'max'}
        ).reset_index()

        global_data = data.groupby('date').agg(
            {'Available': 'sum',
             'Charging': 'sum',
             'Passive': 'sum',
             'Other': 'sum',
             'tod': 'max',
             'dow': 'max',
             'trend': 'max'}
        ).reset_index()
    return area_data, global_data


def split_dataset(data, threshold, keep_initial_dataset=False):
    """Split dataset into two parts at the threshold"""
    n = len(data)
    n_split = threshold * n
    data_left = data.loc[:n_split, ]
    data_right = data.loc[n_split:n, ]
    if keep_initial_dataset:
        return data, data_right
    else:
        return data_left, data_right


def remove_target_columns(data, targets):
    """Remove the target columns from the dataset"""
    return data.drop(targets, axis=1)
