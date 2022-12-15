import os
import shutil

import pandas as pd


def mkdir(d):
    if not os.path.exists(d):
        os.makedirs(d)


def output_submission(station_prediction: pd.DataFrame, area_prediction: pd.DataFrame, global_prediction: pd.DataFrame,
                      targets):
    """
    Transform the predictions into a suitable format before submission

    inputs:
        station_prediction (pd.DataFrame):
            DataFrame containing the results of the prediction at the station level, must contain the following columns: Station, tod, dow, Available, Charging, Passive, Other
        area_prediction (pd.DataFrame):
            DataFrame containing the results of the prediction at the area level, must contain the following columns: area, tod, dow, Available, Charging, Passive, Other
        global_prediction (pd.DataFrame):
            DataFrame containing the results of the prediction at the global level, must contain the following columns: tod, dow, Available, Charging, Passive, Other
        test_station (pd.DataFrame): DataFrame containing the test.csv data.
    """

    # Creating the submission folder and zip file
    mkdir("output")
    station_prediction[['date', 'area', 'Station'] +
                       targets].to_csv("output/station.csv", index=False)
    area_prediction[['date', 'area'] +
                    targets].to_csv("output/area.csv", index=False)
    global_prediction[['date'] +
                      targets].to_csv("output/global.csv", index=False)

    forecast = {"global": global_prediction,
                "area": area_prediction, "station": station_prediction}
    return (forecast)
