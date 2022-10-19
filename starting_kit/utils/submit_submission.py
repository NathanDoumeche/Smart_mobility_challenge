import os
import pandas as pd
import shutil


def mkdir(d):
    if not os.path.exists(d):
        os.makedirs(d)


def submit_submission(station_prediction: pd.DataFrame, area_prediction: pd.DataFrame, global_prediction: pd.DataFrame, targets):
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
    mkdir("sample_result_submission")
    station_prediction[['date', 'area', 'Station'] +
                       targets].to_csv("sample_result_submission/station.csv", index=False)
    area_prediction[['date', 'area'] +
                    targets].to_csv("sample_result_submission/area.csv", index=False)
    global_prediction[['date'] +
                      targets].to_csv("sample_result_submission/global.csv", index=False)

    shutil.make_archive("sample_result_submission",
                        'zip', 'sample_result_submission')

    forecast = {"global": global_prediction, "area": area_prediction, "station": station_prediction}
    return(forecast)