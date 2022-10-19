# import numpy as np
import pandas as pd

def sae(y_true, y_pred):
    """Sum of Absolute errors"""
    return(sum(abs(y_true-y_pred)))


def overall_metric(validation_station, validation_area, validation_global, prediction_station, prediction_area, prediction_global):
    """Overall metric"""

    filter_dates = validation_station[["date"]]

    ### Target list
    targets = [ "Available", "Charging", "Passive", "Other"]

    ### Number of timesteps in the test set
    N = len(validation_global)

    ### Initiating scores list
    scores = []
    ### Filtering dates
    validation_global['date'] = pd.to_datetime(validation_global['date'])
    validation_area['date'] = pd.to_datetime(validation_area['date'])
    validation_global['date'] = pd.to_datetime(validation_station['date'])

    prediction_global['date'] = pd.to_datetime(prediction_global['date'])
    prediction_area['date'] = pd.to_datetime(prediction_area['date'])
    prediction_station['date'] = pd.to_datetime(prediction_station['date'])

    validation_global = validation_global.loc[validation_global['date'].isin(filter_dates['date'])].sort_values(by = 'date', ascending=True).reset_index(drop=True)
    validation_area = validation_area.loc[validation_area['date'].isin(filter_dates['date'])].sort_values(by = 'date', ascending=True).reset_index(drop=True)
    validation_station = validation_station.loc[validation_station['date'].isin(filter_dates['date'])].sort_values(by = 'date', ascending=True).reset_index(drop=True)

    prediction_global = prediction_global.loc[prediction_global['date'].isin(filter_dates['date'])].sort_values(by = 'date', ascending=True).reset_index(drop=True)
    prediction_area = prediction_area.loc[prediction_area['date'].isin(filter_dates['date'])].sort_values(by = 'date', ascending=True).reset_index(drop=True)
    prediction_station = prediction_station.loc[prediction_station['date'].isin(filter_dates['date'])].sort_values(by = 'date', ascending=True).reset_index(drop=True)
    for target in targets:
        scores.append( (sae(validation_global[target],prediction_global[target]) +
                        sae(validation_area[target],prediction_area[target]) +
                        sae(validation_station[target],prediction_station[target]))/N 
                      )
    return(sum(scores))
