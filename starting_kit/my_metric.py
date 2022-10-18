# import numpy as np
import pandas as pd

def sae(y_true, y_pred):
    """Sum of Absolute errors"""
    return(sum(abs(y_true-y_pred)))


def overall_metric(actual_dict, predicted_dict, filter_dates):
    """Overall metric"""

    ### Target list
    targets = [ "Available", "Charging", "Passive", "Other"]

    ### Number of timesteps in the test set
    N = len(actual_dict['global'])

    ### Initiating scores list
    scores = []
    ### Filtering dates
    actual_dict['global']['date'] = pd.to_datetime(actual_dict['global']['date'])
    actual_dict['area']['date'] = pd.to_datetime(actual_dict['area']['date'])
    actual_dict['station']['date'] = pd.to_datetime(actual_dict['station']['date'])

    predicted_dict['global']['date'] = pd.to_datetime(predicted_dict['global']['date'])
    predicted_dict['area']['date'] = pd.to_datetime(predicted_dict['area']['date'])
    predicted_dict['station']['date'] = pd.to_datetime(predicted_dict['station']['date'])

    actual_dict['global'] = actual_dict['global'].loc[actual_dict['global']['date'].isin(filter_dates['date'])].sort_values(by = 'date', ascending=True).reset_index(drop=True)
    actual_dict['area'] = actual_dict['area'].loc[actual_dict['area']['date'].isin(filter_dates['date'])].sort_values(by = 'date', ascending=True).reset_index(drop=True)
    actual_dict['station'] = actual_dict['station'].loc[actual_dict['station']['date'].isin(filter_dates['date'])].sort_values(by = 'date', ascending=True).reset_index(drop=True)

    predicted_dict['global'] = predicted_dict['global'].loc[predicted_dict['global']['date'].isin(filter_dates['date'])].sort_values(by = 'date', ascending=True).reset_index(drop=True)
    predicted_dict['area'] = predicted_dict['area'].loc[predicted_dict['area']['date'].isin(filter_dates['date'])].sort_values(by = 'date', ascending=True).reset_index(drop=True)
    predicted_dict['station'] = predicted_dict['station'].loc[predicted_dict['station']['date'].isin(filter_dates['date'])].sort_values(by = 'date', ascending=True).reset_index(drop=True)


    for target in targets:
        scores.append( (sae(actual_dict['global'][target],predicted_dict['global'][target]) +
                        sae(actual_dict['area'][target],predicted_dict['area'][target]) +
                        sae(actual_dict['station'][target],predicted_dict['station'][target]))/N 
                      )
    return(sum(scores))
