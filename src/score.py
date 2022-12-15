import pandas as pd


def sae(y_true, y_pred):
    """Sum of Absolute errors"""
    return sum(abs(y_true - y_pred))


def overall_score(test_station, test_area, test_global, prediction_station, prediction_area, prediction_global):
    """
    Compute score on predictions

    inputs:
        test_station (pandas DataFrame): Testing DataFrame at station level
        test_area (pandas DataFrame): Testing DataFrame at area level
        test_global (pandas DataFrame): Testing DataFrame at global level
        test_station (pandas DataFrame): Prediction DataFrame at station level
        test_area (pandas DataFrame): Prediction DataFrame at area level
        test_global (pandas DataFrame): Prediction DataFrame at global level

    output:
        score (float): Score of the model

    """

    # Copy input dataframes to prevent unwanted modifications
    validation_station_score = test_station.copy()
    validation_area_score = test_area.copy()
    validation_global_score = test_global.copy()
    prediction_station_score = prediction_station.copy()
    prediction_area_score = prediction_area.copy()
    prediction_global_score = prediction_global.copy()

    # Target list
    targets = ["Available", "Charging", "Passive", "Other"]

    # Number of timesteps in the test set
    N = len(test_global)

    # Initiating scores list
    scores = []

    # Filtering dates
    validation_station_score['date'] = pd.to_datetime(
        validation_station_score['date'])
    validation_area_score['date'] = pd.to_datetime(
        validation_area_score['date'])
    validation_global_score['date'] = pd.to_datetime(
        validation_global_score['date'])
    prediction_station_score['date'] = pd.to_datetime(
        prediction_station_score['date'])
    prediction_area_score['date'] = pd.to_datetime(
        prediction_area_score['date'])
    prediction_global_score['date'] = pd.to_datetime(
        prediction_global_score['date'])

    # Sort values to align the real targets and the predicted ones
    validation_station_score = validation_station_score.sort_values(
        by=['trend', 'Station']).reset_index(drop=True)
    validation_area_score = validation_area_score.sort_values(
        by=['trend', 'area']).reset_index(drop=True)
    validation_global_score = validation_global_score.sort_values(
        by=['trend']).reset_index(drop=True)
    prediction_station_score = prediction_station_score.sort_values(
        by=['trend', 'Station']).reset_index(drop=True)
    prediction_area_score = prediction_area_score.sort_values(
        by=['trend', 'area']).reset_index(drop=True)
    prediction_global_score = prediction_global_score.sort_values(
        by=['trend']).reset_index(drop=True)

    # Iterate over each target to compute the MAE score
    for target in targets:
        scores.append((sae(validation_station_score[target], prediction_station_score[target]) +
                       sae(validation_area_score[target], prediction_area_score[target]) +
                       sae(validation_global_score[target], prediction_global_score[target])) / N
                      )

    return sum(scores)
