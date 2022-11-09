# import numpy as np
import pandas as pd

def sae(y_true, y_pred):
    """Sum of Absolute errors"""
    return(sum(abs(y_true-y_pred)))


def overall_metric(validation_station, validation_area, validation_global, prediction_station, prediction_area, prediction_global, learning_rate, iterations, depth, classif, loss_function, eval_metric):
    """Overall metric"""

    filter_dates = validation_station[["date"]]

    ### Target list
    targets = [ "Available", "Charging", "Passive", "Other"]

    ### Number of timesteps in the test set
    N = len(validation_global)

    ### Initiating scores list
    scores_area = []
    scores_global = []
    scores_station = []
    ### Filtering dates
    validation_global['date'] = pd.to_datetime(validation_global['date'])
    validation_area['date'] = pd.to_datetime(validation_area['date'])
    validation_global['date'] = pd.to_datetime(validation_station['date'])

    prediction_global['date'] = pd.to_datetime(prediction_global['date'])
    prediction_area['date'] = pd.to_datetime(prediction_area['date'])
    prediction_station['date'] = pd.to_datetime(prediction_station['date'])

    validation_global = validation_global.sort_values(by = 'trend').reset_index(drop=True)
    validation_area = validation_area.sort_values(by = ['trend', 'area']).reset_index(drop=True)
    validation_station = validation_station.sort_values(by = ['trend', 'Station']).reset_index(drop=True)

    prediction_global = prediction_global.sort_values(by = 'trend').reset_index(drop=True)
    prediction_area = prediction_area.sort_values(by = ['trend', 'area']).reset_index(drop=True)
    prediction_station = prediction_station.sort_values(by = ['trend', 'Station']).reset_index(drop=True)
    for target in targets:
        # scores.append( (sae(validation_global[target],prediction_global[target]) +
        #                 sae(validation_area[target],prediction_area[target]) +
        #                 sae(validation_station[target],prediction_station[target]))/N
        #               )
        scores_area.append(sae(validation_area[target], prediction_area[target])/N)
        scores_global.append(sae(validation_global[target], prediction_global[target]) / N)
        scores_station.append(sae(validation_station[target], prediction_station[target]) / N)


    print(f"Parameters used: learning_rate {learning_rate}, iterations {iterations}, depth {depth}, classif {classif}, loss_function {loss_function}, eval_metric {eval_metric}\n"
        "Global metric: ", sum(scores_global),"\n","Area metric: ", sum(scores_area),"\n","Station metric: ", sum(scores_station),"\n","Total metric:", sum(scores_area)+sum(scores_station)+sum(scores_global), "-------")
    with open('results.txt', 'a') as f:
        f.write(f"Parameters used: learning_rate {learning_rate}, iterations {iterations}, depth {depth}, classif {classif}, loss_function {loss_function}, eval_metric {eval_metric} \n Global metric: {sum(scores_global)} \n Area metric: {sum(scores_area)} \n Station metric: {sum(scores_station)} \n Total metric: {sum(scores_area) + sum(scores_station) + sum(scores_global)} \n")
    return(sum(scores_area)+sum(scores_station)+sum(scores_global))
