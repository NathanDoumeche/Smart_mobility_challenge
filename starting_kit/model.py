def run_model(model_name, type, training_data):
    if model_name == "mean":
        mean(training_data, type)
    else:
        raise Exception("This model doesn't exist")


# Models
def mean(data, type):
    if type == "station":
        s_naive = data.drop(["date", "trend", "Latitude", "Longitude"], axis=1).groupby(
        ['Station', 'tod', 'dow']).mean().round().reset_index()
        return pd.merge(test_station, station_prediction, on=[
            'Station', 'tod', 'dow'])

    elif type == "station":
        a_naive = data.drop(["date", "trend", "Latitude", "Longitude"], axis=1).groupby(
            ['area', 'tod', 'dow']).mean().round().reset_index()

    elif type == "station":
        g_naive = data.drop(["date", "trend"], axis=1).groupby(
            ['tod', 'dow']).mean().round().reset_index()



    area_prediction = pd.merge(test_area, area_prediction, on=[
                               'area', 'tod', 'dow'])
    global_prediction = pd.merge(
        test_global, global_prediction, on=['tod', 'dow'])