import pandas as pd

class Mean:
    def __init__(self, train_station, train_area, train_global):
        self.train_station = train_station
        self.train_area = train_area
        self.train_global = train_global
        self.station_model = None
        self.area_model = None
        self.global_model = None

    def train(self):
        self.station_model = self.train_station.drop(["date", "trend", "Latitude", "Longitude"], axis=1).groupby(
            ['Station', 'tod', 'dow']).mean().round().reset_index()

        self.area_model = self.train_area.drop(["date", "trend", "Latitude", "Longitude"], axis=1).groupby(
            ['area', 'tod', 'dow']).mean().round().reset_index()

        self.global_model = self.train_global.drop(["date", "trend"], axis=1).groupby(
            ['tod', 'dow']).mean().round().reset_index()

    def predict(self, test_station, test_area, test_global):
        prediction_station = pd.merge(test_station, self.station_model, on=[
            'Station', 'tod', 'dow'])
        prediction_area = pd.merge(test_area, self.area_model, on=[
            'area', 'tod', 'dow'])
        prediction_global = pd.merge(
            test_global, self.global_model, on=['tod', 'dow'])

        #print(prediction_global[prediction_global['trend']<=17504].head().to_string())


        return prediction_station, prediction_area, prediction_global
