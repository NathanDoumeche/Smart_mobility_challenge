import pandas as pd


class Mean:
    def __init__(self, train_data, type):
        self.train_data = train_data
        self.model = None
        self.type = type

    def train(self):
        if self.type == "station":
            columns_to_drop = ["date", "trend", "Latitude", "Longitude"]
            columns_to_group_by = ['Station', 'tod', 'dow']
        if self.type == "area":
            columns_to_drop = ["date", "trend", "Latitude", "Longitude"]
            columns_to_group_by = ['area', 'tod', 'dow']
        if self.type == "global":
            columns_to_drop = ["date", "trend"]
            columns_to_group_by = ['tod', 'dow']

        self.model = self.train_data.drop(columns_to_drop, axis=1).groupby(
            columns_to_group_by).mean(numeric_only=True).round().reset_index()

    def predict(self, test_data):
        if self.type == "station":
            columns_to_merge_on = ['Station', 'tod', 'dow']
        if self.type == "area":
            columns_to_merge_on = ['area', 'tod', 'dow']
        if self.type == "global":
            columns_to_merge_on = ['tod', 'dow']
        prediction = pd.merge(test_data, self.model, on=columns_to_merge_on)

        return prediction
