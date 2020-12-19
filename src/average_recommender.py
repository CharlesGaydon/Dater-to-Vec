import numpy as np


class FactorizationRecommender:
    def __init__(self, k):
        self.mean_rater_rating = None
        self.mean_rated_rating = None

    def fit(self, train_data):
        pass

    def predict(self, rater_id, rated_id):
        score = None
        return score

    def evaluate(self, test_data, type_of_value="r"):

        if type_of_value == "r":
            rmse = 0
            n = 0
            for idx, row in test_data.iterrows():

                try:
                    pred = self.predict(row["rater"], row["rated"])
                    obs = row[type_of_value]
                    rmse += abs(pred - obs)
                    n += 1
                except:
                    pass
                    # Other user never seen before
            return rmse / n, n

        if type_of_value == "m":
            pass
