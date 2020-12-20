import numpy as np
from tqdm import tqdm


class AverageRecommender:
    def __init__(self):
        self.mean_rater_rating = None
        self.mean_rated_rating = None

        self.rater_index_dict = None
        self.rated_index_dict = None

    def fit(self, train_data, value_col_name):

        # initialize the means arrays
        self.mean_rater_rating = (
            train_data.groupby("rater")[value_col_name].mean().reset_index()
        )  # rater_id, mean_value
        self.mean_rated_rating = (
            train_data.groupby("rated")[value_col_name].mean().reset_index()
        )  # rated_id, mean_value

        # get the dictionnary id -> index
        self.rater_index_dict = {
            self.mean_rater_rating["rater"].loc[i]: i
            for i in range(len(self.mean_rater_rating))
        }
        self.rated_index_dict = {
            self.mean_rated_rating["rated"].loc[i]: i
            for i in range(len(self.mean_rated_rating))
        }

    def predict(self, rater_id, rated_id):

        rater_idx = self.rater_index_dict[rater_id]
        if rated_id not in self.rated_index_dict:
            # Other user never seen before
            return None

        rated_idx = self.rated_index_dict[rated_id]
        score = np.sqrt(
            self.mean_rater_rating.iloc[rater_idx, 1]
            * self.mean_rated_rating.iloc[rated_idx, 1]
        )

        return score

    def evaluate(self, test_data, type_of_value="r"):

        if type_of_value == "r":
            rmse = 0
            n = 0
            for idx, row in tqdm(test_data.iterrows()):

                pred = self.predict(row["rater"], row["rated"])
                if pred is not None:
                    obs = row[type_of_value]
                    rmse += abs(pred - obs)
                    n += 1

            return np.round(rmse / n, 3), n

        if type_of_value == "m":
            pass
