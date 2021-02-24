import numpy as np
from tqdm import tqdm


class AverageRecommender:
    def __init__(self):
        self.mean_rater_rating = None
        self.mean_rated_rating = None

        self.rater_index_dict = None
        self.rated_index_dict = None

    def fit(self, train_data):
        """
        :param train_data: pd.DataFrame with columns id_A, id_B, rating.
        """
        A_col, B_col, R_col = train_data.columns
        # initialize the means arrays
        self.mean_rater_rating = (
            train_data.groupby(A_col)[R_col].mean().reset_index()
        )  # rater_id, mean_value
        self.mean_rated_rating = (
            train_data.groupby(B_col)[R_col].mean().reset_index()
        )  # rated_id, mean_value

        # get the dictionnary id -> index
        self.rater_index_dict = {
            self.mean_rater_rating[A_col].loc[i]: i
            for i in range(len(self.mean_rater_rating))
        }
        self.rated_index_dict = {
            self.mean_rated_rating[B_col].loc[i]: i
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

    def evaluate(self, test_data):
        """
        :param test_data: pd.DataFrame with columns id_A, id_B, rating.
        """
        A_col, B_col, R_col = test_data.columns
        rmse = 0
        n = 0
        for idx, row in tqdm(test_data.iterrows()):

            pred = self.predict(row[A_col], row[B_col])
            if pred is not None:
                obs = row[R_col]
                rmse = rmse + abs(pred - obs)
                n = n + 1

        return np.round(rmse / n, 3), n
