import pandas as pd
from src.config import config
from src.ratings import create_utility_matrix
from src.factorization_recommender import FactorizationRecommender

# params
k = 7
type_of_value = "r"  # "r" or "m"


def main():
    train = pd.read_csv(config.train_data_path)
    test = pd.read_csv(config.test_data_path)
    utility_matrix, rater_index_dict, rated_index_dict = create_utility_matrix(
        train, type_of_value
    )

    fr = FactorizationRecommender(k)
    fr.fit(utility_matrix, rater_index_dict, rated_index_dict)
    rmse, n = fr.evaluate(test, type_of_value=type_of_value)
    print(f"RMSE is: {rmse} (n={n})")


if __name__ == "__main__":
    main()
