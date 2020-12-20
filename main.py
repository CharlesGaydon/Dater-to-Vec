import argparse

from src.average_recommender import AverageRecommender
from src.config import config
from src.factorization_recommender import FactorizationRecommender

import pandas as pd


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reco",
        help="Whether to load saved weight or to start learning from scratch",
        default="average",
        choices=["factor", "average", "d2v"],
    )
    parser.add_argument(
        "--value_col_name",
        help="Whether to load saved weight or to start learning from scratch",
        default="r",
        choices=["r", "m"],
    )
    parser = parser.parse_args()
    return parser


def main():
    args = get_args()
    train = pd.read_csv(config.train_data_path)
    test = pd.read_csv(config.test_data_path)
    if args.reco == "factor":
        # add if for the type to evaluate here
        k = 7
        recommender = FactorizationRecommender(k)
    elif args.reco == "average":
        recommender = AverageRecommender()

    value_col_name = args.value_col_name
    recommender.fit(train, value_col_name)
    rmse, n = recommender.evaluate(test, type_of_value=value_col_name)
    print(f"RMSE is: {rmse} (n={n})")


if __name__ == "__main__":
    main()
