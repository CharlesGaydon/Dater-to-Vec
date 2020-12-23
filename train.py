import argparse

from src.average_recommender import AverageRecommender
from src.config import config
from src.d2v_recommender import D2V_Recommender
from src.factorization_recommender import FactorizationRecommender

import pandas as pd

from src.processing import load_d2v_formated_data


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reco",
        help="Whether to load saved weight or to start learning from scratch",
        default="d2v",
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
    if args.reco == "factor":
        # Warning: this does not scale to the full dataset
        train = pd.read_csv(config.train_data_path)
        test = pd.read_csv(config.test_data_path)
        # add if for the type to evaluate here
        k = 7
        recommender = FactorizationRecommender(k)
        recommender.fit(train, args.value_col_name)
        rmse, n = recommender.evaluate(test, type_of_value=args.value_col_name)
        print(f"RMSE is: {rmse} (n={n})")

    elif args.reco == "average":
        train = pd.read_csv(config.train_data_path)
        test = pd.read_csv(config.test_data_path)
        recommender = AverageRecommender()
        recommender.fit(train, args.value_col_name)
        rmse, n = recommender.evaluate(test, type_of_value=args.value_col_name)
        print(f"RMSE is: {rmse} (n={n})")

    elif args.reco == "d2v":
        recommender = D2V_Recommender(**config.d2v_params)

        train = pd.read_csv(config.train_data_path)
        d2v_train = load_d2v_formated_data(config.d2v_train_data_path)

        # # learn embeddings for rated users
        recommender.fit_rated_embeddings(
            d2v_train, save_path=config.rated_embeddings_path
        )
        recommender.load_rated_vec(config.rated_embeddings_path)

        # learn embeddings for raters as the mean of embeddings of those they matched with
        recommender.fit_rater_embeddings(train, save_path=config.rater_embeddings_path)
        recommender.load_rater_vec(config.rater_embeddings_path)
        print(recommender.mean_embeddings.loc["1"])


if __name__ == "__main__":
    main()