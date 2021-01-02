import ast
import os
import sys
import argparse
from tqdm import tqdm
import requests
import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedShuffleSplit


np.random.seed(0)
tqdm.pandas()
sys.path.insert(0, "./src")
from config import config

MAX_ID_IN_DATASET = config.MAX_ID_IN_DATASET


def download_data(data_url, raw_data_path, force_download=False):
    if not os.path.isfile(raw_data_path) or force_download:
        print("downloading - this might take up to 2 minutes.")
        r = requests.get(data_url, allow_redirects=True)
        with open(raw_data_path, "wb") as f:
            f.write(r.content)


def train_test_split(
    raw_data_path,
    train_data_path,
    test_data_path,
    max_id=MAX_ID_IN_DATASET,
    turn_into_matches=True,
):
    """
    Separate 20% of ratings from 100% of users.
    :param nrows: use None for all
    :return:
    """
    assert max_id <= MAX_ID_IN_DATASET
    print("Train-Test splitting")
    ratings = pd.read_csv(raw_data_path, names=["rater", "rated", "r"])
    date = ratings[ratings["rater"] <= max_id]

    # set train/test split
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=0)
    splitter = splitter.split(date, date["rater"])
    train_idx, test_idx = list(splitter)[0]
    train = date.iloc[train_idx].set_index("rater")
    test = date.iloc[test_idx].set_index("rater")

    # get the quantile for the raters
    quantiles = (
        date.groupby(date["rater"])["r"]
        .progress_apply(lambda x: np.quantile(x, q=config.match_threshold))
        .to_frame()
        .reset_index()
    )  # df with rater as index and quantile as value
    quantiles.columns = ["rater", "r_quantile"]

    # apply to get matches
    train = pd.merge(
        train, quantiles, left_on=train.index, right_on=quantiles.rater
    ).drop(columns=["key_0"])
    train["m"] = 1.0 * (train["r"] >= train["r_quantile"])

    test = pd.merge(test, quantiles, left_on=test.index, right_on=quantiles.rater).drop(
        columns=["key_0"]
    )
    test["m"] = 1.0 * (test["r"] >= test["r_quantile"])

    train = train[["rater", "rated", "m"]]
    test = test[["rater", "rated", "m"]]

    # save
    train_data_path.parent.mkdir(parents=True, exist_ok=True)
    test_data_path.parent.mkdir(parents=True, exist_ok=True)

    train.to_csv(train_data_path, index=False)
    test.to_csv(test_data_path, index=False)
    return train, test


def matches_to_matches_triplet(data_path, output_path):
    data = pd.read_csv(data_path, dtype={"rated": str})  # rater, rated, m
    A_col, B_col, m_col = data.columns
    # keep only matches
    print(f"N : {data.shape[0]}")
    data = data[data[m_col] > 0]
    print(f"N : {data.shape[0]} (matches only)")
    # group rated id into a set
    data = data.groupby(A_col)[[B_col]].agg(list)
    data.to_csv(output_path, index=False)


def load_d2v_formated_data(data_path):
    df = pd.read_csv(data_path)
    df = df["rated"].map(ast.literal_eval)
    return df


def list_shuffler(x):
    np.random.shuffle(x)
    return x


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max_id",
        help="Whether to load saved weight or to start learning from scratch",
        default=MAX_ID_IN_DATASET,
    )
    parser = parser.parse_args()
    return parser


def main():
    args = get_args()

    # download the data if not downloaded already
    download_data(config.raw_data_url, config.raw_data_path)

    # Keep last x% of ratings for each rater as test set.
    print(f"Processing N={args.max_id} records.")
    train_test_split(
        config.raw_data_path,
        config.train_data_path,
        config.test_data_path,
        max_id=int(args.max_id),
    )

    matches_to_matches_triplet(config.train_data_path, config.d2v_train_data_path)
    matches_to_matches_triplet(config.test_data_path, config.d2v_test_data_path)


if __name__ == "__main__":
    main()
