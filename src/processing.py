import ast
import os
import sys
import argparse
from tqdm import tqdm
import requests
import pandas as pd
import numpy as np

np.random.seed(0)

sys.path.insert(0, "./src")
from config import config

MAX_ID_IN_DATASET = config.MAX_ID_IN_DATASET


def download_data(data_url, raw_data_path, force_download=False):
    if not os.path.isfile(raw_data_path) or force_download:
        print("downloading - this might take up to 2 minutes.")
        r = requests.get(data_url, allow_redirects=True)
        with open(raw_data_path, "wb") as f:
            f.write(r.content)


def turn_ratings_into_matches(train, test):
    match_threshold = config.match_threshold
    match_threshold_in_ratings = train["r"].quantile(q=match_threshold)
    train = train.assign(m=1 * (train["r"].values >= match_threshold_in_ratings))
    test = test.assign(m=1 * (test["r"].values >= match_threshold_in_ratings))
    return train, test


def train_test_split(
    raw_data_path,
    train_data_path,
    test_data_path,
    max_u_id=MAX_ID_IN_DATASET,
    turn_into_matches=True,
):
    """
    Separate 20% of ratings from 100% of users.
    :param nrows: use None for all
    :return:
    """
    assert max_u_id <= MAX_ID_IN_DATASET
    print("Train-Test splitting")
    ratings = pd.read_csv(raw_data_path, names=["rater", "rated", "r"])

    # Shuffle to avoid bias of ordered rated users
    ratings = ratings.sample(frac=1).reset_index(drop=True)

    # list of all raters, already integers in our dataset
    u = range(1, min(MAX_ID_IN_DATASET, max_u_id) + 1)
    test = pd.DataFrame(columns=ratings.columns)
    train = pd.DataFrame(columns=ratings.columns)
    # TODO: coulb be optimized with a groupby ?
    for u_id in tqdm(u):
        temp = ratings[ratings["rater"] == u_id]
        n = len(temp)
        train_size = int((1 - config.test_ratio) * n)

        dummy_train = temp.iloc[:train_size]
        dummy_test = temp.iloc[train_size:]

        if turn_into_matches:
            dummy_train, dummy_test = turn_ratings_into_matches(dummy_train, dummy_test)

        # TODO: not efficint. Optimize!
        test = pd.concat([test, dummy_test], ignore_index=True)
        train = pd.concat([train, dummy_train], ignore_index=True)

    # save
    train_data_path.parent.mkdir(parents=True, exist_ok=True)
    test_data_path.parent.mkdir(parents=True, exist_ok=True)

    train.to_csv(train_data_path, index=False)
    test.to_csv(test_data_path, index=False)
    return train, test


def matches_to_matches_triplet(data_path, output_path):
    df = pd.read_csv(data_path, dtype={"rated": str})  # rater, rated, r, m
    # keep only matches
    print(f"N : {df.shape[0]}")
    df = df[df["m"] > 0]
    print(f"N : {df.shape[0]} (matches only)")
    # group rated id into a set
    df = df.groupby("rater")[["rated"]].agg(list)
    df.to_csv(output_path, index=False)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max_u_id",
        help="Whether to load saved weight or to start learning from scratch",
        default=MAX_ID_IN_DATASET,
    )
    parser = parser.parse_args()
    return parser


def load_d2v_formated_data(data_path):
    df = pd.read_csv(data_path)
    df = df["rated"].map(ast.literal_eval)
    return df


def list_shuffler(x):
    np.random.shuffle(x)
    return x


def main():
    args = get_args()

    # download the data if not downloaded already
    download_data(config.raw_data_url, config.raw_data_path)

    # Keep last x% of ratings for each rater as test set.
    train_test_split(
        config.raw_data_path,
        config.train_data_path,
        config.test_data_path,
        max_u_id=int(args.max_u_id),
    )

    matches_to_matches_triplet(config.train_data_path, config.d2v_train_data_path)
    matches_to_matches_triplet(config.test_data_path, config.d2v_test_data_path)


if __name__ == "__main__":
    main()
