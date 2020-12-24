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
    df = ratings[ratings["rater"]<=max_u_id]
    
    # set train/test split
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=0)
    sss = sss.split(df, df["rater"])
    train_idx, test_idx = list(sss)[0]
    df_train = df.iloc[train_idx].set_index("rater")
    df_test = df.iloc[test_idx].set_index("rater")

    # get the quantile for the raters
    quantiles = df.groupby(df["rater"])["r"].apply(lambda x: np.quantile(x, q=config.match_threshold)).to_frame().reset_index()  # df with rater as index and quantile as value
    quantiles.columns = ["rater","r_quantile"]
    
    # apply to get matches
    df_train = pd.merge(df_train, quantiles, left_on=df_train.index, right_on=quantiles.rater).drop(columns=["key_0"])
    df_train["m"] = 1.0 * (df_train["r"]>=df_train["r_quantile"])

    df_test = pd.merge(df_test, quantiles, left_on=df_test.index, right_on=quantiles.rater).drop(columns=["key_0"])
    df_test["m"] = 1.0 * (df_test["r"]>=df_test["r_quantile"])
    
    df_train = df_train[["rater","rated","r","m"]]
    df_test = df_test[["rater","rated","r","m"]]

    # save
    train_data_path.parent.mkdir(parents=True, exist_ok=True)
    test_data_path.parent.mkdir(parents=True, exist_ok=True)

    df_train.to_csv(train_data_path, index=False)
    df_test.to_csv(test_data_path, index=False)
    return df_train, df_test


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
