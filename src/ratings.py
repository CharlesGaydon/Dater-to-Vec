import os
import pandas as pd
import numpy as np
import requests
from tqdm import tqdm
import sys

sys.path.insert(0, "./src")
from config import config

MAX_ID_IN_DATASET = 135359


def download_data(data_url, raw_data_path, force_download=False):
    if not os.path.isfile(raw_data_path) or force_download:
        print("downloading - this might take up to 2 minutes.")
        r = requests.get(data_url, allow_redirects=True)
        with open(raw_data_path, "wb") as f:
            f.write(r.content)


def turn_ratings_into_matches(train, test, match_threshold=0.65):
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

    # list of all raters, already integers in our dataset
    u = range(1, min(MAX_ID_IN_DATASET, max_u_id) + 1)
    test = pd.DataFrame(columns=ratings.columns)
    train = pd.DataFrame(columns=ratings.columns)
    test_ratio = 0.2  # fraction of data to be used as test set.
    for u_id in tqdm(u):
        temp = ratings[ratings["rater"] == u_id]
        n = len(temp)
        train_size = int((1 - test_ratio) * n)

        dummy_train = temp.iloc[:train_size]
        dummy_test = temp.iloc[train_size:]

        if turn_into_matches:
            dummy_train, dummy_test = turn_ratings_into_matches(dummy_train, dummy_test)

        test = pd.concat([test, dummy_test], ignore_index=True)
        train = pd.concat([train, dummy_train], ignore_index=True)

    train.to_csv(train_data_path, index=False)
    test.to_csv(test_data_path, index=False)
    return train, test


# assume users are ordered and we start at 0 ?
def create_utility_matrix(data, value_col_name):

    rater_list = data["rater"].tolist()
    rated_list = data["rated"].tolist()
    value_list = data[value_col_name].tolist()
    rater_ids = list(set(data["rater"]))
    rated_ids = list(set(data["rated"]))

    rater_index_dict = {rater_ids[i]: i for i in range(len(rater_ids))}
    pd_dict = {v_id: [np.nan for i in range(len(rater_ids))] for v_id in rated_ids}
    for i in range(0, len(data)):
        rater_id = rater_list[i]
        rated_id = rated_list[i]
        value = value_list[i]
        pd_dict[rated_id][rater_index_dict[rater_id]] = value
    utility_matrix = pd.DataFrame(pd_dict)
    utility_matrix.index = rater_ids

    rated_cols = list(utility_matrix.columns)
    rated_index_dict = {rated_cols[i]: i for i in range(len(rated_cols))}
    # rater_index gives us a mapping of rater_id to index of rater
    # rated_index provides the same for rated
    return utility_matrix, rater_index_dict, rated_index_dict


def matches_to_matches_triplet(data):
    pass


def main():
    download_data(config.raw_data_url, config.raw_data_path)
    train, test = train_test_split(
        config.raw_data_path,
        config.train_data_path,
        config.test_data_path,
        max_u_id=10000,
    )


if __name__ == "__main__":
    main()
