import src.conf as conf


def download_data(path):
    raw_data_path = ""
    return raw_data_path


def ratings_to_matches_normalization(train_data_path, test_data_path):
    pass


def train_test_split(raw_data_path, train_data_path, test_data_path):
    """
    Separate 20% of ratings from 100% of users.
    :return:
    """
    pass


def to_sparse_array(data):
    pass


def matches_to_matches_triplet(data):
    pass


def main():
    download_data(conf["raw_data_url"], conf["raw_data_path"])
    train_test_split(
        conf["raw_data_path"], conf["train_data_path"], conf["test_data_path"]
    )
    ratings_to_matches_normalization(
        conf["train_data_path"],
        conf["test_data_path"],
        conf["norm_train_data_path"],
        conf["norm_test_data_path"],
    )


if __name__ == "__main__":
    main()
