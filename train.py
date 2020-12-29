import argparse

from src.config import config
from src.d2v_recommender import D2V_Recommender

import pandas as pd

from src.processing import load_d2v_formated_data


# TODO: keep only the d2v recommender here
def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--resume_training",
        help="Whether to resume to a previously trained w2v model",
        default="y",
        choices=["y", "n"],
    )
    parser.add_argument(
        "--steps",
        help="Steps (0, 1 or 2 e.g. '13') to compute, with 0=Word2Vec, 1=raters embedding, 2=training data preparation",
        default="123",
        choices=["1", "2", "3", "12", "13", "23", "123"],
    )
    parser = parser.parse_args()
    return parser


def main():
    args = get_args()
    recommender = D2V_Recommender(**config.d2v_params)
    train = pd.read_csv(config.train_data_path)
    ## STEP 1
    if "1" in args.steps:
        print("learn embeddings for rated users")
        d2v_train = load_d2v_formated_data(config.d2v_train_data_path)
        resume_training = args.resume_training == "y"
        recommender.fit_rated_embeddings(
            d2v_train,
            config.w2v_model_path,
            config.rated_embeddings_path,
            resume_training=resume_training,
        )
        del d2v_train
    else:
        recommender.load_rated_vec(config.rated_embeddings_path)
    ## STEP 2
    if "2" in args.steps:
        print(
            "Learn embeddings for raters as the mean of embeddings of those they matched with"
        )
        recommender.fit_rater_embeddings(train, save_path=config.rater_embeddings_path)
    else:
        recommender.load_rater_vec(config.rater_embeddings_path)
    ## STEP 3
    if "3" in args.steps:
        print("Prepare training datasets")
        test = pd.read_csv(config.test_data_path)
        recommender.prepare_X_y_dataset(
            train, test, data_dict_path=config.data_dict_path
        )


if __name__ == "__main__":
    main()
