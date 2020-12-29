import multiprocessing
import pickle

from gensim.models import Word2Vec
from gensim.models import KeyedVectors

from tqdm import tqdm

import numpy as np
import pandas as pd

from src.processing import list_shuffler
from autosklearn.classification import AutoSklearnClassifier

np.random.seed(0)
tqdm.pandas()


class D2V_Recommender:
    def __init__(
        self,
        embedding_size=100,
        window=3,
        min_count=1,
        workers=multiprocessing.cpu_count() - 1,
        num_epochs=50,
    ):
        self.embedding_size = embedding_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.num_epochs = num_epochs

        self.w2v_model = Word2Vec(
            size=self.embedding_size,
            window=self.window,
            min_count=self.min_count,
            workers=self.workers,
        )

        self.wv = None  # access to embeddings with self.wv["rated_user_id"]
        self.mean_embeddings = (
            None  # access to embeddings with self.mean_embeddings["rater_user_id"]
        )
        self.data_dict = None  # dict of arrays with X_train, X_test, y_train, y_test

    def fit_rated_embeddings(
        self, d2v_train, w2v_model_path, rated_embeddings_path, resume_training=False
    ):
        """

        :param d2v_train: a pd.Series of list of strings of rated_ids that were co-swiped
        :return:
        """

        if resume_training:
            self.load_w2v_model(w2v_model_path)
            self.load_rated_vec(rated_embeddings_path)

        model = self.w2v_model

        if model.train_count == 0:
            model.build_vocab(d2v_train.values)

        for epoch_ in tqdm(range(self.num_epochs)):
            model.train(d2v_train.values, total_examples=model.corpus_count, epochs=1)
            d2v_train.apply(list_shuffler)  # inplace checked.

            if epoch_ % 10 == 0:
                print("Saving model weights and embeddings.")
                self.w2v_model = model
                self.wv = model.wv
                self.save_rated_vec(rated_embeddings_path)
                self.save_w2v_model(w2v_model_path)

        self.w2v_model = model
        self.wv = model.wv
        self.save_rated_vec(rated_embeddings_path)
        self.save_w2v_model(w2v_model_path)

    def fit_rater_embeddings(self, train, save_path=False):
        """

        :param df_: a pd.Series of list of strings of rated_ids that were co-swiped
        :return:
        """
        df_ = train.copy()
        df_ = df_[df_["m"] > 0]  # select only those who matched

        # save the average embedding of matched people for all raters
        # TODO: optimize in one operation
        df_["rated_emb"] = df_["rated"].apply(lambda x: self.wv[str(x)])
        df_ = df_.groupby("rater")["rated_emb"].apply(np.mean)
        df_.index = df_.index.astype(str)
        self.mean_embeddings = df_
        if save_path:
            self.save_rater_vec(save_path)

    def prepare_X_y_dataset(self, train_, test_, data_dict_path=False):
        """
        :param train
        """
        A_col, B_col, m_col = train_.columns

        def get_vec_diff(rater_rated):
            rater, rated = rater_rated
            vec1 = self.get_single_rater_vec(rater)
            vec2 = self.get_single_rated_vec(rated)
            if vec2 is not None:
                return (vec1 - vec2)[0]
            else:
                return None

        train_[A_col] = train_[[A_col, B_col]].progress_apply(
            lambda x: get_vec_diff(x), axis=1
        )
        train_ = train_.dropna()
        X_train = train_[A_col].values
        y_train = train_[m_col].values
        del train_
        test_[A_col] = (
            test_[[A_col, B_col]]
            .progress_apply(lambda x: get_vec_diff(x), axis=1)
            .values
        )
        test_ = test_.dropna()
        X_test = test_[A_col].values
        y_test = test_[m_col].values
        del test_

        print(
            f"After skipping unseen rated users: Train size = {y_train.shape} Test size = {y_test.shape}"
        )
        data_dict = {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
        }

        self.data_dict = data_dict

        if data_dict_path:
            self.save_data_dict(data_dict_path)

    def fit_classifier(self, data_dict_path, tmp_automl_path, output_automl_path):
        data_dict = self.load_data_dict(data_dict_path)
        X_train = data_dict["X_train"]
        #         X_test = data_dict["X_test"]
        y_train = data_dict["y_train"]
        #         y_test = data_dict["y_test"]

        # Auto-ML
        automl = AutoSklearnClassifier(
            time_left_for_this_task=60 * 30,
            per_run_time_limit=60,
            tmp_folder=tmp_automl_path,
            output_folder=output_automl_path,
        )

        automl.fit(X_train, y_train, dataset_name="d2v")

        print(automl.show_models())

        return automl

    def predict(self, u, v):
        # get embedding of u

        # get embedding of v

        # get distance from one another

        # create a score inversely proportional to the distance
        pass

    def evaluate(self, test_data):
        pass

    def get_single_rated_vec(self, rated_id):

        try:
            return self.wv[str(rated_id)]
        except KeyError:
            # The rate user did not appear in the training dataset
            return None

    def get_single_rater_vec(self, rated_id):
        # Should always exist
        return self.mean_embeddings.loc[str(rated_id)].values

    def save_rated_vec(self, wordvectors_path):
        wordvectors_path.parent.mkdir(parents=True, exist_ok=True)
        self.wv.save(str(wordvectors_path))

    def load_rated_vec(self, wordvectors_path):
        self.wv = KeyedVectors.load(str(wordvectors_path), mmap="r")
        return self.wv

    def save_w2v_model(self, w2v_model_path):
        w2v_model_path.parent.mkdir(parents=True, exist_ok=True)
        self.w2v_model.save(str(w2v_model_path))

    def load_w2v_model(self, w2v_model_path):
        self.w2v_model = Word2Vec.load(str(w2v_model_path))
        return self.w2v_model

    def save_rater_vec(self, rater_embeddings_path):
        # saving as a numpy array for later loading of embeddings
        rater_embeddings_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(rater_embeddings_path, arr=self.mean_embeddings.reset_index().values)

    def load_rater_vec(self, rater_embeddings_path):
        arr = np.load(rater_embeddings_path, allow_pickle=True)
        self.mean_embeddings = pd.DataFrame(index=arr[:, 0].astype(str), data=arr[:, 1])
        return self.mean_embeddings

    def save_data_dict(self, data_dict_path):
        with open(data_dict_path, "wb") as handle:
            pickle.dump(self.data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load_data_dict(self, data_dict_path):
        with open(data_dict_path, "rb") as handle:
            self.data_dict = pickle.load(handle)
        return self.data_dict
