import multiprocessing
from os import makedirs

from gensim.models import Word2Vec
from gensim.models import KeyedVectors

from tqdm import tqdm

import numpy as np
import pandas as pd

from src.processing import list_shuffler

np.random.seed(0)


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

    def fit_rated_embeddings(self, d2v_train, save_path=False):
        """

        :param d2v_train: a pd.Series of list of strings of rated_ids that were co-swiped
        :return:
        """
        model = self.w2v_model

        if model.train_count == 0:
            model.build_vocab(d2v_train.values)

        for epoch_ in tqdm(range(self.num_epochs)):
            model.train(d2v_train.values, total_examples=model.corpus_count, epochs=1)
            d2v_train.apply(list_shuffler)  # inplace checked.

        self.w2v_model = model
        self.wv = model.wv

        if save_path:
            self.save_rated_vec(save_path)

    def fit_rater_embeddings(self, train, save_path=False):
        """

        :param df_: a pd.Series of list of strings of rated_ids that were co-swiped
        :return:
        """
        df_ = train.copy()
        df_ = df_[df_["m"] > 0]  # select only those who matched

        # save the average embedding of matched people for all raters
        df_["rated_emb"] = df_["rated"].apply(lambda x: self.wv[str(x)])
        df_ = df_.groupby("rater")["rated_emb"].apply(np.mean)
        self.mean_embeddings = df_
        if save_path:
            self.save_rater_vec(save_path)

    def predict(self, u, v):
        # get embedding of u

        # get embedding of v

        # get distance from one another

        # create a score inversely proportional to the distance
        pass

    def evaluate(self, test_data):
        pass

    def save_rated_vec(self, wordvectors_path):
        wordvectors_path.parent.mkdir(parents=True, exist_ok=True)
        self.wv.save(str(wordvectors_path))

    def load_rated_vec(self, wordvectors_path):
        self.wv = KeyedVectors.load(str(wordvectors_path), mmap="r")

    def save_rater_vec(self, rater_embeddings_path):
        # saving as a numpy array for later loading of embeddings
        rater_embeddings_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(rater_embeddings_path, arr=self.mean_embeddings.reset_index().values)

    def load_rater_vec(self, rater_embeddings_path):
        arr = np.load(rater_embeddings_path, allow_pickle=True)
        self.mean_embeddings = pd.DataFrame(index=arr[:, 0].astype(str), data=arr[:, 1])
