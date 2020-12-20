import multiprocessing
from os import makedirs

from gensim.models import Word2Vec
from gensim.models import KeyedVectors

from tqdm import tqdm

import numpy as np

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

        self.model = Word2Vec(
            size=self.embedding_size,
            window=self.window,
            min_count=self.min_count,
            workers=self.workers,
        )

    def fit_embeddings(self, df):
        """

        :param df: a pd.Series of list of strings of rated_ids that were co-swiped
        :return:
        """
        model = self.model

        if model.train_count == 0:
            model.build_vocab(df.values)

        for epoch_ in tqdm(range(self.num_epochs)):
            model.train(df.values, total_examples=model.corpus_count, epochs=1)
            df.apply(list_shuffler)  # inplace checked.

        self.model = model
        self.wv = model.wv

    def prepare_for_prediction(self, df):
        """

        :param df: a pd.Series of list of strings of rated_ids that were co-swiped
        :return:
        """

        # save the average embedding of matched people for all raters

        self.mean_embeddings = None

    def predict(self, u, v):
        # get embedding of u

        # get embedding of v

        # get distance from one another

        # create a score inversely proportional to the distance
        pass

    def evaluate(self, test_data):
        pass

    def save_wv(self, wordvectors_path):
        wordvectors_path.parent.mkdir(parents=True, exist_ok=True)
        self.wv.save(str(wordvectors_path))

    def load_wv(self, wordvectors_path):
        self.wv = KeyedVectors.load(str(wordvectors_path), mmap="r")
