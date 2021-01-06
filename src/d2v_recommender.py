import multiprocessing
import pickle
import logging

logger = logging.getLogger(__name__)

from gensim.models import Word2Vec, KeyedVectors
from gensim.models.callbacks import CallbackAny2Vec


logging.getLogger("gensim").setLevel(
    logging.WARNING
)  # avoid seeing all logs from gensim.


from tqdm import tqdm

import numpy as np
import pandas as pd

from src.processing import list_shuffler
from autosklearn.classification import AutoSklearnClassifier
from autosklearn.metrics import roc_auc


np.random.seed(0)
tqdm.pandas()


class LossLogger(CallbackAny2Vec):
    """Output loss at each epoch"""

    def __init__(self):
        self.epoch = 1
        self.losses = []

    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        self.losses.append(loss)
        if self.epoch == 1:
            logger.info(f"Epoch: {self.epoch}    Loss: {loss}")
        else:
            logger.info(
                f"Epoch: {self.epoch}    Loss: {loss - self.loss_previous_step}"
            )
        self.epoch += 1
        self.loss_previous_step = loss


class ModelSaver(CallbackAny2Vec):
    """Output loss at each epoch"""

    def __init__(
        self, d2v_object, rated_embeddings_path, w2v_model_path, log_frequency=5
    ):
        self.epoch = 1
        self.log_frequency = log_frequency
        self.d2v_object = d2v_object
        self.rated_embeddings_path = rated_embeddings_path
        self.w2v_model_path = w2v_model_path

    def on_epoch_begin(self, model):
        pass

    def on_epoch_end(self, model):
        if self.epoch % self.log_frequency == 0:
            logger.info("Saving model weights and embeddings.")
            self.d2v_object.w2v_model = model
            self.d2v_object.wv = model.wv
            self.d2v_object.save_rated_vec(self.rated_embeddings_path)
            self.d2v_object.save_w2v_model(self.w2v_model_path)
        self.epoch += 1


class D2V_Recommender:
    def __init__(
        self,
        embedding_size=100,
        window=3,
        min_count=1,
        workers=multiprocessing.cpu_count() - 1,
        num_epochs=50,
        sample=0,  # do not downsample
    ):
        self.embedding_size = embedding_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.num_epochs = num_epochs
        self.sample = sample

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
        # Prepare the data iterator
        d2v_train_iterator = self.build_data_iterator(d2v_train)

        # Initiate the model
        loss_logger = LossLogger()
        model_saver = ModelSaver(self, rated_embeddings_path, w2v_model_path)
        self.w2v_model = Word2Vec(
            size=self.embedding_size,
            window=self.window,
            min_count=self.min_count,
            workers=self.workers,
            sample=self.sample,
            callbacks=[loss_logger, model_saver],
        )
        model = self.w2v_model
        if resume_training:
            logger.info("Resuming previous training.")
            model = self.load_w2v_model(w2v_model_path)
            model.build_vocab(d2v_train_iterator, update=True)
        elif model.train_count == 0:
            model.build_vocab(d2v_train_iterator)

        # train and save final model
        model.train(
            d2v_train_iterator,
            total_examples=model.corpus_count,
            epochs=self.num_epochs,
            compute_loss=True,
        )

        self.w2v_model = model
        self.wv = model.wv
        self.save_rated_vec(rated_embeddings_path)
        self.save_w2v_model(w2v_model_path)

    def fit_rater_embeddings(self, input_train, save_path=False):
        """
        :param df_: a pd.Series of list of strings of rated_ids that were co-swiped
        :return:
        """
        A_col, B_col, m_col = input_train.columns
        train_ = input_train.copy()
        train_ = train_[train_[m_col] > 0]  # select only those who matched

        # save the average embedding of matched people for all raters
        # TODO: optimize in one operation
        train_[B_col] = train_[B_col].apply(self.get_single_rated_vec)
        train_ = train_.dropna(
            subset=[B_col]
        )  # avoid considering the rated people appearing only once.
        train_ = train_.groupby(A_col)[B_col].apply(np.mean)
        train_.index = train_.index.astype(str)
        train_ = train_.to_frame()
        self.mean_embeddings = train_
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
            if vec1 is not None and vec2 is not None:
                return (vec1 - vec2)[0]
            else:
                return None

        train_[A_col] = train_[[A_col, B_col]].progress_apply(
            lambda x: get_vec_diff(x), axis=1
        )
        train_ = train_.dropna()
        X_train = train_[A_col].values
        X_train = np.stack(X_train)
        y_train = train_[m_col].values
        del train_
        test_[A_col] = (
            test_[[A_col, B_col]]
            .progress_apply(lambda x: get_vec_diff(x), axis=1)
            .values
        )
        test_ = test_.dropna()
        X_test = test_[A_col].values
        X_test = np.stack(X_test)
        y_test = test_[m_col].values
        del test_

        logger.info(
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

    def predict(self, u, v):
        # get embedding of u
        vec_A = self.get_single_rater_vec(u)
        vec_B = self.get_single_rated_vec(v)
        X_vec = vec_A.values - vec_B
        pred = self.classifier.predict(X_vec)
        return pred

    def predict_proba(self, u, v):
        # get embedding of u
        vec_A = self.get_single_rater_vec(u)
        vec_B = self.get_single_rated_vec(v)
        X_vec = vec_A.values - vec_B
        proba = self.classifier.predict_proba(X_vec)
        return proba

    def build_data_iterator(self, data):
        class shuffle_generator:
            def __init__(self, data):
                self.data = data

            def __iter__(self):
                self.data.apply(np.random.shuffle)
                return shuffle_generator_iter(self.data)

        class shuffle_generator_iter:
            def __init__(self, data):
                self.i = 0
                self.data = data
                self.data_length = len(data)

            def __iter__(self):
                # Iterators are iterables too.
                # Adding this functions to make them so.
                return self

            def __next__(self):
                if self.i < self.data_length:
                    i = self.i
                    self.i += 1
                    return self.data[i]  # a list
                else:
                    raise StopIteration()

        return shuffle_generator(data)

    def set_classifier(self, classifier):
        self.classifier = classifier

    def get_single_rated_vec(self, rated_id):

        try:
            return self.wv[str(rated_id)]
        except KeyError:
            # The rate user did not appear in the training dataset
            return None

    def get_single_rater_vec(self, rated_id):
        # Should always exist
        try:
            return self.mean_embeddings.loc[str(rated_id)].values
        except KeyError:
            # The rater user did not have rated people with the sufficient number of occurence (default: >3)
            return None

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
        self.w2v_model = Word2Vec.load(str(w2v_model_path), mmap="r")
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
