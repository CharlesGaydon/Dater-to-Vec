import multiprocessing
from pathlib import Path
import inspect

DEV_MODE = True


class Config:
    def __init__(self, abs_root):

        self.abs_root = abs_root

        self.MAX_ID_IN_DATASET = 135359

        # inputs
        self.raw_data_url = "http://www.occamslab.com/petricek/data/ratings.dat"
        self.data_folder = abs_root / "data/"
        self.raw_data_path = self.data_folder / "ratings.dat"

        # training
        self.test_ratio = 0.20  # fraction of data to be used as test set.
        self.match_threshold = 0.85  # 1-match_threshold best rated others are selected, plus others with equal score.

        self.d2v_params = {
            "embedding_size": 100,
            "window": 3,
            "min_count": 1,
            "workers": multiprocessing.cpu_count() - 1,
            "num_epochs": 100,
        }

        if DEV_MODE:

            self.d2v_params = {
                "embedding_size": 50,
                "window": 3,
                "min_count": 1,
                "workers": multiprocessing.cpu_count() - 2,
                "num_epochs": 2,
            }

            self.data_folder = self.data_folder / "dev/"

        self.train_data_path = self.data_folder / "matches_train.dat"
        self.test_data_path = self.data_folder / "matches_test.dat"

        self.d2v_train_data_path = self.data_folder / "matches_train_for_d2v.dat"
        self.d2v_test_data_path = self.data_folder / "matches_test_for_d2v.dat"

        self.rated_embeddings_path = self.data_folder / "models/rated.vectors"
        self.rater_embeddings_path = self.data_folder / "models/rater.vectors.npy"


project_absolute_root = (
    Path(inspect.getfile(inspect.currentframe())).absolute().parent.parent
)
config = Config(project_absolute_root)
