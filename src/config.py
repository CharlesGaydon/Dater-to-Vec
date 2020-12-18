from pathlib import Path
import inspect


class Config:
    def __init__(self, abs_root):
        self.raw_data_url = "http://www.occamslab.com/petricek/data/ratings.dat"
        self.data_folder = abs_root / "data/"
        self.raw_data_path = self.data_folder / "ratings.dat"
        self.train_data_path = self.data_folder / "matches_train.dat"
        self.test_data_path = self.data_folder / "matches_test.dat"


project_absolute_root = (
    Path(inspect.getfile(inspect.currentframe())).absolute().parent.parent
)
config = Config(project_absolute_root)
