from app.time_utils import get_timestamp_str
import os
import shutil
import constants as cons

class Experiment:

    DATA_PATH = 'data'
    ARCHIVED_EXPERIMENTS_PATH = 'archived_experiments'
    
    DEFAULT_INPUT_CSV_FOR_TRAINING = 'train_dataset_full.csv'
    DEFAULT_INPUT_CSV_FOR_PREDICTION = 'X_test_1st_raw.csv'

    DEFAULT_TRAIN_SET_FILE = 'train.csv'
    DEFAULT_VAL_SET_FILE = 'val.csv'
    DEFAULT_TRAIN_DTYPES_FILE = 'train_dtypes.pkl'
    DEFAULT_VAL_DTYPES_FILE = 'val_dtypes.pkl'
    DEFAULT_TEST_FEATURES_FILE = 'test_features.csv'
    DEFAULT_TEST_DTYPES_FILE = 'test_dtypes.pkl'
    DEFAULT_TEST_LABELS_FILE = 'test_labels.csv'

    DEFAULT_PREDICTIONS_FILE = 'predictions.csv'
    DEFAULT_PREDICTED_PROBABILITIES_FILE = 'predicted_probabilities.csv'

    DEFAULT_RESULTS_FILE = 'results.csv'

    DEFAULT_TRANSFORMER_FILE = 'transformer.pkl'
    DEFAULT_MODEL_FILE = 'model.pkl'

    @classmethod
    def new(cls, csv_for_training, verbose=False):
        run_id = get_timestamp_str()
        experiment = cls(run_id, verbose)
        experiment._init_new_experiment()
        experiment.set_input_csv_for_training(csv_for_training)
        return experiment
    
    @classmethod
    def existing(cls, run_id, verbose=False):

        archived_experiment_path = f"{cls.ARCHIVED_EXPERIMENTS_PATH}/experiment_{run_id}"
        if not os.path.exists(archived_experiment_path):
            raise ValueError(f"Experiment {run_id} does not exist at {archived_experiment_path}")
        
        experiment = cls(run_id, verbose)
        experiment._restore_from_archive()
        return experiment

    def __init__(self, run_id, verbose=False):

        self.verbose = verbose
        self.run_id = run_id
        if self.verbose:
            print(f"\nInitializing experiment {self.run_id}...")

        self.data_path = self.DATA_PATH
        self.archived_experiments_path = self.ARCHIVED_EXPERIMENTS_PATH

        self.experiment_name = f"experiment_{self.run_id}"
        self.experiment_path = f"{self.DATA_PATH}/{self.experiment_name}"
        self.train_path = f"{self.experiment_path}/train"
        self.preprocess_path = f"{self.experiment_path}/preprocess"
        self.predict_path = f"{self.experiment_path}/predict"
        self.result_path = f"{self.experiment_path}/result"
        self.experiment_data_path = f"{self.experiment_path}/data"
        self.input_csv_for_training = f"{self.experiment_data_path}/input_for_training.csv"
        self.input_csv_for_prediction = f"{self.experiment_data_path}/input_for_prediction.csv"
        self.model_path = f"{self.train_path}/{self.DEFAULT_MODEL_FILE}"
        self.predictions_path = f"{self.predict_path}/{self.DEFAULT_PREDICTIONS_FILE}"
        self.predictions_probabilities_path = f"{self.predict_path}/{self.DEFAULT_PREDICTED_PROBABILITIES_FILE}"
        self.labels_path = f"{self.preprocess_path}/{self.DEFAULT_TEST_LABELS_FILE}"
        self.features_path = f"{self.preprocess_path}/{self.DEFAULT_TEST_FEATURES_FILE}"
        self.transformer_path = f"{self.preprocess_path}/{self.DEFAULT_TRANSFORMER_FILE}"

    def _init_new_experiment(self):
        self.clear_data_path(self.data_path)
        self.create_experiment_folders()

    def _restore_from_archive(self):
        self.clear_data_path(self.data_path)
        shutil.copytree(f"{self.archived_experiments_path}/{self.experiment_name}", self.experiment_path)

    def set_input_csv_for_training(self, csv_for_training):
        self._save_csv(csv_for_training, self.input_csv_for_training)

    def set_input_csv_for_prediction(self, csv_full_path):
        self._save_csv(csv_full_path, self.input_csv_for_prediction)

    def clear_data_path(self, data_path):
        for item in os.listdir(data_path):
            item_path = os.path.join(data_path, item)
            if item != 'best_model' and item != self.DEFAULT_INPUT_CSV_FOR_TRAINING and item != self.DEFAULT_INPUT_CSV_FOR_PREDICTION:
                if os.path.isfile(item_path):
                    os.remove(item_path)
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)

    def create_experiment_folders(self):
        os.makedirs(self.train_path)
        os.makedirs(self.preprocess_path)
        os.makedirs(self.predict_path)
        os.makedirs(self.result_path)
        os.makedirs(self.experiment_data_path)

    def _save_csv(self, from_path, to_path):
        if from_path != to_path:
            shutil.copy(from_path, to_path)

    def finish(self):
        if os.path.exists(f"{self.archived_experiments_path}/{self.experiment_name}"):
            shutil.rmtree(f"{self.archived_experiments_path}/{self.experiment_name}")
        shutil.copytree(self.experiment_path, f"{self.archived_experiments_path}/{self.experiment_name}")

        if self.verbose:
            print(f"\nExperiment {self.run_id} finished and archived at {self.archived_experiments_path}/{self.experiment_name}")
    

