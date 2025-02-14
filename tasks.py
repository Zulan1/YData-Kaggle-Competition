import constants as cons
from invoke import task
from app.time_utils import get_timestamp_str
from experiments import Experiment
import os

DEFAULT_CSV_FOR_TRAINING = f"{Experiment.DATA_PATH}/{Experiment.DEFAULT_INPUT_CSV_FOR_TRAINING}"
DEFAULT_CSV_FOR_PREDICTION = f"{Experiment.DATA_PATH}/{Experiment.DEFAULT_INPUT_CSV_FOR_PREDICTION}"

# Determine if the current OS is Windows, if not, do not use pty since it is not supported in Windows 
if os.name == 'nt':
        pty_arg = False
else:
    pty_arg = True

    

@task
def pipeline(
    c,
    csv_for_training=DEFAULT_CSV_FOR_TRAINING,
    limit_data=False,
    model="default",
    n_trials=100, 
    gpu=False, 
    metric=None,
    run_id=None):

    """Run full training pipeline:
    1. Preprocess training data
    2. Train model with RandomForest defaults
    3. Preprocess holdout data
    4. Generate predictions
    5. Analyze results
    """
    experiment = Experiment.new(csv_for_training, verbose=True)

    limit_data_cmd = "--limit-data" if limit_data else ""


    c.run(
        "python preprocess.py "
        "--mode=train "
        f"--csv-for-preprocessing={experiment.input_csv_for_training} "
        f"--output-path={experiment.preprocess_path} "
        f"{limit_data_cmd} "
        "--verbose",
        hide=False,
        pty=pty_arg
    )

    # c.run(
    #     f"python train.py --optuna-search "
    #     f"--input-path={experiment.preprocess_path} "
    #     f"--n-trials={n_trials} "
    #     f"--output-path={experiment.train_path} "
    #     f"{'--gpu' if gpu else ''}"
    #     )
    
    c.run(
        f"python train.py --use-default-model "
        f"--input-path={experiment.preprocess_path} "
        f"--output-path={experiment.train_path} "
        f"--metric={metric} "
        "--verbose",
        hide=False,
        pty=pty_arg
    )
    c.run(
        f"python predict.py "
        f"--model-path={experiment.model_path} "
        f"--test-features-path={experiment.test_features_path} "
        f"--test-dtypes-path={experiment.test_dtypes_path} "
        f"--output-path={experiment.predict_path} "
        "--verbose",
        hide=False,
        pty=pty_arg
    )

    c.run(
        f"python result.py "
        f"--predictions-path={experiment.predictions_path} "
        f"--predicted-probabilities-path={experiment.predictions_probabilities_path} "
        f"--labels-path={experiment.labels_path} "
        f"--features-path={experiment.features_path} "
        f"--model-path={experiment.model_path} "
        f"--output-path={experiment.result_path}",
        hide=False,
        pty=pty_arg
    )
    
    experiment.finish()

def new_func(c, n_trials, gpu, experiment):
    c.run(
         f"python train.py --optuna-search "
         f"--input-path={experiment.preprocess_path} "
         f"--n-trials={n_trials} "
         f"--output-path={experiment.train_path} "
         f"{'--gpu' if gpu else ''}"
         )

@task
def inference_pipeline(c, run_id, csv_for_prediction=DEFAULT_CSV_FOR_PREDICTION):

    print(f"\nStarting inference pipeline using experiment {run_id}...\n")

    try:
        experiment = Experiment.existing(run_id, verbose=True)
    except ValueError as e:
        print(f"Error: {e}")
        print("Please provide a valid experiment run_id that exists in the archived_experiments directory")
        return

    experiment.set_input_csv_for_prediction(csv_for_prediction)

    c.run(
        "python preprocess.py "
        "--mode=inference "
        f"--csv-for-preprocessing={experiment.input_csv_for_prediction} "
        f"--transformer-path={experiment.transformer_path} "
        f"--output-path={experiment.preprocess_path} "
        "--verbose ",
        hide=False,
        pty=pty_arg
    )

    c.run(
        f"python predict.py "
        f"--model-path={experiment.full_model_path} "
        f"--test-features-path={experiment.external_test_features_path} "
        f"--test-dtypes-path={experiment.external_test_dtypes_path} "
        f"--output-path={experiment.predict_path} "
        "--verbose",
        hide=False,
        pty=pty_arg
    )

    experiment.finish()