import constants as cons
from invoke import task
from app.time_utils import get_timestamp_str
from experiments import Experiment

DEFAULT_CSV_FOR_TRAINING = f"{Experiment.DATA_PATH}/{Experiment.DEFAULT_INPUT_CSV_FOR_TRAINING}"
DEFAULT_CSV_FOR_PREDICTION = f"{Experiment.DATA_PATH}/{Experiment.DEFAULT_INPUT_CSV_FOR_PREDICTION}"

@task
def pipeline(
    c,
    csv_for_training=DEFAULT_CSV_FOR_TRAINING, 
    n_trials=50, 
    gpu=False, 
    run_id=None):

    """Run full training pipeline:
    1. Preprocess training data
    2. Train model with RandomForest defaults
    3. Preprocess holdout data
    4. Generate predictions
    5. Analyze results
    """

    experiment = Experiment.new(csv_for_training, verbose=True)

    c.run(
        "python preprocess.py "
        "--mode=train "
        f"--csv-for-preprocessing={experiment.input_csv_for_training} "
        f"--output-path={experiment.preprocess_path} "
        "--verbose",
        hide=False,
        pty=True
    )
    
    # c.run(
    #     f"python train.py --optuna-search "
    #     f"--input-path={experiment.preprocess_path} "
    #     f"--n-trials={n_trials} "
    #     f"--output-path={experiment.train_path} "
    #     f"{'--gpu' if gpu else ''}"
    #     )
    
    c.run(
        "python train.py "
        "--use-default-model "
        f"--input-path={experiment.preprocess_path} "
        f"--output-path={experiment.train_path} ",
        hide=False,
        pty=True
    )
    
    c.run(
        f"python predict.py "
        f"--model-path={experiment.model_path} "
        f"--test-features-path={experiment.test_features_path} "
        f"--test-dtypes-path={experiment.test_dtypes_path} "
        f"--output-path={experiment.predict_path} "
        "--verbose",
        hide=False,
        pty=True
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
        pty=True
    )
    
    experiment.finish()


@task
def debug_pipeline(c):
    """Run full training pipeline:
    1. Preprocess training data
    2. Train model with RandomForest defaults
    3. Preprocess holdout data
    4. Generate predictions
    5. Analyze results
    """
    
    run_id = get_timestamp_str()
    c.run(
        "python preprocess.py "
        "--mode=train "
        f"--output-path=./data/preprocess_{run_id} "
        f"--input-path=./data/{cons.DEFAULT_INTERNAL_DATA_FILE} "
        "--verbose"
        )

    c.run(
        "python train.py "
        "--use-default-model "
        f"--input-path=./data/preprocess_{run_id}/ "
        f"--output-path=./data/train_{run_id}/ "
    )

    c.run(
        f"python predict.py "
        f"--model-path=./data/train_{run_id}/{cons.DEFAULT_MODEL_FILE} "
        f"--transformer-path=./data/preprocess_{run_id}/{cons.DEFAULT_TRANSFORMER_FILE} "
        f"--input-path=./data/preprocess_{run_id}/ "
        f"--output-path=./data/predictions_{run_id}/ "
        "--verbose"
        )
    
    c.run(f"python result.py --output-path=./data/ "
          f"--predictions-path=./data/predictions_{run_id}/{cons.DEFAULT_PREDICTIONS_FILE} "
          f"--labels-path=./data/preprocess_{run_id}/{cons.DEFAULT_TEST_LABELS_FILE} "
          f"--features-path=./data/preprocess_{run_id}/{cons.DEFAULT_TEST_FEATURES_FILE} "
          f"--predicted-probabilities-path=./data/predictions_{run_id}/{cons.DEFAULT_PREDICTED_PROBABILITIES_FILE} "
          f"--output-path=./data/result_{run_id}/ "
          f"--model-path=./data/train_{run_id}/{cons.DEFAULT_MODEL_FILE}"
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
        pty=True
    )

    c.run(
        f"python predict.py "
        f"--model-path={experiment.model_path} "
        f"--test-features-path={experiment.external_test_features_path} "
        f"--test-dtypes-path={experiment.external_test_dtypes_path} "
        f"--output-path={experiment.predict_path} "
        "--verbose",
        hide=False,
        pty=True
    )

    experiment.finish()