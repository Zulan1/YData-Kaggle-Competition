import constants as cons
from invoke import task
from app.time_utils import get_timestamp_str

@task
def pipeline(c, n_trials=50, gpu=False, run_id=None):
    """Run full training pipeline:
    1. Preprocess training data
    2. Train model with RandomForest defaults
    3. Preprocess holdout data
    4. Generate predictions
    5. Analyze results
    """
    if run_id is None:
        run_id = get_timestamp_str()
    c.run(
        "python preprocess.py "
        "--mode=train "
        f"--output-path=./data/preprocess_{run_id} "
        f"--input-path=./data/{cons.DEFAULT_INTERNAL_DATA_FILE} "
        "--verbose"
        )

    c.run(
        f"python train.py --optuna-search "
        f"--input-path=./data/preprocess_{run_id}/ "
        f"--n-trials={n_trials} "
        f"--output-path=./data/train_{run_id}/ "
        f"{'--gpu' if gpu else ''}"
        )

    c.run(
        f"python predict.py "
        f"--model-path=./data/train_{run_id}/{cons.DEFAULT_MODEL_FILE} "
        f"--input-path=./data/preprocess_{run_id}/{cons.DEFAULT_TEST_FEATURES_FILE} "
        f"--output-path=./data/predictions_{run_id}/ "
        "--verbose"
        )
    c.run(f"python result.py --output-path=./data/ "
          f"--predictions-path=./data/predictions_{run_id}/{cons.DEFAULT_PREDICTIONS_FILE} "
          f"--labels-path=./data/preprocess_{run_id}/{cons.DEFAULT_TEST_LABELS_FILE} "
          f"--features-path=./data/preprocess_{run_id}/{cons.DEFAULT_TEST_FEATURES_FILE} "
          f"--output-path=./data/result_{run_id}/"
          )


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
        "--model-type=LogisticRegression "
        "--lr-params=0.01 "
        f"--input-path ./data/preprocess_{run_id}/ "
        f"--output-path ./data/train_{run_id}/ "
    )

    c.run(
        f"python predict.py "
        f"--model-path=./data/train_{run_id}/{cons.DEFAULT_MODEL_FILE} "
        f"--input-path=./data/preprocess_{run_id}/{cons.DEFAULT_TEST_FEATURES_FILE} "
        f"--output-path=./data/predictions_{run_id}/ "
        "--verbose"
        )
    c.run(f"python result.py --output-path=./data/ "
          f"--predictions-path=./data/predictions_{run_id}/{cons.DEFAULT_PREDICTIONS_FILE} "
          f"--labels-path=./data/preprocess_{run_id}/{cons.DEFAULT_TEST_LABELS_FILE} "
          f"--features-path=./data/preprocess_{run_id}/{cons.DEFAULT_TEST_FEATURES_FILE} "
          f"--output-path=./data/result_{run_id}/"
          )


@task
def inference_pipeline(c, transformer_path, model_path):
    run_id = get_timestamp_str()
    c.run(
            "python preprocess.py "
            "--mode=test "
            f"--run-id={run_id} "
            "--input-path=./data/ "
            "--output-path=./data/ "
            f"--transformer-path={transformer_path} "
            "--verbose "
        )

    c.run(
            f"python predict.py "
            f"--run-id={run_id} "
            "--input-path=./data/ "
            "--output-path=./data/ "
            f"--model-path={model_path} "
            "--verbose "
        )