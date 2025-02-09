from invoke import task
from app.time_utils import get_timestamp_str


@task
def preprocess(c, test=False):
    """Preprocess raw data into train/test splits"""
    run_id = get_timestamp_str()
    print("Running raw data preprocess step...")
    cmd = f"python preprocess.py --input-path=./data/train_dataset_full.csv --output-path=./data/ --verbose --run-id={run_id}"
    if test:
        cmd += " --test"
    c.run(cmd)


@task 
def train(c):
    """Train model on preprocessed data using RandomForest with default hyperparameters"""
    run_id = get_timestamp_str()
    print("Running training step...")
    c.run(
        f"python train.py --model-type=RandomForest "
        f"--n-estimators=40 "
        f"--criterion=gini "
        f"--max-depth=10 "
        f"--min-samples-split=57 "
        f"--class-weight=balanced_subsample "
        f"--run-id={run_id} "
        f"--output-path=./data/ "
        f"--input-path=./data/"
    )


@task
def predict(c):
    """Generate predictions on test data"""
    run_id = get_timestamp_str()
    print("Running prediction step...")
    c.run(f"python predict.py --input-path data/test_dataset.csv --output-path data/ --run-id={run_id}")


@task
def analyze(c):
    """Analyze model results"""
    print("Running analysis step...")


@task
def echo(c, name):
    """Simple echo task for testing"""
    print(f"Hello, {name}!")


@task
def pipeline(c, n_trials=100, gpu=False):
    """Run full training pipeline:
    1. Preprocess training data
    2. Train model with RandomForest defaults
    3. Preprocess holdout data
    4. Generate predictions
    5. Analyze results
    """
    run_id = get_timestamp_str()
    c.run(f"python preprocess.py --mode=train --run-id={run_id} --output-path=./data/ --input-path=./data/ --verbose")
    train_cmd = \
        f"python train.py --optuna-search " \
        f"--input-path=./data/ " \
        f"--run-id={run_id} " \
        f"--n-trials={n_trials} " \
        f"--output-path=./data/ " \
        f"--scoring-method=auprc"
    
    if gpu:
        train_cmd += "--gpu"
    c.run(train_cmd)
    c.run(f"python preprocess.py --mode=test --run-id={run_id} --input-path=./data/preprocess_{run_id}/ --output-path=./data/ --verbose")
    c.run(f"python predict.py --run-id={run_id} --output-path=./data/ --input-path=./data/ --verbose")
    c.run(f"python result.py --run-id={run_id} --output-path=./data/ --input-path=./data/ --error-analysis")


@task
def debug_pipeline(c):
    """Run pipeline on external test data:
    1. Preprocess test data
    2. Generate predictions
    3. Preprocess additional test data
    4. Analyze results
    """
    run_id = get_timestamp_str()
    c.run(f"python preprocess.py --mode=train --run-id={run_id} --input-path=./data/train_dataset_full.csv --output-path=./data/ --verbose")
    c.run(
        "python train.py "
        "model-type=XGBoost "
        "--xgb-params='["
            "100," # n_estimators
            "0.1," # learning_rate
            "6," # max_depth
            "0.6," # subsample
            "0.6," # gamma
            "1," # reg_lambda
            "scale_pos_weight=True" # is_balanced
        "]'"
        "--input-path ./data/"
        f"--run-id {run_id} "
        "--output-path ./models/ "
    )
    c.run(f"python preprocess.py --mode=test --run-id={run_id} --input-path=./data/--output-path=./data/ --verbose")
    c.run(f"python predict.py --run-id={run_id} --output-path=./data/ --input-path=./data/ --verbose")
    c.run(f"python result.py --run-id={run_id} --output-path=./data/ --input-path=./data/")