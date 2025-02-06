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
def pipeline(c):
    """Run full training pipeline:
    1. Preprocess training data
    2. Train model with RandomForest defaults
    3. Preprocess holdout data
    4. Generate predictions
    5. Analyze results
    """
    run_id = get_timestamp_str()

    # save the lastest run id to a file, for easy retrieval for evaluation and error analysis purposes:
    with open("latest_run_id.txt", "w") as f:
        f.write('./data/last_run_id.txt')
    

    c.run(f"python preprocess.py --mode=train --run-id={run_id} --output-path=./data/ --input-path=./data/ --verbose")
    c.run(
        f"python train.py --model-type=RandomForest "
        f"--n-estimators=500 "
        f"--criterion=gini "
        f"--max-depth=10 "
        f"--min-samples-split=5 "
        f"--min-samples-leaf=2 "
        f"--class-weight=balanced_subsample "
        f"--run-id={run_id} "
        f"--output-path=./data/ "
        f"--input-path=./data/preprocess_{run_id}"
    )
    c.run(f"python preprocess.py --mode=test --run-id={run_id} --input-path=./data/preprocess_{run_id}/ --output-path=./data/ --verbose")
    c.run(f"python predict.py --run-id={run_id} --output-path=./data/ --input-path=./data/ --verbose")
    c.run(f"python result.py --run-id={run_id} --output-path=./data/ --input-path=./data/ --error-analysis")


@task
def external_pipeline(c):
    """Run pipeline on external test data:
    1. Preprocess test data
    2. Generate predictions
    3. Preprocess additional test data
    4. Analyze results
    """
    run_id = get_timestamp_str()
    c.run(f"python preprocess.py --mode=train --run-id={run_id} --input-path=./data/train_dataset_full.csv --output-path=./data/ --verbose")
    c.run(
        f"python train.py --model-type=LogisticRegression "
        f"--C=0.001 "
        f"--run-id={run_id} "
        f"--output-path=./data/ "
        f"--input-path=./data/preprocess_{run_id}"
    )
    c.run(f"python preprocess.py --mode=test --run-id={run_id} --input-path=./data/--output-path=./data/ --verbose")
    c.run(f"python predict.py --run-id={run_id} --output-path=./data/ --input-path=./data/ --verbose")
    c.run(f"python result.py --run-id={run_id} --output-path=./data/ --input-path=./data/")