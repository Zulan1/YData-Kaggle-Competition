from invoke import task
from app.time_utils import get_timestamp_str




@task
def preprocess(c, test=False):
    run_id = get_timestamp_str()
    print("Running raw data preprocess step...")
    cmd = f"python preprocess.py --input-path=./data/train_dataset_full.csv --output-path=./data/ --verbose --run-id={run_id}"
    if test:
        cmd += " --test"
    c.run(cmd)

          
@task
def train(c):
    run_id = get_timestamp_str()
    print("Running training step...")
    c.run(f"python train.py --model-type=LogisticRegression --C 0.5 --run-id={run_id} --output-path=./data/")
    # Add invoke logic for training
@task
def predict(c):
    run_id = get_timestamp_str()
    print("Running prediction step...")
    c.run(f'python predict.py --input-path data/test_dataset.csv --out-path data/ --run-id={run_id}')
    # Add invoke logic for prediction
@task
def analyze(c):
    print("Running analysis step...")
    # Add invoke logic for result analysis
@task
def echo(c, name):
    print(f"Hello, {name}!")
@task
def pipeline(c):
    run_id = get_timestamp_str()
    c.run(f"python preprocess.py --run-id={run_id} --output-path=./data/ --input-path=./data/train_dataset_full.csv --verbose")
    c.run(f"python train.py --model-type=LogisticRegression --C 0.5 --run-id={run_id} --output-path=./data/")
    c.run(f"python preprocess.py --test --run-id={run_id} --input-path=./data/train_dataset_full.csv --output-path=./data/ --verbose")
    c.run(f"python predict.py --run-id={run_id} --output-path=./data/ --input-path=./data/ --verbose")
    c.run(f"python result.py --run-id={run_id} --output-path=./data/ --input-path=./data/")
    
 