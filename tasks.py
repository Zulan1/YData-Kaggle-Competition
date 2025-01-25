from invoke import task
@task
def preprocess(c):
    print("Running preprocess step...")
    c.run("python preprocess.py --input-path=./data/train_dataset_full.csv --out-path=./data/ --verbose")
          
@task
def train(c):
    print("Running training step...")
    c.run("python train.py --optuna-search")
    # Add invoke logic for training
@task
def predict(c):
    print("Running prediction step...")
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
    c.run("python preprocess.py")
    # c.run("python train.py")
    
 