from invoke import task

@task
def preprocess(ctx):
    print("Running preprocess step...")
    ctx.run('python preprocess.py --input-path data/train_dataset_full.csv --out-path data/')
    # Add invoke logic for preprocessing

@task
def train(ctx):
    print("Running training step...")
    ctx.run('python train.py --optuna-search')
    # Add invoke logic for training

@task
def predict(ctx):
    print("Running prediction step...")
    ctx.run('python predict.py --input-path data/test_dataset.csv --out-path data/')
    # Add invoke logic for prediction

@task
def analyze(ctx):
    print("Running analysis step...")
    ctx.run('python analyze.py --input-path data/predictions.csv')

