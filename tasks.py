''''
from invoke import task
@task
def preprocess(c):
    print("Running raw data preprocess step...")
    c.run("python preprocess.py --input-path=./data/train_dataset_full.csv --out-path=./data/ --verbose")
          
@task
def train(c):
    print("Running training step...")
    c.run("python train.py --optuna-search")
    # Add invoke logic for training
@task
def predict(c):
    print("Running prediction step...")
    c.run('python predict.py --input-path data/test_dataset.csv --out-path data/')
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
'''
 
from invoke import task
import os

# Utility function to check if a file exists before running a task
def check_file_exists(filepath):
    if not os.path.exists(filepath):
        print(f"⚠️ WARNING: {filepath} not found. Did you run the required previous steps?")
        return False
    return True

@task
def sysinfo(c):
    """Displays system information and environment details."""
    print("🔍 Gathering system information...\n")
    c.run("echo '📂 Current Directory:' && pwd")
    c.run("echo '🐍 Python Version:' && python3 --version")
    c.run("echo '💾 Disk Usage:' && df -h | grep '/$'")
    c.run("echo '🖥️  CPU Info:' && lscpu | grep 'Model name'")
    c.run("echo '📦 Installed Python Packages:' && pip list | head -10")
    print("✅ System info check complete.")

@task
def preprocess(c):
    """Preprocess raw data before training."""
    print("🚀 Running raw data preprocessing...")
    c.run("python preprocess.py --input-path=./data/train_dataset_full.csv --out-path=./data/ --verbose")
    print("✅ Preprocessing completed.")

@task(pre=[preprocess])
def train(c):
    """Train the model after preprocessing."""
    print("🚀 Running training step...")
    required_files = ["./data/train.csv", "./data/val.csv"]
    if all(check_file_exists(f) for f in required_files):
        c.run("python train.py --optuna-search")
        print("✅ Training completed.")
    else:
        print("❌ Training skipped due to missing preprocessed data.")

@task(pre=[train])
def predict(c):
    """Run prediction after training."""
    print("🚀 Running prediction step...")
    required_file = "./models/latest_model.pth"
    if check_file_exists(required_file):
        c.run("python predict.py --model-path=models/latest_model.pth --input-data=data/test_dataset.csv")
        print("✅ Prediction completed.")
    else:
        print("❌ Prediction skipped due to missing trained model.")

@task(pre=[predict])
def analyze(c):
    """Analyze prediction results."""
    print("🚀 Running analysis step...")
    required_file = "./data/predictions.csv"
    if check_file_exists(required_file):
        c.run("python analyze.py --results-path=data/predictions.csv")
        print("✅ Analysis completed.")
    else:
        print("❌ Analysis skipped due to missing predictions.")

@task(pre=[preprocess, train, predict, analyze])
def pipeline(c):
    """Executes the full pipeline: preprocess → train → predict → analyze."""
    print("🎯 Full pipeline execution completed.")
