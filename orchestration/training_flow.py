# orchestration/training_flow.py

from prefect import flow, task
import subprocess

@task
def preprocess_data():
    print("🔧 Running feature engineering...")
    subprocess.run(["python", "src/data/features.py"], check=True)

@task
def train_models():
    print("🤖 Training models...")
    subprocess.run(["python", "src/models/train.py"], check=True)

@flow(name="turbine_retraining_flow")
def turbine_training_flow():
    print("🚀 Starting turbine model retraining workflow...")
    preprocess_data()
    train_models()
    print("✅ Retraining pipeline completed!")

if __name__ == "__main__":
    turbine_training_flow()
