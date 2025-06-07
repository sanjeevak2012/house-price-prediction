import mlflow

experiment_name = "Ridge"
entry_point = "Training" # this is described in MLproject file

mlflow.set_tracking_uri("http://127.0.0.1:5000")

mlflow.projects.run(
    uri=".", # Points to the current folder, where an MLproject file lives.
    entry_point=entry_point, # Matches the Training block in MLproject, which is defined as:
    experiment_name=experiment_name, # Organizes all runs under the “ElasticNet” experiment in MLflow’s UI.
    env_manager="conda" # Tells MLflow to recreate the environment using conda.yaml before running train.py.
)

# * When you run `python run.py`, MLflow:

# 1. Reads `MLproject` in the same directory.
# 2. Finds the `Training` entry point (which is `python train.py`).
# 3. Spins up a Conda environment defined by `conda.yaml`.
# 4. Executes `train.py` under the “ElasticNet” experiment, logging everything to `http://127.0.0.1:5000`