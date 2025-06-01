import mlflow.sagemaker # Imports the MLflow SageMaker integration module
from mlflow.deployments import get_deploy_client # Imports the function to get a deployment client

# Define the name for your SageMaker endpoint
endpoint_name="prod-endpoint"

# Specify the MLflow Model URI (Uniform Resource Identifier) for the model to be deployed.
# This points to where your trained model artifacts are stored.
# - "s3://mlflow-project-artifacts/" is the S3 bucket where MLflow stores artifacts.
# - "4/" is likely a run ID within MLflow (or part of a larger path).
# - "d2ad59e0241c4f6f9212ff7e22ca780a/" is the specific MLflow Run ID.
# - "artifacts/" is the standard subdirectory where MLflow stores artifacts for a run.
# - "XGBRegressor" is the name given to the saved XGBoost model within that run's artifacts.
model_uri="s3://mlflow-project-artifacts/4/d2ad59e0241c4f6f9212ff7e22ca780a/artifacts/XGBRegressor"

# Define your configuration parameters for the SageMaker deployment as a Python dictionary.
config = {
    # ARN (Amazon Resource Name) of the IAM role that SageMaker will assume
    # to access resources like your S3 bucket (where the model is) and ECR (for the container image).
    "execution_role_arn": "arn:aws:iam::816680701120:role/house-price-role",

    # The S3 bucket name where MLflow artifacts (including your model) are stored.
    "bucket_name": "mlflow-project-artifacts",

    # The Docker image URL from Amazon Elastic Container Registry (ECR)
    # that SageMaker will use to deploy your model.
    # This image was typically built using `mlflow sagemaker build-and-push-container`.
    # It contains the MLflow serving logic and dependencies.
    "image_url": "816680701120.dkr.ecr.us-east-1.amazonaws.com/xgb:2.9.1",

    # The AWS region where you want to deploy your SageMaker endpoint.
    "region_name": "us-east-1",

    # If True, SageMaker archives the existing endpoint if one with the same name exists.
    # Here, it's False, meaning it will try to create a new one or fail if it exists.
    "archive": False,

    # The type of EC2 instance to use for the SageMaker endpoint.
    # 'ml.m5.xlarge' is a general-purpose instance type.
    "instance_type": "ml.m5.xlarge",

    # The number of instances to deploy for the endpoint.
    # '1' means a single instance for serving.
    "instance_count": 1,

    # If True, the function waits for the deployment to complete before returning.
    # If False, it returns immediately and the deployment happens asynchronously.
    "synchronous": True
}

# Initialize a deployment client specifically for SageMaker.
# This client provides methods to interact with SageMaker for model deployments.
client = get_deploy_client("sagemaker")

# Create the SageMaker deployment (endpoint).
# This command tells SageMaker to set up an endpoint using the specified model,
# the provided container image, and the configuration parameters.
client.create_deployment(
    name=endpoint_name, # The name for the SageMaker endpoint
    model_uri=model_uri, # The location of your MLflow model artifacts
    flavor="python_function", # Specifies the MLflow model flavor (how it was saved and should be loaded)
                              # 'python_function' is a generic flavor that MLflow can serve.
    config=config, # The dictionary containing all the deployment-specific settings
)