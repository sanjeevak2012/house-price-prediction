from data import test
import boto3
import json

endpoint_name = "prod-endpoint"
region = 'us-east-1'

sm = boto3.client('sagemaker', region_name=region)
smrt = boto3.client('runtime.sagemaker', region_name=region)

# This is the raw data 
raw_test_20 = test[:1]

# Convert to dense array and then to list as in your original code
test_data_for_prediction = raw_test_20.toarray()[:, :-1].tolist()
test_data_json = json.dumps({'instances': test_data_for_prediction})

print("Raw data stored in test[:20] (after preprocessing and before JSON conversion):")
print(test_data_for_prediction) # This will show the actual list of lists being sent

prediction = smrt.invoke_endpoint(
    EndpointName=endpoint_name,
    Body=test_data_json,
    ContentType='application/json'
)

prediction = prediction['Body'].read().decode("ascii")

print("\nPrediction:")
print(prediction)