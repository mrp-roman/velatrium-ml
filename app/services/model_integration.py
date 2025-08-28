import boto3

def invoke_model(data):
    runtime = boto3.client('sagemaker-runtime')

    # Prepare the payload for the model
    payload = {
        "features": data["network_activity"],  # Example: Adjust based on your model's input format
    }

    response = runtime.invoke_endpoint(
        EndpointName='risk-assessment-endpoint',
        ContentType='application/json',
        Body=json.dumps(payload),
    )

    # Parse and return the model's response
    result = json.loads(response['Body'].read())
    return result
