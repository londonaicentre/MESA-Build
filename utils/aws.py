import boto3, os
from botocore.exceptions import ClientError

def upload_file(region_name, file_name, bucket, object_name=None, path=None):
    if object_name is None:
        object_name = os.path.basename(file_name)
    try:
        boto3.client("s3", region_name=region_name).upload_file(
            file_name, 
            bucket, 
            path + "/" + object_name if path else object_name
        )
    except ClientError as e:
        print(e)
        return False
    return True

def start_batch_inference(region_name, job_id, model_id, role_arn, bucket, batch_file):
    try:
        boto3.client("bedrock", region_name=region_name).create_model_invocation_job(
            jobName="genollama-" + job_id.replace("/", "-"),
            modelId=model_id,
            roleArn=role_arn,
            inputDataConfig={
                "s3InputDataConfig": {
                    "s3Uri": "s3://" + bucket + "/" + job_id + "/input/" + batch_file
                }
            },
            outputDataConfig={
                "s3OutputDataConfig": {
                    "s3Uri": "s3://" + bucket + "/" + job_id + "/output/" 
                }
            },
        )
    except ClientError as e:
        print(e)
        return False
    return True