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