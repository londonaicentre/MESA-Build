import json, os
import boto3 # type: ignore
from botocore.exceptions import ClientError
from typing import Any

s3_client = boto3.client('s3')
BUCKET_NAME = os.environ['BUCKET_NAME']


def handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    """Generate a presigned URL for an S3 object."""
    object: str = event.get('pathParameters', {}).get('object')
    if not object or '/..' in object or object.startswith('/'):
        return {'statusCode': 400, 'body': json.dumps({'error': 'invalid object'})}
    try:
        presigned_url: str = s3_client.generate_presigned_url(
            'get_object', Params={'Bucket': BUCKET_NAME, 'Key': object}, ExpiresIn=3600
        )
    except ClientError as e:
        return {'statusCode': 500, 'body': json.dumps({'error': 'failed to generate URL'})}
    return {'statusCode': 200, 'body': json.dumps({'url': presigned_url})}