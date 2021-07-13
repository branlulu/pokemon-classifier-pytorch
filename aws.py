"""
This module was used to fetch model weights from AWS S3 Bucket.
However, it is now no longer necessary, as the model is now hosted in the repo.
To retrieve the file, make sure to set the corresponding environment variables
with your AWS credentials and AWS S3 bucket and file information.
"""

import boto3
import os

bucket_name = 'placeholder-bucket-name'
bucket_path = 'placeholder-file-path'

# Function to get file from S3 given bucket name and file path
def get_file_from_S3(bucket_name, bucket_path):
    client = boto3.client(
        's3',
        aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'], # configured in OS
        aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY'], # configured in OS
    )
    s3 = boto3.resource('s3')
    obj = s3.Object(bucket_name, bucket_path)
    body = obj.get()['Body'].read()
    return body