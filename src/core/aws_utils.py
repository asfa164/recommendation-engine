import boto3
import json
from botocore.exceptions import ClientError


class AwsUtils:
    def __init__(self, region_name, aws_endpoint_url):
        self.region_name = region_name
        self.aws_endpoint_url = aws_endpoint_url

    def get_secrets(self, secret_name):
        session = boto3.session.Session()
        if self.aws_endpoint_url:
            client = session.client(
                service_name="secretsmanager",
                region_name=self.region_name,
                endpoint_url=self.aws_endpoint_url,
            )
        else:
            client = session.client(
                service_name="secretsmanager",
                region_name=self.region_name,
            )

        try:
            get_secret_value_response = client.get_secret_value(SecretId=secret_name)
            return json.loads(get_secret_value_response["SecretString"])
        except ClientError as e:
            raise e
