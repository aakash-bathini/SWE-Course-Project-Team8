"""
AWS Lambda deployment configuration for Phase 2
Handles Lambda function setup, S3 integration, and CloudWatch logging
"""

import json
import boto3
import logging
from typing import Dict, Any
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)


class AWSDeployment:
    """AWS deployment manager for Lambda functions"""

    def __init__(self, region: str = "us-east-1"):
        self.region = region
        self.lambda_client = boto3.client("lambda", region_name=region)
        self.s3_client = boto3.client("s3", region_name=region)
        self.cloudwatch_client = boto3.client("logs", region_name=region)
        self.iam_client = boto3.client("iam", region_name=region)

    def create_lambda_function(
        self, function_name: str, role_arn: str, zip_file_path: str, handler: str = "app.handler"
    ) -> Dict[str, Any]:
        """Create a Lambda function"""
        try:
            with open(zip_file_path, "rb") as zip_file:
                zip_content = zip_file.read()

            response = self.lambda_client.create_function(
                FunctionName=function_name,
                Runtime="python3.11",
                Role=role_arn,
                Handler=handler,
                Code={"ZipFile": zip_content},
                Description="Trustworthy Model Registry API",
                Timeout=30,
                MemorySize=512,
                Environment={"Variables": {"ENVIRONMENT": "production", "LOG_LEVEL": "INFO"}},
            )

            logger.info(f"Lambda function created: {function_name}")
            return dict(response)

        except ClientError as e:
            logger.error(f"Failed to create Lambda function: {str(e)}")
            raise

    def create_iam_role(self, role_name: str) -> str:
        """Create IAM role for Lambda function"""
        trust_policy = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": {"Service": "lambda.amazonaws.com"},
                    "Action": "sts:AssumeRole",
                }
            ],
        }

        try:
            response = self.iam_client.create_role(
                RoleName=role_name,
                AssumeRolePolicyDocument=json.dumps(trust_policy),
                Description="Role for Trustworthy Model Registry Lambda",
            )

            # Attach basic execution policy
            self.iam_client.attach_role_policy(
                RoleName=role_name,
                PolicyArn="arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole",
            )

            # Attach S3 access policy
            s3_policy = {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Action": ["s3:GetObject", "s3:PutObject", "s3:DeleteObject"],
                        "Resource": "arn:aws:s3:::trustworthy-model-registry/*",
                    }
                ],
            }

            self.iam_client.put_role_policy(
                RoleName=role_name,
                PolicyName="S3AccessPolicy",
                PolicyDocument=json.dumps(s3_policy),
            )

            logger.info(f"IAM role created: {role_name}")
            return str(response["Role"]["Arn"])

        except ClientError as e:
            logger.error(f"Failed to create IAM role: {str(e)}")
            raise

    def create_s3_bucket(self, bucket_name: str) -> bool:
        """Create S3 bucket for model storage"""
        try:
            # For regions other than us-east-1, we need to specify LocationConstraint
            if self.region == "us-east-1":
                self.s3_client.create_bucket(Bucket=bucket_name)
            else:
                self.s3_client.create_bucket(
                    Bucket=bucket_name,
                    CreateBucketConfiguration={"LocationConstraint": self.region}
                )
            logger.info(f"S3 bucket created: {bucket_name} in region {self.region}")
            return True

        except ClientError as e:
            if e.response["Error"]["Code"] == "BucketAlreadyExists":
                logger.info(f"S3 bucket already exists: {bucket_name}")
                return True
            else:
                logger.error(f"Failed to create S3 bucket: {str(e)}")
                raise

    def setup_cloudwatch_logs(self, function_name: str) -> str:
        """Setup CloudWatch logs for Lambda function"""
        log_group_name = f"/aws/lambda/{function_name}"

        try:
            self.cloudwatch_client.create_log_group(
                logGroupName=log_group_name, retentionInDays=14  # Free tier limit
            )
            logger.info(f"CloudWatch log group created: {log_group_name}")
            return log_group_name

        except ClientError as e:
            if e.response["Error"]["Code"] == "ResourceAlreadyExistsException":
                logger.info(f"CloudWatch log group already exists: {log_group_name}")
                return log_group_name
            else:
                logger.error(f"Failed to create CloudWatch log group: {str(e)}")
                raise

    def deploy_function(self, function_name: str, zip_file_path: str) -> Dict[str, Any]:
        """Deploy updated function code"""
        try:
            with open(zip_file_path, "rb") as zip_file:
                zip_content = zip_file.read()

            response = self.lambda_client.update_function_code(
                FunctionName=function_name, ZipFile=zip_content
            )

            logger.info(f"Lambda function updated: {function_name}")
            return dict(response)

        except ClientError as e:
            logger.error(f"Failed to update Lambda function: {str(e)}")
            raise


# Lambda handler for FastAPI
def handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """AWS Lambda handler for FastAPI application"""
    try:
        from mangum import Mangum
        from app import app

        # Create ASGI adapter
        asgi_handler = Mangum(app)

        # Process the request
        response = asgi_handler(event, context)

        return response

    except Exception as e:
        logger.error(f"Lambda handler error: {str(e)}")
        return {
            "statusCode": 500,
            "body": json.dumps({"error": "Internal server error"}),
            "headers": {"Content-Type": "application/json", "Access-Control-Allow-Origin": "*"},
        }


# Deployment script
def deploy_to_aws() -> Dict[str, str]:
    """Deploy the application to AWS"""
    deployment = AWSDeployment()

    # Configuration
    function_name = "trustworthy-model-registry"
    role_name = "trustworthy-model-registry-role"
    bucket_name = "trustworthy-model-registry"

    try:
        # Create IAM role
        role_arn = deployment.create_iam_role(role_name)

        # Create S3 bucket
        deployment.create_s3_bucket(bucket_name)

        # Setup CloudWatch logs
        log_group = deployment.setup_cloudwatch_logs(function_name)

        # Create Lambda function (requires zip file)
        # deployment.create_lambda_function(function_name, role_arn, "deployment.zip")

        logger.info("AWS deployment setup completed successfully")

        return {
            "function_name": function_name,
            "role_arn": role_arn,
            "bucket_name": bucket_name,
            "log_group": log_group,
        }

    except Exception as e:
        logger.error(f"AWS deployment failed: {str(e)}")
        raise


if __name__ == "__main__":
    # Run deployment
    result = deploy_to_aws()
    print(json.dumps(result, indent=2))
