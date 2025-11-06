"""S3 storage adapter for persistent artifact storage in AWS Lambda.

This module provides S3-backed storage for artifacts, replacing ephemeral /tmp storage.
Artifacts are stored as JSON metadata files in S3, with optional file storage.
"""

from __future__ import annotations

import json
import os
from typing import Dict, Any, Optional, List
import boto3
from botocore.exceptions import ClientError, NoCredentialsError

from datetime import datetime

logger = None
try:
    import logging

    logger = logging.getLogger(__name__)
except Exception:
    pass


class S3Storage:
    """S3-backed storage for artifacts and metadata"""

    def __init__(self, bucket_name: str, region: str = "us-east-1"):
        """Initialize S3 storage

        Args:
            bucket_name: S3 bucket name for storing artifacts
            region: AWS region (default: us-east-1)
        """
        self.bucket_name = bucket_name
        self.region = region
        self.s3_client = None

        try:
            # Initialize S3 client (uses IAM role credentials in Lambda)
            self.s3_client = boto3.client("s3", region_name=region)
            if logger:
                logger.info(f"S3 storage initialized: bucket={bucket_name}, region={region}")
        except NoCredentialsError:
            if logger:
                logger.warning("AWS credentials not found - S3 storage unavailable")
            self.s3_client = None
        except Exception as e:
            if logger:
                logger.error(f"Failed to initialize S3 client: {e}")
            self.s3_client = None

    def _ensure_bucket_exists(self) -> bool:
        """Ensure S3 bucket exists, create if needed"""
        if not self.s3_client:
            return False

        try:
            self.s3_client.head_bucket(Bucket=self.bucket_name)
            return True
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "")
            if error_code == "404":
                # Bucket doesn't exist, try to create it
                try:
                    self.s3_client.create_bucket(
                        Bucket=self.bucket_name,
                        CreateBucketConfiguration={"LocationConstraint": self.region},
                    )
                    if logger:
                        logger.info(f"Created S3 bucket: {self.bucket_name}")
                    return True
                except Exception as create_err:
                    if logger:
                        logger.error(f"Failed to create S3 bucket: {create_err}")
                    return False
            else:
                if logger:
                    logger.error(f"Failed to access S3 bucket: {e}")
                return False

    def save_artifact_metadata(self, artifact_id: str, metadata: Dict[str, Any]) -> bool:
        """Save artifact metadata to S3

        Args:
            artifact_id: Unique artifact identifier
            metadata: Artifact metadata dictionary

        Returns:
            True if successful, False otherwise
        """
        if not self.s3_client:
            return False

        if not self._ensure_bucket_exists():
            return False

        try:
            key = f"artifacts/{artifact_id}/metadata.json"
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=key,
                Body=json.dumps(metadata, indent=2),
                ContentType="application/json",
            )
            if logger:
                logger.info(f"Saved artifact metadata to S3: {key}")
            return True
        except Exception as e:
            if logger:
                logger.error(f"Failed to save artifact metadata to S3: {e}")
            return False

    def get_artifact_metadata(self, artifact_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve artifact metadata from S3

        Args:
            artifact_id: Unique artifact identifier

        Returns:
            Artifact metadata dictionary, or None if not found
        """
        if not self.s3_client:
            return None

        try:
            key = f"artifacts/{artifact_id}/metadata.json"
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=key)
            metadata = json.loads(response["Body"].read().decode("utf-8"))
            if logger:
                logger.debug(f"Retrieved artifact metadata from S3: {key}")
            return metadata
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "")
            if error_code == "NoSuchKey":
                # Artifact not found
                return None
            if logger:
                logger.error(f"Failed to retrieve artifact metadata from S3: {e}")
            return None
        except Exception as e:
            if logger:
                logger.error(f"Error retrieving artifact metadata from S3: {e}")
            return None

    def delete_artifact_metadata(self, artifact_id: str) -> bool:
        """Delete artifact metadata from S3

        Args:
            artifact_id: Unique artifact identifier

        Returns:
            True if successful, False otherwise
        """
        if not self.s3_client:
            return False

        try:
            key = f"artifacts/{artifact_id}/metadata.json"
            self.s3_client.delete_object(Bucket=self.bucket_name, Key=key)
            if logger:
                logger.info(f"Deleted artifact metadata from S3: {key}")
            return True
        except Exception as e:
            if logger:
                logger.error(f"Failed to delete artifact metadata from S3: {e}")
            return False

    def list_artifacts(self, artifact_type: Optional[str] = None) -> List[str]:
        """List all artifact IDs from S3

        Args:
            artifact_type: Optional filter by artifact type

        Returns:
            List of artifact IDs
        """
        if not self.s3_client:
            return []

        artifact_ids = []
        try:
            prefix = "artifacts/"
            paginator = self.s3_client.get_paginator("list_objects_v2")
            pages = paginator.paginate(Bucket=self.bucket_name, Prefix=prefix)

            for page in pages:
                if "Contents" not in page:
                    continue

                for obj in page["Contents"]:
                    key = obj["Key"]
                    # Extract artifact_id from path: artifacts/{id}/metadata.json
                    if key.endswith("/metadata.json"):
                        artifact_id = key.split("/")[1]
                        if artifact_type:
                            # Need to fetch metadata to check type
                            metadata = self.get_artifact_metadata(artifact_id)
                            if (
                                metadata
                                and metadata.get("metadata", {}).get("type") == artifact_type
                            ):
                                artifact_ids.append(artifact_id)
                        else:
                            artifact_ids.append(artifact_id)

            if logger:
                logger.debug(f"Listed {len(artifact_ids)} artifacts from S3")
            return artifact_ids
        except Exception as e:
            if logger:
                logger.error(f"Failed to list artifacts from S3: {e}")
            return []

    def count_artifacts_by_type(self, artifact_type: str) -> int:
        """Count artifacts of a specific type

        Args:
            artifact_type: Artifact type to count

        Returns:
            Count of artifacts
        """
        artifact_ids = self.list_artifacts(artifact_type=artifact_type)
        return len(artifact_ids)

    def upload_file(
        self,
        artifact_id: str,
        file_key: str,
        file_content: bytes,
        content_type: str = "application/octet-stream",
    ) -> bool:
        """Upload a file to S3 for an artifact

        Args:
            artifact_id: Unique artifact identifier
            file_key: File path/key (e.g., 'model.zip')
            file_content: File content as bytes
            content_type: MIME type

        Returns:
            True if successful, False otherwise
        """
        if not self.s3_client:
            return False

        if not self._ensure_bucket_exists():
            return False

        try:
            key = f"artifacts/{artifact_id}/files/{file_key}"
            self.s3_client.put_object(
                Bucket=self.bucket_name, Key=key, Body=file_content, ContentType=content_type
            )
            if logger:
                logger.info(f"Uploaded file to S3: {key}")
            return True
        except Exception as e:
            if logger:
                logger.error(f"Failed to upload file to S3: {e}")
            return False

    def download_file(self, artifact_id: str, file_key: str) -> Optional[bytes]:
        """Download a file from S3

        Args:
            artifact_id: Unique artifact identifier
            file_key: File path/key (e.g., 'model.zip')

        Returns:
            File content as bytes, or None if not found
        """
        if not self.s3_client:
            return None

        try:
            key = f"artifacts/{artifact_id}/files/{file_key}"
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=key)
            content = response["Body"].read()
            if logger:
                logger.debug(f"Downloaded file from S3: {key}")
            return content
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "")
            if error_code == "NoSuchKey":
                return None
            if logger:
                logger.error(f"Failed to download file from S3: {e}")
            return None
        except Exception as e:
            if logger:
                logger.error(f"Error downloading file from S3: {e}")
            return None

    def delete_artifact_files(self, artifact_id: str) -> bool:
        """Delete all files for an artifact

        Args:
            artifact_id: Unique artifact identifier

        Returns:
            True if successful, False otherwise
        """
        if not self.s3_client:
            return False

        try:
            prefix = f"artifacts/{artifact_id}/files/"
            paginator = self.s3_client.get_paginator("list_objects_v2")
            pages = paginator.paginate(Bucket=self.bucket_name, Prefix=prefix)

            deleted_count = 0
            for page in pages:
                if "Contents" not in page:
                    continue

                objects_to_delete = [{"Key": obj["Key"]} for obj in page["Contents"]]
                if objects_to_delete:
                    self.s3_client.delete_objects(
                        Bucket=self.bucket_name, Delete={"Objects": objects_to_delete}
                    )
                    deleted_count += len(objects_to_delete)

            if logger:
                logger.info(f"Deleted {deleted_count} files for artifact {artifact_id}")
            return True
        except Exception as e:
            if logger:
                logger.error(f"Failed to delete artifact files from S3: {e}")
            return False


# Global S3 storage instance (initialized on first use)
_s3_storage: Optional[S3Storage] = None


def get_s3_storage() -> Optional[S3Storage]:
    """Get or create S3 storage instance"""
    global _s3_storage

    if _s3_storage is not None:
        return _s3_storage

    bucket_name = os.environ.get("S3_BUCKET_NAME")
    if not bucket_name:
        return None

    region = os.environ.get("AWS_REGION", "us-east-1")
    _s3_storage = S3Storage(bucket_name=bucket_name, region=region)
    return _s3_storage if _s3_storage.s3_client else None
