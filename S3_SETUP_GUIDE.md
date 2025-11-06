# S3 Storage Setup Guide

## Overview
This guide explains how to set up S3 for persistent artifact storage in AWS Lambda, replacing ephemeral `/tmp` storage.

## Prerequisites
- AWS Account with appropriate permissions
- AWS CLI configured (or use AWS Console)
- Lambda function already deployed

## Step 1: Create S3 Bucket

### Option A: Using AWS CLI
```bash
aws s3 mb s3://trustworthy-registry-artifacts --region us-east-1
```

### Option B: Using AWS Console
1. Go to AWS S3 Console
2. Click "Create bucket"
3. Bucket name: `trustworthy-registry-artifacts` (or your preferred name)
4. AWS Region: `us-east-1` (or your preferred region)
5. Block all public access: ✅ Enabled (recommended)
6. Click "Create bucket"

## Step 2: Configure Lambda IAM Role

You need to grant your Lambda function permission to access S3:

### Option A: Using AWS CLI
```bash
# Get your Lambda execution role ARN
LAMBDA_ROLE=$(aws lambda get-function-configuration \
  --function-name trustworthy-model-registry \
  --query 'Role' --output text)

# Extract role name from ARN (format: arn:aws:iam::ACCOUNT:role/ROLE_NAME)
ROLE_NAME=$(echo $LAMBDA_ROLE | awk -F'/' '{print $2}')

# Attach S3 policy
aws iam put-role-policy \
  --role-name $ROLE_NAME \
  --policy-name S3AccessPolicy \
  --policy-document '{
    "Version": "2012-10-17",
    "Statement": [
      {
        "Effect": "Allow",
        "Action": [
          "s3:PutObject",
          "s3:GetObject",
          "s3:DeleteObject",
          "s3:ListBucket"
        ],
        "Resource": [
          "arn:aws:s3:::trustworthy-registry-artifacts",
          "arn:aws:s3:::trustworthy-registry-artifacts/*"
        ]
      }
    ]
  }'
```

### Option B: Using AWS Console
1. Go to IAM Console → Roles
2. Find your Lambda execution role (e.g., `lambda-execution-role`)
3. Click "Add permissions" → "Create inline policy"
4. Switch to JSON tab
5. Paste:
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:PutObject",
        "s3:GetObject",
        "s3:DeleteObject",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::trustworthy-registry-artifacts",
        "arn:aws:s3:::trustworthy-registry-artifacts/*"
      ]
    }
  ]
}
```
6. Name: `S3AccessPolicy`
7. Click "Create policy"

## Step 3: Update Lambda Environment Variables

### Option A: Using AWS CLI
```bash
aws lambda update-function-configuration \
  --function-name trustworthy-model-registry \
  --environment "Variables={
    USE_S3=1,
    S3_BUCKET_NAME=trustworthy-registry-artifacts,
    AWS_REGION=us-east-1,
    USE_SQLITE=0,
    ENVIRONMENT=production,
    LOG_LEVEL=INFO
  }"
```

### Option B: Using AWS Console
1. Go to Lambda Console → Functions → `trustworthy-model-registry`
2. Click "Configuration" → "Environment variables"
3. Click "Edit"
4. Add/Update:
   - `USE_S3` = `1`
   - `S3_BUCKET_NAME` = `trustworthy-registry-artifacts`
   - `AWS_REGION` = `us-east-1`
   - `USE_SQLITE` = `0` (disable SQLite when using S3)
   - `ENVIRONMENT` = `production`
   - `LOG_LEVEL` = `INFO`
5. Click "Save"

## Step 4: Update CI/CD Workflow (Optional)

If you want S3 configured automatically via CI/CD, update `.github/workflows/cd.yml`:

```yaml
- name: Update Lambda Configuration
  run: |
    aws lambda update-function-configuration \
      --function-name trustworthy-model-registry \
      --runtime python3.11 \
      --handler app.handler \
      --timeout 60 \
      --memory-size 1024 \
      --environment "Variables={
        USE_S3=1,
        S3_BUCKET_NAME=trustworthy-registry-artifacts,
        AWS_REGION=us-east-1,
        USE_SQLITE=0,
        ENVIRONMENT=production,
        LOG_LEVEL=INFO
      }"
```

## Step 5: Verify Setup

### Test S3 Storage
```bash
# Deploy your code
git push origin main

# Wait for deployment, then test via API
curl -X PUT https://YOUR_API_URL/authenticate \
  -H "Content-Type: application/json" \
  -d '{
    "user": {"name": "ece30861defaultadminuser", "is_admin": true},
    "secret": {"password": "correcthorsebatterystaple123(!__+@**(A'"'"'"`;DROP TABLE packages;"}
  }'

# Check CloudWatch logs for "S3 storage initialized"
aws logs tail /aws/lambda/trustworthy-model-registry --follow
```

### Verify S3 Objects
```bash
# List artifacts in S3
aws s3 ls s3://trustworthy-registry-artifacts/artifacts/ --recursive
```

## Storage Architecture

### S3 Structure
```
trustworthy-registry-artifacts/
├── artifacts/
│   ├── model-1-1234567890/
│   │   ├── metadata.json       # Artifact metadata
│   │   └── files/              # Optional: uploaded files
│   │       └── model.zip
│   ├── dataset-2-1234567891/
│   │   └── metadata.json
│   └── code-3-1234567892/
│       └── metadata.json
```

### Metadata Format
```json
{
  "metadata": {
    "name": "artifact-name",
    "id": "model-1-1234567890",
    "type": "model"
  },
  "data": {
    "url": "https://huggingface.co/model",
    "hf_data": [...]
  },
  "created_at": "2025-11-05T20:00:00",
  "created_by": "username"
}
```

## Migration from SQLite

If you have existing artifacts in SQLite (`/tmp/registry.db`), you can migrate them:

1. **Export from SQLite** (if still accessible):
```bash
# On Lambda /tmp (if you have access)
sqlite3 /tmp/registry.db ".dump" > artifacts_backup.sql
```

2. **Use the API** to re-ingest artifacts:
- The API will automatically store them in S3

## Troubleshooting

### S3 Storage Not Initializing
- Check `S3_BUCKET_NAME` environment variable is set correctly
- Verify Lambda IAM role has S3 permissions
- Check CloudWatch logs for error messages

### Permission Errors
- Ensure IAM role has `s3:PutObject`, `s3:GetObject`, `s3:DeleteObject`, `s3:ListBucket`
- Verify bucket name matches exactly (case-sensitive)

### Bucket Not Found
- Ensure bucket exists in the same region as Lambda
- Check `AWS_REGION` environment variable matches bucket region

### Cost Considerations
- S3 Storage: ~$0.023/GB/month
- PUT requests: $0.005 per 1,000 requests
- GET requests: $0.0004 per 1,000 requests
- For typical usage: Very low cost (< $1/month)

## Fallback Behavior

If S3 is enabled but unavailable:
- Falls back to SQLite (if enabled)
- Falls back to in-memory storage
- Errors are logged but don't crash the application

## Benefits

✅ **Persistent Storage**: Data survives Lambda container recycling
✅ **Scalable**: Handles millions of artifacts
✅ **Cost-Effective**: Very low cost for typical usage
✅ **Reliable**: 99.999999999% (11 9's) durability
✅ **Production-Ready**: Suitable for real-world deployments

