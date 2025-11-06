# S3 Storage Setup Guide

## Overview
This guide explains how to set up S3 for persistent artifact storage in AWS Lambda, replacing ephemeral `/tmp` storage. **S3 is required for production** to ensure data persistence across Lambda cold starts.

## Prerequisites
- AWS Account with appropriate permissions
- AWS CLI configured (or use AWS Console)
- Lambda function already deployed via CI/CD or manually

## Step 1: Create S3 Bucket

### Option A: Using AWS CLI (Recommended)
```bash
# Set your bucket name (must be globally unique)
BUCKET_NAME="trustworthy-registry-artifacts-$(date +%s)"

# Create bucket in us-east-1 (or your preferred region)
aws s3 mb s3://${BUCKET_NAME} --region us-east-1

# Enable versioning (optional but recommended)
aws s3api put-bucket-versioning \
  --bucket ${BUCKET_NAME} \
  --versioning-configuration Status=Enabled

# Block public access (security best practice)
aws s3api put-public-access-block \
  --bucket ${BUCKET_NAME} \
  --public-access-block-configuration \
    "BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true"

echo "Bucket created: ${BUCKET_NAME}"
echo "Save this bucket name for Step 3!"
```

### Option B: Using AWS Console
1. Go to [AWS S3 Console](https://console.aws.amazon.com/s3/)
2. Click **"Create bucket"**
3. **Bucket name**: `trustworthy-registry-artifacts-XXXXX` (use a unique suffix - bucket names are globally unique)
4. **AWS Region**: `us-east-1` (or your Lambda's region)
5. **Block all public access**: ✅ **Enabled** (recommended for security)
6. Click **"Create bucket"**
7. **Save the bucket name** - you'll need it for Step 3

## Step 2: Configure Lambda IAM Role

Your Lambda function needs permission to access S3. Add these permissions to your Lambda execution role:

### Option A: Using AWS CLI
```bash
# Set your Lambda function name
FUNCTION_NAME="trustworthy-model-registry"

# Get Lambda execution role ARN
LAMBDA_ROLE=$(aws lambda get-function-configuration \
  --function-name ${FUNCTION_NAME} \
  --query 'Role' --output text)

# Extract role name from ARN (format: arn:aws:iam::ACCOUNT:role/ROLE_NAME)
ROLE_NAME=$(echo $LAMBDA_ROLE | awk -F'/' '{print $2}')

# Set your bucket name (from Step 1)
BUCKET_NAME="trustworthy-registry-artifacts-XXXXX"

# Attach S3 policy
aws iam put-role-policy \
  --role-name ${ROLE_NAME} \
  --policy-name S3AccessPolicy \
  --policy-document "{
    \"Version\": \"2012-10-17\",
    \"Statement\": [
      {
        \"Effect\": \"Allow\",
        \"Action\": [
          \"s3:PutObject\",
          \"s3:GetObject\",
          \"s3:DeleteObject\",
          \"s3:ListBucket\"
        ],
        \"Resource\": [
          \"arn:aws:s3:::${trustworthy-registry-artifacts-47906}\",
          \"arn:aws:s3:::${trustworthy-registry-artifacts-47906}/*\"
        ]
      }
    ]
  }"

echo "✅ S3 permissions added to Lambda role: ${ROLE_NAME}"
```

### Option B: Using AWS Console
1. Go to [IAM Console](https://console.aws.amazon.com/iam/) → **Roles**
2. Find your Lambda execution role (e.g., `lambda-execution-role` or similar)
3. Click **"Add permissions"** → **"Create inline policy"**
4. Switch to **JSON** tab
5. Paste the following (replace `YOUR_BUCKET_NAME` with your bucket name from Step 1):
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
        "arn:aws:s3:::trustworthy-registry-artifacts-47906",
        "arn:aws:s3:::trustworthy-registry-artifacts-47906/*"
      ]
    }
  ]
}
```
6. Click **"Review policy"**
7. **Name**: `S3AccessPolicy`
8. Click **"Create policy"**

## Step 3: Update Lambda Environment Variables

### Option A: Using AWS CLI (Recommended)
```bash
# Set your values
FUNCTION_NAME="trustworthy-model-registry"
BUCKET_NAME="trustworthy-registry-artifacts-XXXXX"  # From Step 1

# Update Lambda configuration
# Note: AWS_REGION is automatically set by Lambda and cannot be modified
aws lambda update-function-configuration \
  --function-name ${FUNCTION_NAME} \
  --environment "Variables={
    USE_S3=1,
    S3_BUCKET_NAME=${BUCKET_NAME},
    USE_SQLITE=0,
    ENVIRONMENT=production,
    LOG_LEVEL=INFO
  }"

echo "✅ Lambda environment variables updated"
```

### Option B: Using AWS Console
1. Go to [Lambda Console](https://console.aws.amazon.com/lambda/) → **Functions** → `trustworthy-model-registry`
2. Click **"Configuration"** → **"Environment variables"**
3. Click **"Edit"**
4. Add/Update these variables:
   - `USE_S3` = `1` (enables S3 storage)
   - `S3_BUCKET_NAME` = `your-bucket-name-from-step-1` (REQUIRED)
   - `USE_SQLITE` = `0` (disable SQLite in production)
   - `ENVIRONMENT` = `production`
   - `LOG_LEVEL` = `INFO`
   
   **Note:** `AWS_REGION` is automatically set by Lambda based on your function's region and cannot be modified. The code will automatically detect it.
5. Click **"Save"**

## Step 4: Update CI/CD Workflow (Optional)

If you want S3 configured automatically via GitHub Actions, update `.github/workflows/cd.yml`:

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
        S3_BUCKET_NAME=${{ secrets.S3_BUCKET_NAME }},
        USE_SQLITE=0,
        ENVIRONMENT=production,
        LOG_LEVEL=INFO
      }"
```

**Don't forget to add `S3_BUCKET_NAME` as a GitHub secret**:
1. Go to your GitHub repo → **Settings** → **Secrets and variables** → **Actions**
2. Click **"New repository secret"**
3. Name: `S3_BUCKET_NAME`
4. Value: Your bucket name from Step 1
5. Click **"Add secret"**

## Step 5: Verify Setup

### Test S3 Storage Initialization
```bash
# After deploying, check CloudWatch logs
aws logs tail /aws/lambda/trustworthy-model-registry --follow --since 5m

# Look for: "S3 storage initialized: bucket=YOUR_BUCKET_NAME"
```

### Test via API
```bash
# Get your API Gateway URL
API_URL="https://YOUR_API_ID.execute-api.us-east-1.amazonaws.com"

# Authenticate
TOKEN=$(curl -X PUT ${API_URL}/authenticate \
  -H "Content-Type: application/json" \
  -d '{
    "user": {"name": "ece30861defaultadminuser", "is_admin": true},
    "secret": {"password": "correcthorsebatterystaple123(!__+@**(A'"'"'"`;DROP TABLE packages;"}
  }' | jq -r '.')

echo "Token: ${TOKEN}"

# Create an artifact (will be stored in S3)
curl -X POST ${API_URL}/artifact/model \
  -H "Authorization: ${TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{"url": "https://huggingface.co/google/gemma-2-2b"}'

# Verify artifact exists in S3
aws s3 ls s3://YOUR_BUCKET_NAME/artifacts/ --recursive
```

### Verify S3 Objects
```bash
# List all artifacts in S3
aws s3 ls s3://YOUR_BUCKET_NAME/artifacts/ --recursive

# Example output:
# artifacts/model-1-1234567890/metadata.json
# artifacts/model-2-1234567891/metadata.json
```

## Storage Architecture

### S3 Structure
```
YOUR_BUCKET_NAME/
├── artifacts/
│   ├── model-1-1234567890/
│   │   ├── metadata.json       # Artifact metadata (required)
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
    "hf_data": [
      {
        "readme_text": "...",
        "pipeline_tag": "text-generation",
        "datasets": [...]
      }
    ]
  },
  "created_at": "2025-11-05T20:00:00",
  "created_by": "username"
}
```

## Environment Variables Summary

### Production (Lambda) - Use S3 Only
```bash
USE_S3=1                    # Enable S3 storage
S3_BUCKET_NAME=your-bucket  # REQUIRED: Your S3 bucket name
# Note: AWS_REGION is automatically set by Lambda (cannot be modified)
USE_SQLITE=0                # Disable SQLite in production
ENVIRONMENT=production       # Production environment
LOG_LEVEL=INFO              # Logging level
```

### Local Development - Use SQLite
```bash
USE_SQLITE=1                                  # Enable SQLite
SQLALCHEMY_DATABASE_URL=sqlite:///registry.db  # Local database file
USE_S3=0                                       # Disable S3 locally
ENVIRONMENT=development                        # Development environment
```

## Troubleshooting

### S3 Storage Not Initializing
- ✅ Check `S3_BUCKET_NAME` environment variable is set correctly
- ✅ Verify Lambda IAM role has S3 permissions (Step 2)
- ✅ Check CloudWatch logs for error messages:
  ```bash
  aws logs tail /aws/lambda/trustworthy-model-registry --follow
  ```
- ✅ Ensure bucket exists in the same region as Lambda

### Permission Errors
- ✅ Ensure IAM role has:
  - `s3:PutObject` (write metadata)
  - `s3:GetObject` (read metadata)
  - `s3:DeleteObject` (delete artifacts)
  - `s3:ListBucket` (list artifacts and pagination - covers ListObjectsV2 API)
- ✅ Verify bucket name matches exactly (case-sensitive)
- ✅ Check bucket region matches Lambda's region (AWS_REGION is automatically set by Lambda)

### Bucket Not Found Errors
- ✅ Ensure bucket exists in the same region as Lambda
- ✅ Verify bucket region matches Lambda's region (AWS_REGION is automatically set by Lambda)
- ✅ Verify bucket name is correct (no typos)

### Artifacts Not Persisting
- ✅ Check CloudWatch logs for S3 errors
- ✅ Verify S3 bucket has write permissions
- ✅ Test S3 access manually:
  ```bash
  aws s3 cp test.json s3://YOUR_BUCKET_NAME/test.json
  aws s3 rm s3://YOUR_BUCKET_NAME/test.json
  ```

### Cost Considerations
- **S3 Storage**: ~$0.023/GB/month
- **PUT requests**: $0.005 per 1,000 requests
- **GET requests**: $0.0004 per 1,000 requests
- **For typical usage**: Very low cost (< $1/month for small deployments)
- **Data Transfer**: First 100GB free, then $0.09/GB

## Fallback Behavior

The system gracefully handles storage failures:

1. **If S3 is enabled but unavailable**:
   - Falls back to SQLite (if enabled)
   - Falls back to in-memory storage
   - Errors are logged but don't crash the application

2. **If S3 bucket doesn't exist**:
   - System will attempt to create it (if permissions allow)
   - Falls back to SQLite/in-memory if creation fails

3. **If S3 access is denied**:
   - Falls back to SQLite (if enabled)
   - Falls back to in-memory storage
   - Error logged: "Failed to access S3 bucket"

## Benefits

✅ **Persistent Storage**: Data survives Lambda container recycling  
✅ **Scalable**: Handles millions of artifacts  
✅ **Cost-Effective**: Very low cost for typical usage  
✅ **Reliable**: 99.999999999% (11 9's) durability  
✅ **Production-Ready**: Suitable for real-world deployments  
✅ **No Data Loss**: Unlike `/tmp` storage, data persists across cold starts  

## Quick Reference

### Required Environment Variables (Production)
```bash
USE_S3=1
S3_BUCKET_NAME=your-bucket-name
# Note: AWS_REGION is automatically set by Lambda (read-only)
USE_SQLITE=0
ENVIRONMENT=production
```

### Verify Setup
```bash
# 1. Check bucket exists
aws s3 ls s3://YOUR_BUCKET_NAME

# 2. Check Lambda has permissions
aws iam get-role-policy --role-name YOUR_ROLE --policy-name S3AccessPolicy

# 3. Check Lambda environment variables
aws lambda get-function-configuration --function-name trustworthy-model-registry --query 'Environment.Variables'

# 4. Test via API (see Step 5)
```

## Next Steps

After setup:
1. ✅ Deploy your code to Lambda
2. ✅ Test creating an artifact via API
3. ✅ Verify artifact appears in S3
4. ✅ Test downloading artifacts
5. ✅ Test rating/metrics calculation
6. ✅ Monitor CloudWatch logs for any issues

---

**Need Help?** Check CloudWatch logs first:
```bash
aws logs tail /aws/lambda/trustworthy-model-registry --follow
```
