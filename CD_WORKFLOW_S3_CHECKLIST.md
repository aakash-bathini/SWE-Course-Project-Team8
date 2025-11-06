# CD Workflow S3 Configuration Checklist

## ✅ What Was Fixed

1. **Added S3_BUCKET_NAME validation** - Workflow now checks if the secret exists before deployment
2. **Added environment variable verification** - After deployment, verifies that `S3_BUCKET_NAME` and `USE_S3` are set correctly in Lambda
3. **Added debug output** - Shows what values are being set during deployment

## 🔍 How to Verify CD Workflow is Working

### Step 1: Check GitHub Secrets

Go to your GitHub repository:
1. **Settings** → **Secrets and variables** → **Actions**
2. Verify these secrets exist:
   - ✅ `AWS_ACCESS_KEY_ID`
   - ✅ `AWS_SECRET_ACCESS_KEY`
   - ✅ `S3_BUCKET_NAME` ← **THIS IS CRITICAL FOR S3**

### Step 2: Check CD Workflow Run

1. Go to **Actions** tab in GitHub
2. Find the latest **CD** workflow run
3. Check the **"Deploy Lambda via S3"** step logs
4. Look for:
   ```
   ✅ S3_BUCKET_NAME secret is configured
   Bucket name: <your-bucket-name>
   ```
5. After deployment, look for:
   ```
   🔍 Verifying Lambda environment variables...
   ✅ S3_BUCKET_NAME is set: <your-bucket-name>
   ✅ USE_S3 is set to: 1
   ```

### Step 3: Verify in AWS Lambda Console

1. Go to AWS Console → Lambda → Functions → `trustworthy-model-registry`
2. Go to **Configuration** → **Environment variables**
3. Verify these are set:
   - `USE_S3` = `1`
   - `S3_BUCKET_NAME` = `<your-bucket-name>`
   - `USE_SQLITE` = `0`
   - `ENVIRONMENT` = `production`
   - `LOG_LEVEL` = `INFO`

### Step 4: Check CloudWatch Logs

After triggering a Lambda request, check CloudWatch logs for:
```
DEBUG: [S3 Status] USE_S3=True, S3_BUCKET_NAME=<your-bucket>, ENVIRONMENT=production
DEBUG: [S3 Status] ✅ S3 storage initialized and ready
```

## ❌ Common Issues

### Issue 1: `S3_BUCKET_NAME` not set in GitHub Secrets
**Symptom:** Workflow fails with error: `❌ ERROR: S3_BUCKET_NAME secret is not set`
**Fix:** Add `S3_BUCKET_NAME` secret in GitHub repository settings

### Issue 2: Environment variables not set in Lambda
**Symptom:** Workflow succeeds but verification shows `⚠️ WARNING: S3_BUCKET_NAME environment variable is not set`
**Fix:** 
- Check if the secret value is empty
- Check Lambda IAM permissions for `lambda:UpdateFunctionConfiguration`

### Issue 3: S3 storage initialization fails
**Symptom:** CloudWatch shows `⚠️ USE_S3=True but s3_storage is None`
**Fix:**
- Verify S3 bucket exists and is accessible
- Verify Lambda IAM role has S3 permissions:
  - `s3:GetObject`
  - `s3:PutObject`
  - `s3:DeleteObject`
  - `s3:ListBucket`

## 🚀 Next Steps

1. **Merge `testing-local` to `main`** to deploy these changes
2. **Trigger a CD workflow run** (push to `main` or use workflow_dispatch)
3. **Check the workflow logs** for the new validation messages
4. **Verify CloudWatch logs** show S3 status on each request

## 📝 Notes

- The workflow deploys from `main` branch only
- `S3_BUCKET_NAME` secret must match your actual S3 bucket name
- The bucket should already exist (workflow doesn't create it, only the Lambda deployment bucket)
- Lambda IAM role needs permissions to access the S3 bucket

