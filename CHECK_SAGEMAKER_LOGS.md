# How to Check SageMaker Logs in CloudWatch

## Log Groups Location

### ✅ Lambda Function Logs (PRIMARY - Check This First)
**Log Group:** `/aws/lambda/trustworthy-model-registry`

**What to look for:**
- `"SageMaker LLM service initialized: region=us-east-1, endpoint=trustworthy-registry-llm"`
- `"SageMaker service available, attempting invocation"`
- `"Performance metric attempt 1 with AWS SageMaker"`
- `"Invoking SageMaker endpoint: trustworthy-registry-llm"`
- `"SageMaker endpoint invocation successful"`

**If you DON'T see these:** SageMaker is not being used (check Lambda environment variables)

### ✅ SageMaker Endpoint Logs (SECONDARY - Confirms Invocations)
**Log Group:** `/aws/sagemaker/Endpoints/trustworthy-registry-llm`
**Log Stream:** `variant1/i-058da17b324581fdf` (instance-specific)

**What to look for:**
- POST requests to `/invocations` (actual inference calls)
- Response times and model outputs
- Errors if invocations fail

**Note:** You'll see `GET /ping` every 5 seconds (health checks) - this is normal, but you need POST requests for actual usage.

## Current Status from Your Logs

From what you showed:
- ✅ SageMaker endpoint is running (health checks working)
- ✅ Model loaded (gpt2)
- ❌ **NO inference requests** (only health checks)

This means Lambda is **NOT calling SageMaker yet**.

## How to Trigger SageMaker Usage

SageMaker is called during:
1. **Model Ingestion** → Triggers performance metric calculation
2. **Model Rating** → Triggers relationship analysis

### Test Command:
```bash
# 1. Watch Lambda logs in real-time
aws logs tail /aws/lambda/trustworthy-model-registry --follow | grep -i sagemaker

# 2. In another terminal, ingest a model
curl -X POST https://3vfheectz4.execute-api.us-east-1.amazonaws.com/prod/models/ingest \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{"url": "https://huggingface.co/gpt2"}'

# 3. You should see SageMaker logs appear in the first terminal
```

## What You Should See

### In Lambda Logs (`/aws/lambda/trustworthy-model-registry`):
```
SageMaker LLM service initialized: region=us-east-1, endpoint=trustworthy-registry-llm, model_id=meta-textgeneration-llama-3-8b-instruct
SageMaker service available, attempting invocation
Performance metric attempt 1 with AWS SageMaker
Invoking SageMaker endpoint: trustworthy-registry-llm
SageMaker endpoint invocation successful
Performance metric JSON parse succeeded on attempt 1 with SageMaker
```

### In SageMaker Endpoint Logs (`/aws/sagemaker/Endpoints/trustworthy-registry-llm`):
```
POST /invocations HTTP/1.1
Content-Type: application/json
... (inference request)
... (model response)
```

## Troubleshooting

If you don't see SageMaker logs in Lambda:

1. **Check Lambda Environment Variable:**
   ```bash
   aws lambda get-function-configuration \
     --function-name trustworthy-model-registry \
     --query 'Environment.Variables.SAGEMAKER_ENDPOINT_NAME'
   ```
   Should return: `"trustworthy-registry-llm"`

2. **Check Endpoint Status:**
   ```bash
   aws sagemaker describe-endpoint \
     --endpoint-name trustworthy-registry-llm \
     --query 'EndpointStatus'
   ```
   Should return: `"InService"`

3. **Check IAM Permissions:**
   ```bash
   aws iam get-role-policy \
     --role-name lambda-trustworthy-registry-role \
     --policy-name SageMakerInvokePolicy
   ```

