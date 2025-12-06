# CloudWatch Logs for SageMaker Verification

## Log Group Location
**Log Group:** `/aws/lambda/trustworthy-model-registry`

## Success Indicators (SageMaker IS Working)

Look for these log messages to confirm SageMaker is being used:

### Initialization
```
SageMaker LLM service initialized: region=us-east-1, endpoint=trustworthy-registry-llm, model_id=meta-textgeneration-llama-3-8b-instruct
```

### Performance Metric (README Analysis)
```
SageMaker service available, attempting invocation
Performance metric attempt 1 with AWS SageMaker
Invoking SageMaker endpoint: trustworthy-registry-llm
SageMaker endpoint invocation successful
Performance metric JSON parse succeeded on attempt 1 with SageMaker
```

### Relationship Analysis
```
SageMaker service available for relationship analysis
Relationship analysis attempt 1 with AWS SageMaker
Invoking SageMaker chat endpoint: trustworthy-registry-llm
SageMaker chat endpoint invocation successful
Relationship analysis JSON parse succeeded on attempt 1 with SageMaker
```

## Failure Indicators (SageMaker NOT Working)

If you see these messages, SageMaker is not being used:

### Missing Configuration
```
SAGEMAKER_ENDPOINT_NAME not set, SageMaker service unavailable
SageMaker service not available (get_sagemaker_service returned None)
SageMaker endpoint name is empty - invocations will fail
```

### Initialization Failures
```
SageMaker Runtime client initialization failed: ...
Failed to initialize SageMaker service: ...
```

### Invocation Failures
```
SageMaker invocation failed: ...
SageMaker chat invocation failed: ...
SageMaker endpoint validation error: ...
SageMaker model error: ...
```

### Fallback to Other LLMs
If you see these, it means SageMaker failed and the system fell back:
```
Performance metric attempt X with Gemini (fallback)
Relationship analysis attempt X with Gemini (fallback)
Performance metric attempt X with Purdue GenAI (fallback)
Relationship analysis attempt X with Purdue GenAI (fallback)
```

## AWS CLI Commands

### Real-time Log Monitoring
```bash
# Follow logs in real-time
aws logs tail /aws/lambda/trustworthy-model-registry --follow

# Filter for SageMaker messages only
aws logs tail /aws/lambda/trustworthy-model-registry --follow | grep -i sagemaker

# Show last hour of logs
aws logs tail /aws/lambda/trustworthy-model-registry --since 1h
```

### Search for Specific Patterns
```bash
# Search for SageMaker initialization
aws logs filter-log-events \
  --log-group-name /aws/lambda/trustworthy-model-registry \
  --filter-pattern "SageMaker LLM service initialized" \
  --start-time $(date -u -d '1 hour ago' +%s)000

# Search for SageMaker invocations
aws logs filter-log-events \
  --log-group-name /aws/lambda/trustworthy-model-registry \
  --filter-pattern "Invoking SageMaker" \
  --start-time $(date -u -d '1 hour ago' +%s)000

# Search for SageMaker errors
aws logs filter-log-events \
  --log-group-name /aws/lambda/trustworthy-model-registry \
  --filter-pattern "SageMaker" \
  --start-time $(date -u -d '1 hour ago' +%s)000 | grep -i "error\|failed\|warning"
```

### Check Recent Log Streams
```bash
# List recent log streams
aws logs describe-log-streams \
  --log-group-name /aws/lambda/trustworthy-model-registry \
  --order-by LastEventTime \
  --descending \
  --max-items 5

# Get events from a specific log stream
aws logs get-log-events \
  --log-group-name /aws/lambda/trustworthy-model-registry \
  --log-stream-name <STREAM_NAME> \
  --limit 100
```

## CloudWatch Console

1. Go to **CloudWatch** → **Logs** → **Log groups**
2. Find `/aws/lambda/trustworthy-model-registry`
3. Click on the most recent log stream
4. Use the search box to filter for:
   - `SageMaker` - All SageMaker-related logs
   - `SageMaker LLM service initialized` - Initialization
   - `Invoking SageMaker` - Actual invocations
   - `SageMaker endpoint invocation successful` - Success
   - `SageMaker.*failed` - Failures

## What to Do If No SageMaker Logs Appear

1. **Check Lambda Environment Variables:**
   ```bash
   aws lambda get-function-configuration \
     --function-name trustworthy-model-registry \
     --query 'Environment.Variables.SAGEMAKER_ENDPOINT_NAME'
   ```
   Should return: `"trustworthy-registry-llm"`

2. **Check SageMaker Endpoint Status:**
   ```bash
   aws sagemaker describe-endpoint \
     --endpoint-name trustworthy-registry-llm \
     --query 'EndpointStatus'
   ```
   Should return: `"InService"`

3. **Check IAM Permissions:**
   ```bash
   aws iam get-role-policy \
     --role-name trustworthy-model-registry-lambda-role \
     --policy-name SageMakerInvokePolicy
   ```

4. **Trigger a Model Ingest:**
   - Upload or ingest a model through the API
   - This triggers performance metric and relationship analysis
   - Watch logs in real-time: `aws logs tail /aws/lambda/trustworthy-model-registry --follow`

## Expected Log Flow During Model Ingest

When a model is ingested, you should see this sequence:

1. Model upload/ingest starts
2. `SageMaker LLM service initialized: ...` (if endpoint is configured)
3. `SageMaker service available, attempting invocation`
4. `Performance metric attempt 1 with AWS SageMaker`
5. `Invoking SageMaker endpoint: trustworthy-registry-llm`
6. `SageMaker endpoint invocation successful`
7. `Performance metric JSON parse succeeded on attempt 1 with SageMaker`
8. Similar sequence for relationship analysis

If you don't see steps 2-7, SageMaker is not being used and the system is falling back to Gemini/Purdue GenAI.
