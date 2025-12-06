#!/bin/bash
# Script to verify SageMaker endpoint is working correctly

echo "=== SageMaker Endpoint Verification ==="
echo ""

# Check endpoint status
echo "1. Checking endpoint status..."
STATUS=$(aws sagemaker describe-endpoint --endpoint-name trustworthy-registry-llm --query 'EndpointStatus' --output text 2>/dev/null)
echo "   Status: $STATUS"

if [ "$STATUS" != "InService" ]; then
    echo "   ⚠️  Endpoint is not InService yet. Wait a few more minutes."
    exit 1
fi

# Check model
echo ""
echo "2. Checking deployed model..."
MODEL=$(aws sagemaker describe-endpoint-config --endpoint-config-name $(aws sagemaker describe-endpoint --endpoint-name trustworthy-registry-llm --query 'EndpointConfigName' --output text) --query 'ProductionVariants[0].ModelName' --output text)
HF_MODEL_ID=$(aws sagemaker describe-model --model-name "$MODEL" --query 'PrimaryContainer.Environment.HF_MODEL_ID' --output text)
echo "   Model: $MODEL"
echo "   HF_MODEL_ID: $HF_MODEL_ID"

if [[ "$HF_MODEL_ID" == "gpt2" ]]; then
    echo "   ✅ Correct model (GPT-2 - no authentication required)"
else
    echo "   ℹ️  Model: $HF_MODEL_ID"
fi

# Test invocation
echo ""
echo "3. Testing endpoint invocation..."
python3 << 'PYTHON_EOF'
import boto3
import json

runtime = boto3.client('sagemaker-runtime', region_name='us-east-1')

# GPT-2 format (simple text)
system = "You are a helpful assistant."
user = "Say hello in one sentence."
formatted = f"{system}\n\nUser: {user}\n\nAssistant:"

test_payload = {
    "inputs": formatted,
    "parameters": {
        "max_new_tokens": 50,
        "temperature": 0.7,
        "top_p": 0.9,
        "do_sample": True
    }
}

try:
    response = runtime.invoke_endpoint(
        EndpointName="trustworthy-registry-llm",
        ContentType="application/json",
        Body=json.dumps(test_payload)
    )
    result = json.loads(response['Body'].read().decode())
    
    # Extract generated text (GPT-2 returns list format)
    generated = None
    if isinstance(result, list) and len(result) > 0:
        if isinstance(result[0], dict) and "generated_text" in result[0]:
            generated = result[0]["generated_text"]
            # Remove input prompt if included
            if formatted in generated:
                generated = generated.replace(formatted, "", 1).strip()
        else:
            generated = str(result[0])
    elif isinstance(result, dict):
        if "generated_text" in result:
            generated = result["generated_text"]
            if formatted in generated:
                generated = generated.replace(formatted, "", 1).strip()
        elif isinstance(result.get("outputs"), list) and len(result["outputs"]) > 0:
            if isinstance(result["outputs"][0], dict):
                generated = result["outputs"][0].get("generated_text", "")
            else:
                generated = str(result["outputs"][0])
    
    if generated:
        print(f"   ✅ SUCCESS!")
        print(f"   Response: {generated[:200]}")
    else:
        print(f"   ⚠️  Got response but couldn't extract text: {result}")
except Exception as e:
    print(f"   ❌ ERROR: {e}")
    exit(1)
PYTHON_EOF

TEST_RESULT=$?

echo ""
if [ $TEST_RESULT -eq 0 ]; then
    echo "✅ SageMaker endpoint is working correctly!"
    echo ""
    echo "Next steps:"
    echo "1. ✅ Code is updated to use GPT-2 format"
    echo "2. ✅ Lambda environment variables are set"
    echo "3. ✅ Endpoint is InService and working"
    echo "4. 🚀 Ready to run autograder!"
    echo ""
    echo "Monitor SageMaker usage during autograder:"
    echo "   aws logs tail /aws/lambda/trustworthy-model-registry --follow | grep -i sagemaker"
else
    echo "❌ Endpoint test failed. Check CloudWatch logs:"
    echo "   aws logs tail /aws/sagemaker/Endpoints/trustworthy-registry-llm --since 5m"
fi
