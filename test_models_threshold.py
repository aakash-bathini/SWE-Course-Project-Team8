#!/usr/bin/env python3
"""Test multiple models to find one that passes all thresholds"""
import requests
import json
import time
import sys

# Wait for server
time.sleep(2)

# Authenticate
auth_payload = {
    'user': {'name': 'ece30861defaultadminuser', 'is_admin': True},
    'secret': {'password': 'correcthorsebatterystaple123(!__+@**(A\'"`;DROP TABLE packages;'}
}
try:
    auth_resp = requests.put('http://localhost:8000/authenticate', json=auth_payload, timeout=10)
    if auth_resp.status_code != 200:
        print(f'Auth failed: {auth_resp.status_code}')
        sys.exit(1)
    token_str = auth_resp.text.strip()
    if token_str.startswith('"') and token_str.endswith('"'):
        token_str = json.loads(token_str)
    token = token_str.replace('bearer ', '').replace('Bearer ', '')
    print(f'✅ Authenticated')
except Exception as e:
    print(f'Auth error: {e}')
    sys.exit(1)

headers = {'X-Authorization': f'bearer {token}'}

# Test candidates - models likely to have good documentation and GitHub repos
candidates = [
    'bert-base-uncased',  # Popular, well-documented
    'distilbert-base-uncased-distilled-squad',  # Has GitHub
    'facebook/bart-large',  # Facebook models often have good docs
    't5-small',  # Google model
    'gpt2',  # Very popular
]

for model_name in candidates:
    print(f'\n{"="*60}')
    print(f'Testing: {model_name}')
    print(f'{"="*60}')
    
    try:
        ingest_resp = requests.post(
            f'http://localhost:8000/models/ingest?model_name={model_name}',
            headers=headers,
            timeout=180
        )
        
        print(f'Status: {ingest_resp.status_code}')
        
        if ingest_resp.status_code == 201:
            data = ingest_resp.json()
            model_id = data['metadata']['id']
            print(f'✅ SUCCESS! Model ID: {model_id}')
            
            # Check rate to see actual scores
            print(f'\nChecking rate scores...')
            rate_resp = requests.get(
                f'http://localhost:8000/artifact/model/{model_id}/rate',
                headers=headers,
                timeout=30
            )
            if rate_resp.status_code == 200:
                rate_data = rate_resp.json()
                print(f'Net Score: {rate_data.get("net_score", "N/A")}')
                print(f'Reproducibility: {rate_data.get("reproducibility", "N/A")}')
                print(f'Reviewedness: {rate_data.get("reviewedness", "N/A")}')
                print(f'Tree Score: {rate_data.get("tree_score", "N/A")}')
                print(f'License: {rate_data.get("license", "N/A")}')
                print(f'Ramp Up Time: {rate_data.get("ramp_up_time", "N/A")}')
                print(f'\n🎉 {model_name} PASSED! Use this model.')
                sys.exit(0)
        elif ingest_resp.status_code == 424:
            print(f'❌ FAILED THRESHOLD: {ingest_resp.text[:300]}')
        else:
            print(f'❌ FAILED: {ingest_resp.text[:300]}')
    except Exception as e:
        print(f'Error: {e}')
        continue

print('\n❌ No models passed. The threshold is strict - try manual upload with POST /models/upload')

