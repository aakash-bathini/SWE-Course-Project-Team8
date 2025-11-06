# Trustworthy Model Registry - Phase 2 Complete Guide

## 🎯 Project Overview

**ACME Corporation's Trustworthy Model Registry Phase 2** - A secure, scalable model registry built on top of Phase 1 with enhanced security, authentication, and cloud deployment capabilities.

**Team:** Aakash Bathini (@aakash-bathini), Neal Singh (@NSingh1227), Vishal Madhudi (@vishalm3416), Rishi Mantri (@rishimantri795)  
**Group:** 8  
**Track:** Security Extended Track  
**Status:** Milestone 2 Complete ✅ | Milestone 3 Complete ✅ | Testing & Refinement 🚀

---

## 📋 Table of Contents

1. [Quick Start](#quick-start)
2. [Architecture](#architecture)
3. [API Documentation](#api-documentation)
4. [Frontend Guide](#frontend-guide)
5. [AWS Deployment](#aws-deployment)
6. [CI/CD Pipeline](#cicd-pipeline)
7. [Testing](#testing)
8. [Development Workflow](#development-workflow)
10. [Security](#security)
11. [Troubleshooting](#troubleshooting)
12. [Next Steps](#next-steps)

---

Deployed Frontend: https://main.d1vmhndnokays2.amplifyapp.com/dashboard
Deployed Backend: https://han6e7iv6e.execute-api.us-east-1.amazonaws.com

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Node.js 18+
- AWS CLI (for deployment)

### Backend Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Start FastAPI server
python3 -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

### Frontend Setup
```bash
# Install and start frontend
cd frontend
npm install
npm start
```

Open the app at http://localhost:3000 and log in.
In your browser DevTools Console, run:

localStorage.setItem('token', 'demo_token'); location.reload();

After reload you’ll see the nav buttons. Go to:

Dashboard: http://localhost:3000/dashboard
Upload: http://localhost:3000/upload
Search: http://localhost:3000/search
Download: http://localhost:3000/download
Health: http://localhost:3000/health

To log out later:
localStorage.removeItem('token'); location.reload();

### Verify Installation
```bash
# Test API endpoints
curl http://localhost:8000/health
curl http://localhost:8000/docs

# Test authentication
curl -X POST http://localhost:8000/authenticate \
  -H "Content-Type: application/json" \
  -d '{"user": {"name": "ece30861defaultadminuser", "is_admin": true}, "secret": {"password": "correcthorsebatterystaple123(!__+@**(A'"`;DROP TABLE packages;"}}'
```

### Default Credentials
- **Username:** `ece30861defaultadminuser`
- **Password:** `correcthorsebatterystaple123(!__+@**(A'"`;DROP TABLE packages;`

---

## 🏗️ Architecture

### System Overview
```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Web Frontend  │    │   API Clients    │    │   Admin Panel   │
│   (React/TS)    │    │   (Python/JS)    │    │   (React/TS)    │
└─────────┬───────┘    └─────────┬────────┘    └─────────┬───────┘
          │                      │                       │
          └──────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────▼─────────────┐
                    │      FastAPI Backend      │
                    │    (Local + AWS Lambda)   │
                    └─────────────┬─────────────┘
                                  │
        ┌─────────────────────────┼─────────────────────────┐
        │                         │                         │
┌───────▼────────┐    ┌──────────▼──────────┐    ┌────────▼────────┐
│   SQLite DB    │    │       AWS S3        │    │  JWT + bcrypt   │
│   (Local)      │    │   (Deployment)     │    │ (Authentication)│
└────────────────┘    └─────────────────────┘    └─────────────────┘
```

### Directory Structure
```
├── app.py                          # Main FastAPI application
├── requirements.txt                # Python dependencies
├── frontend/                       # React frontend
│   ├── package.json
│   └── src/
│       ├── App.tsx
│       ├── components/
│       └── services/
├── src/
│   ├── auth/                       # JWT authentication
│   ├── aws/                        # AWS deployment
│   ├── api/                        # Phase 1 API integrations
│   ├── metrics/                    # Original Phase 1 metrics + Phase 2 adapter
│   ├── orchestration/              # Phase 2 orchestration using Phase 1 metrics
│   ├── scoring/                    # Original Phase 1 scoring with weights
│   ├── models/                     # Phase 1 model types
│   └── config_parsers_nlp/        # Phase 1 NLP parsing utilities
├── .github/workflows/ci.yml        # CI/CD pipeline
└── README.md                       # This comprehensive guide
```

### Phase 1 Metrics Integration
- ✅ **8 Original Metrics Preserved**: size, license, performance_claims, code_quality, bus_factor, dataset_quality, ramp_up_time, available_dataset_code
- ✅ **Original Calculation Logic**: All Phase 1 metric calculations preserved exactly as implemented
- ✅ **Proper Weighting**: Original Phase 1 weights maintained (size: 0.1, license: 0.1, ramp_up_time: 0.2, etc.)
- ✅ **Phase 2 Adapter**: Seamless integration between web-based Phase 2 and CLI-based Phase 1 metrics
- ✅ **Async Support**: Original metrics work asynchronously in Phase 2 web environment

### Phase 2 New Metrics (Milestone 2)
- ✅ **Reproducibility**: Tests if demo code in model card executes successfully (0.0, 0.5, or 1.0)
- ✅ **Reviewedness**: Calculates fraction of code from reviewed PRs via GitHub API (-1 if no repo, 0.0-1.0 otherwise)
- ✅ **Treescore**: Average net score of parent models in lineage graph (recursive traversal)

---

## 📚 API Documentation

### Authentication Endpoints
```http
PUT /authenticate
Content-Type: application/json

{
  "user": {
    "name": "ece30861defaultadminuser",
    "is_admin": true
  },
  "secret": {
    "password": "correcthorsebatterystaple123(!__+@**(A'"`;DROP TABLE packages;"
  }
}

Response:
{
  "access_token": "demo_token",
  "token_type": "bearer",
  "expires_in": 36000
}
```

```http
POST /register
Authorization: Bearer <token>
Content-Type: application/json

{
  "username": "newuser",
  "password": "password123",
  "permissions": ["search", "download"]
}
```

### Model Management Endpoints
```http
GET /models
Authorization: Bearer <token>

Response: [ModelResponse, ...]
```

```http
POST /models/upload
Authorization: Bearer <token>
Content-Type: multipart/form-data

file: <zip_file>
name: optional_model_name

Response: 201 Created
{
  "metadata": {
    "name": "model_name",
    "id": "model-123-1234567890",
    "type": "model"
  },
  "data": {
    "url": "local://model-123-1234567890"
  }
}
```

```http
GET /models/{id}/rate
Authorization: Bearer <token>

Response:
{
  "name": "model_name",
  "category": "classification",
  "net_score": 0.75,
  "ramp_up_time": 0.8,
  "bus_factor": 0.7,
  "performance_claims": 0.9,
  "license": 1.0,
  "dataset_and_code_score": 0.8,
  "dataset_quality": 0.7,
  "code_quality": 0.8,
  "reproducibility": 1.0,
  "reviewedness": 0.65,
  "tree_score": 0.75,
  "size_score": {
    "raspberry_pi": 0.5,
    "jetson_nano": 0.7,
    "desktop_pc": 0.9,
    "aws_server": 1.0
  }
}
```

```http
GET /models/{id}/download
Authorization: Bearer <token>
Query: ?aspect=full|weights|datasets|code

Response: 200 OK
Content-Type: application/zip
Headers:
  X-File-Checksum: <sha256_hash>
  X-File-Aspect: <aspect>
  
Body: ZIP file with filtered content
```

```http
POST /models/ingest
Authorization: Bearer <token>
Query: ?model_name=huggingface_model_name
```

### System Endpoints
```http
GET /health

Response:
{
  "status": "healthy",
  "timestamp": "2025-10-24T13:25:38.450023",
  "uptime": "0:00:00",
  "models_count": 0,
  "users_count": 1,
  "last_hour_activity": {
    "uploads": 0,
    "downloads": 0,
    "searches": 0
  }
}
```

```http
GET /docs
# Interactive OpenAPI documentation
```

---

## ⚛️ Frontend Guide

### Features
- **Authentication**: Secure login with JWT tokens
- **Dashboard**: Role-based access control and system statistics
- **Model Management**: Upload, search, and download models
- **Health Monitoring**: Real-time system health dashboard
- **Accessibility**: WCAG 2.1 AA compliance

### Development Commands
```bash
cd frontend

# Install dependencies
npm install

# Start development server
npm start

# Build for production
npm run build

# Run tests
npm test

# Type checking
npm run type-check

# Linting
npm run lint
```

### Key Components
- `App.tsx` - Main application with routing
- `LoginPage.tsx` - Authentication interface
- `DashboardPage.tsx` - Main dashboard
- `HealthDashboard.tsx` - System monitoring
- `apiService.ts` - API integration
- `authService.ts` - Authentication management

### Accessibility Features
- Keyboard navigation support
- Screen reader compatibility
- High contrast mode support
- Focus indicators
- Skip links for navigation

---

## ☁️ AWS Deployment

### Environment Variables

#### Backend (Lambda) Environment Variables
The following environment variables are configured automatically via CI/CD when deploying to AWS Lambda:

- `USE_S3=1` - Enable S3 for persistent storage (automatically enabled in production)
- `S3_BUCKET_NAME=trustworthy-registry-artifacts` - S3 bucket name for artifact storage
- `AWS_REGION=us-east-1` - AWS region for S3 bucket
- `ENVIRONMENT=production` - Set environment to production (disables SQLite)
- `LOG_LEVEL=INFO` - Set logging level

**Note:** 
- **Production (Lambda)**: Uses S3 storage only. SQLite is automatically disabled in production.
- **Local Development**: Uses SQLite database (stored in `./registry.db`). Set `USE_SQLITE=1` and `ENVIRONMENT=development` (or omit `ENVIRONMENT`).
- SQLite in `/tmp` is NOT used in production - S3 provides persistent storage across Lambda invocations.

#### Frontend (AWS Amplify) Environment Variables
Configure these in AWS Amplify Console → App Settings → Environment Variables:

- `REACT_APP_API_URL` - Your API Gateway URL (e.g., `https://han6e7iv6e.execute-api.us-east-1.amazonaws.com`)

**Note:** For local development, frontend defaults to `http://localhost:8000` if `REACT_APP_API_URL` is not set.

### Automated Deployment (CI/CD)

The project uses GitHub Actions for automated deployment to AWS. When code is merged to `main` branch:

1. **Backend Deployment:**
   - Packages Lambda function with Python 3.11 runtime
   - Uploads deployment package to S3
   - Updates Lambda function code and configuration
   - Sets environment variables automatically

2. **Frontend Deployment:**
   - Built via AWS Amplify (configured separately)
   - Uses environment variable `REACT_APP_API_URL` for API endpoint

### Manual Backend Deployment (Lambda + API Gateway)

#### 1. Prepare Deployment Package
```bash
# Install production dependencies
pip install -r requirements-prod.txt -t ./lambda_package

# Copy application code
cp app.py lambda_package/
cp -r src/ lambda_package/

# Create deployment zip
cd lambda_package
zip -r9 ../lambda_deployment.zip .
cd ..
```

#### 2. Create IAM Role (if not exists)
```bash
# Create trust policy
cat > trust-policy.json << EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "lambda.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
EOF

# Create role
aws iam create-role \
  --role-name trustworthy-model-registry-lambda-role \
  --assume-role-policy-document file://trust-policy.json

# Attach execution policy
aws iam attach-role-policy \
  --role-name trustworthy-model-registry-lambda-role \
  --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole
```

#### 3. Deploy Lambda Function
```bash
# Get account ID
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

# Upload to S3 (required for packages > 50MB)
aws s3 mb s3://trustworthy-registry-lambda-deployment-us-east-1 2>/dev/null || true
aws s3 cp lambda_deployment.zip s3://trustworthy-registry-lambda-deployment-us-east-1/

# Update Lambda function code
aws lambda update-function-code \
  --function-name trustworthy-model-registry \
  --s3-bucket trustworthy-registry-lambda-deployment-us-east-1 \
  --s3-key lambda_deployment.zip

# Update Lambda configuration (timeout, memory, environment)
aws lambda update-function-configuration \
  --function-name trustworthy-model-registry \
  --timeout 60 \
  --memory-size 1024 \
  --environment "Variables={USE_SQLITE=0,ENVIRONMENT=production,LOG_LEVEL=INFO}"
```

#### 4. Create/Update API Gateway
```bash
# Create REST API (if not exists)
API_ID=$(aws apigateway create-rest-api \
  --name trustworthy-model-registry-team8-api \
  --query 'id' --output text)

# Get root resource ID
ROOT_ID=$(aws apigateway get-resources \
  --rest-api-id $API_ID \
  --query 'items[0].id' --output text)

# Create proxy resource
PROXY_ID=$(aws apigateway create-resource \
  --rest-api-id $API_ID \
  --parent-id $ROOT_ID \
  --path-part '{proxy+}' \
  --query 'id' --output text)

# Create ANY method
aws apigateway put-method \
  --rest-api-id $API_ID \
  --resource-id $PROXY_ID \
  --http-method ANY \
  --authorization-type NONE

# Create Lambda integration
aws apigateway put-integration \
  --rest-api-id $API_ID \
  --resource-id $PROXY_ID \
  --http-method ANY \
  --type AWS_PROXY \
  --integration-http-method POST \
  --uri arn:aws:apigateway:us-east-1:lambda:path/2015-03-31/functions/arn:aws:lambda:us-east-1:${ACCOUNT_ID}:function:trustworthy-model-registry/invocations

# Deploy API
aws apigateway create-deployment \
  --rest-api-id $API_ID \
  --stage-name prod

# Get API URL
API_URL="https://${API_ID}.execute-api.us-east-1.amazonaws.com/prod"
echo "Backend URL: $API_URL"
```

### Frontend Deployment (AWS Amplify)

The frontend is deployed via AWS Amplify. Configure the following:

1. **Connect Repository:** Connect GitHub repository to Amplify
2. **Build Settings:** Use default React build settings
3. **Environment Variables:** Add `REACT_APP_API_URL` pointing to your API Gateway URL
4. **Deploy:** Amplify automatically builds and deploys on push to `main` branch

**Current Production URLs:**
- Frontend: `https://main.d1vmhndnokays2.amplifyapp.com/dashboard`
- Backend: `https://han6e7iv6e.execute-api.us-east-1.amazonaws.com`

### Lambda Configuration

- **Runtime:** Python 3.11
- **Handler:** `app.handler` (Mangum adapter for FastAPI)
- **Timeout:** 60 seconds (increased for metric calculations)
- **Memory:** 1024 MB (increased for HuggingFace API calls)
- **Environment Variables:** Set automatically via CI/CD
  - `USE_S3=1` - S3 enabled for persistent storage (production only)
  - `S3_BUCKET_NAME=trustworthy-registry-artifacts` - S3 bucket name
  - `AWS_REGION=us-east-1` - AWS region
  - `ENVIRONMENT=production` - Production environment (disables SQLite)
  - `LOG_LEVEL=INFO` - Logging level
- **Storage:** S3 provides persistent storage across Lambda invocations (production only)
- **Local Development:** SQLite database stored in `./registry.db` (when `USE_SQLITE=1` and not in production)

### Monitoring & Logging

- **CloudWatch Logs:** Automatically created at `/aws/lambda/trustworthy-model-registry`
- **Retention:** 14 days (free tier limit)
- **Access:** View logs in AWS Console → CloudWatch → Log Groups

### Troubleshooting AWS Deployment

#### Lambda Timeout Errors
- Increase timeout to 60+ seconds for metric calculations
- Increase memory to 1024+ MB for HuggingFace API calls

#### CORS Errors
- Ensure frontend URL is in CORS allow_origins in `app.py`
- Verify API Gateway has CORS enabled

#### Environment Variable Issues
- Verify environment variables are set in Lambda function configuration
- Check CloudWatch logs for environment variable errors

---

## 🔄 CI/CD Pipeline

### GitHub Actions Workflow

The CI/CD pipeline includes 8 comprehensive jobs:

#### **Backend Jobs:**
- ✅ **Backend Lint** - Black formatting + flake8 linting
- ✅ **Backend Typecheck** - MyPy type checking
- ✅ **Backend Test** - pytest with coverage reporting
- ✅ **API Test** - FastAPI endpoint validation

#### **Frontend Jobs:**
- ✅ **Frontend Test** - TypeScript compilation + ESLint + Jest tests

#### **Security & Quality:**
- ✅ **Security Scan** - Bandit + Safety vulnerability scanning
- ✅ **Quality Gate** - 60% coverage threshold enforcement

#### **Deployment:**
- ✅ **Deploy AWS** - Automated Lambda deployment

### GitHub Configuration

#### **Automated Tools:**
- ✅ **Dependabot** - Weekly dependency updates for Python, Node.js, GitHub Actions
- ✅ **GitHub Copilot Auto-Review** - Automated code review with security analysis
- ✅ **Branch Protection** - PR reviews required, status checks must pass
- ✅ **Issue Templates** - Standardized bug reports and feature requests
- ✅ **PR Template** - Comprehensive pull request checklist
- ✅ **CODEOWNERS** - Automatic review assignment based on code changes
- ✅ **Security Policy** - Vulnerability reporting process

#### **Team Assignments:**
- **Backend Python**: @aakash-bathini, @NSingh1227
- **FastAPI App**: @aakash-bathini, @vishalm3416
- **Frontend React**: @rishimantri795, @aakash-bathini
- **CI/CD Workflows**: @NSingh1227, @vishalm3416
- **AWS Deployment**: @NSingh1227, @aakash-bathini
- **Authentication**: @aakash-bathini, @NSingh1227
- **Metrics & Scoring**: @aakash-bathini, @vishalm3416
- **API Integrations**: @vishalm3416, @aakash-bathini

### Testing CI/CD Pipeline

#### **Local Testing:**
```bash
# Test code quality tools
python -m black --check .
python -m flake8 .
python -m mypy .
python -m bandit -r src/
python -m safety check

# Test FastAPI application
python -c "from fastapi.testclient import TestClient; from app import app; client = TestClient(app); print(client.get('/health').status_code)"

# Test Phase 1 metrics integration
python -c "import asyncio; from src.metrics.phase2_adapter import orchestrate_phase2_metrics; print(asyncio.run(orchestrate_phase2_metrics({'url': 'https://huggingface.co/test', 'name': 'test'})))"
```

#### **GitHub Actions Testing:**
1. **Create test branch:**
   ```bash
   git checkout -b test/cicd-pipeline
   echo "# Test CI/CD" >> test_cicd.md
   git add test_cicd.md
   git commit -m "test: trigger CI/CD pipeline"
   git push origin test/cicd-pipeline
   ```

2. **Create Pull Request:**
   - Use the PR template
   - Monitor "Actions" tab for all 8 jobs
   - Verify branch protection blocks merges

3. **Test Dependabot:**
   - Dependabot creates PRs automatically
   - PRs assigned to team members
   - CI/CD runs on Dependabot PRs

---

## 🧪 Testing

### Backend Testing
```bash
# Run all tests with coverage
pytest --cov=src --cov-report=xml

# Run specific tests
pytest tests/test_auth.py
pytest tests/test_api.py

# Run with verbose output
pytest -v
```

### Frontend Testing
```bash
cd frontend

# Run tests
npm test

# Run with coverage
npm test -- --coverage

# Type checking
npm run type-check

# Linting
npm run lint
```

### Integration Testing
```bash
# Test API endpoints
curl http://localhost:8000/health
curl http://localhost:8000/docs

# Test authentication
curl -X POST http://localhost:8000/authenticate \
  -H "Content-Type: application/json" \
  -d '{"user": {"name": "ece30861defaultadminuser", "is_admin": true}, "secret": {"password": "correcthorsebatterystaple123(!__+@**(A'"`;DROP TABLE packages;"}}'
```

### Phase 1 Metrics Testing
```bash
# Test individual metrics
python -c "
import asyncio
from src.metrics.registry import get_all_metrics
from src.models.model_types import EvalContext

async def test_metrics():
    metrics = get_all_metrics()
    context = EvalContext(url='https://huggingface.co/test')
    
    for metric_id, metric_fn in metrics:
        result = await metric_fn(context)
        print(f'{metric_id}: {result}')

asyncio.run(test_metrics())
"

# Test Phase 2 adapter
python -c "
import asyncio
from src.metrics.phase2_adapter import orchestrate_phase2_metrics

async def test_adapter():
    result = await orchestrate_phase2_metrics({
        'url': 'https://huggingface.co/google/gemma-2-2b',
        'name': 'test_model'
    })
    print(f'Net Score: {result.get(\"net_score\", 0)}')
    print(f'Sub-scores: {len(result.get(\"sub_scores\", {}))}')

asyncio.run(test_adapter())
"
```

---

## 🔄 Development Workflow

### Code Standards
- **Python**: Black formatting, flake8 linting, mypy type checking
- **TypeScript**: ESLint, strict type checking
- **Testing**: Minimum 60% line coverage
- **Documentation**: OpenAPI specs, docstrings

### Git Workflow
```bash
# Create feature branch
git checkout -b feature/new-feature

# Make changes and commit
git add .
git commit -m "Add new feature"

# Push and create PR
git push origin feature/new-feature
```

### Security Practices
- All PRs require code review
- Security-sensitive code requires 2+ reviewers
- Input validation on all endpoints
- Secure password storage with bcrypt
- Automated security scanning with Bandit and Safety
- Vulnerability reporting via SECURITY.md

---

## 🔒 Security

### Authentication & Authorization
- JWT token-based authentication
- Role-based access control (RBAC)
- Password hashing with bcrypt
- Token expiration and validation

### Input Validation
- Request validation using Pydantic models
- SQL injection prevention
- XSS protection
- CSRF protection

### Dependencies
- Regular dependency updates via Dependabot
- Security scanning with Safety
- Vulnerability scanning with Bandit

### Infrastructure
- HTTPS enforcement
- Secure headers
- Environment variable protection
- AWS IAM role-based access

### Security Scanning
We use automated security scanning tools:
- **Bandit**: Python security linter
- **Safety**: Dependency vulnerability scanner
- **GitHub Security Advisories**: Automated vulnerability detection
- **Dependabot**: Automated dependency updates

### Security Checklist
Before submitting code, ensure:
- [ ] No hardcoded secrets or credentials
- [ ] Input validation implemented
- [ ] Authentication checks in place
- [ ] Authorization properly enforced
- [ ] Error messages don't leak sensitive information
- [ ] Dependencies are up to date
- [ ] Security tests pass

### Reporting Vulnerabilities
If you discover a security vulnerability:
1. **DO NOT** create a public GitHub issue
2. Send email to: **abathin@purdue.edu**
3. Include: Description, steps to reproduce, impact assessment, suggested fix

**Response Timeline:**
- Initial Response: Within 48 hours
- Status Update: Within 7 days
- Resolution: Within 30 days

---

## 🚨 Troubleshooting

### Common Issues

#### **Port Already in Use**
```bash
# Kill processes using port 8000
lsof -ti:8000 | xargs kill -9

# Or use a different port
python3 -m uvicorn app:app --port 8001
```

#### **CORS Errors**
- Check CORS configuration in FastAPI
- Verify allowed origins include frontend domain

#### **Lambda Timeout**
- Increase timeout in Lambda configuration
- Optimize code for faster execution

#### **API Gateway 502 Errors**
- Check Lambda function logs
- Verify integration configuration

#### **Frontend Not Loading**
- Check S3 bucket policy
- Verify CloudFront distribution

### Debug Commands
```bash
# Test Lambda function directly
aws lambda invoke \
  --function-name trustworthy-model-registry-team8 \
  --payload '{"httpMethod": "GET", "path": "/health"}' \
  response.json

# Check API Gateway logs
aws logs describe-log-groups --log-group-name-prefix /aws/apigateway/

# Test S3 website
curl -I http://trustworthy-model-registry-team8-frontend.s3-website-us-east-1.amazonaws.com

# Check running processes
ps aux | grep uvicorn
ps aux | grep python
```

### Environment Issues
```bash
# Check Python version
python3 --version

# Check Node version
node --version

# Check AWS CLI
aws --version

# Check installed packages
pip list | grep fastapi
npm list --depth=0
```

---

## 📊 Monitoring & Logs

### CloudWatch Logs
```bash
# View Lambda logs
aws logs describe-log-groups --log-group-name-prefix /aws/lambda/trustworthy-model-registry-team8

# Stream logs
aws logs tail /aws/lambda/trustworthy-model-registry-team8 --follow
```

### Local Logging
```bash
# View application logs
tail -f app.log

# Check error logs
grep ERROR app.log
```

### Health Monitoring
- Real-time system metrics
- Activity monitoring (uploads, downloads, searches)
- User and model counts
- System uptime tracking

---

## 🎯 Milestone 2 Implementation Details

### What Was Implemented

#### 0. Delivery 1 Compliance Checklist ✅
- CI/CD: GitHub Actions runs tests on every PR; deploys to AWS on merge
- CRUD: Upload, Rate, Download, Delete endpoints conform to OpenAPI
- Ingest: `POST /models/ingest` filters on ≥0.5 for all non-latency metrics and auto-uploads
- Enumerate: `GET /models` supports cursor-based pagination for large datasets
- AWS: Uses multiple components (Lambda, API Gateway, S3, CloudWatch)
- Default user: `ece30861defaultadminuser` present; if auth enabled, password as specified
- OpenAPI: `/docs` served; schema validation enforced in CI

#### 1. Download Endpoint ✅
**Endpoint**: `GET /models/{id}/download?aspect=<type>`
- **Location**: `app.py` lines 544-634
- **Aspects**: full, weights, datasets, code
- **Features**: 
  - SHA256 checksums for integrity verification
  - File streaming for large files
  - Sub-aspect filtering (full/weights/datasets/code)
  - Proper error handling for missing files
  - ZIP compression for downloads
- **Storage**: `uploads/{artifact_id}/` directory

#### 2. Upload Endpoint ✅
**Endpoint**: `POST /models/upload`
- **Location**: `app.py` lines 417-542
- **Accepts**: multipart/form-data with ZIP file
- **Features**: 
  - ZIP validation and extraction
  - Model card parsing (README.md)
  - SHA256 checksum calculation
  - Automatic metrics calculation trigger
  - SQLite metadata storage
  - Audit logging
- **Storage**: `uploads/{artifact_id}/` directory

#### 3. New Phase 2 Metrics ✅

**Reproducibility** (`src/metrics/reproducibility.py` - 137 lines)
- **Logic**: Extracts and safely runs demo code from model card
- **Scores**: 
  - 0.0 = Code fails to execute
  - 0.5 = Partial execution success
  - 1.0 = Perfect execution
- **Safety**: 5-second timeout, isolated subprocess

**Reviewedness** (`src/metrics/reviewedness.py` - 168 lines)
- **Logic**: GitHub API analysis of PR reviews
- **Scores**: 
  - -1.0 = No GitHub repository found
  - 0.0-1.0 = Fraction of commits from reviewed PRs
- **Features**: Respects GITHUB_TOKEN env var, rate limit handling

**Treescore** (`src/metrics/treescore.py` - 158 lines)
- **Logic**: Average net score of parent models in lineage graph
- **Scores**: 
  - 0.0 = No parent models
  - 0.0-1.0 = Average of parent scores (recursive)
- **Features**: Limits to 5 parents, uses simplified metrics to avoid recursion

#### 4. Enumerate Endpoint ✅
**Endpoint**: `GET /models`
- **Location**: `app.py` lines 801-859
- **Pagination**: Cursor-based (`cursor`, `limit`), stable ordering, default `limit=25`
- **Response**: `{ "items": [...], "next_cursor": "..." }`
- **Features**:
  - Cursor-based pagination for efficient handling of millions of models
  - Filters to only return model artifacts (not datasets/code)
  - Stable sorting by artifact ID for consistent pagination
  - Permission check for "search" permission
- **Notes**: Designed to avoid DoS by paginating large registries; results power directory view

#### 5. Ingest Endpoint ✅
**Endpoint**: `POST /models/ingest?model_name=<huggingface_id>`
- **Location**: `app.py` lines 700-798
- **Behavior**: 
  - Fetches HuggingFace metadata via `scrape_hf_url`
  - Computes all 11 metrics (8 Phase 1 + 3 Phase 2)
  - Enforces ≥0.5 threshold on **all non-latency metrics** (excludes metrics ending with "_latency")
  - On pass: Creates artifact entry and logs audit trail
  - On fail: Returns 424 with detailed list of failing metrics
- **Features**:
  - Permission check for "upload" permission
  - Proper error handling for invalid model names
  - Integrates with SQLite storage if enabled
  - Audit logging for all ingest operations
- **Testing**: External calls stubbed in tests; idempotent for the same source

### New Files Added (6 total, ~1,040 lines)

1. `src/metrics/reproducibility.py` - Reproducibility metric (137 lines)
2. `src/metrics/reviewedness.py` - Reviewedness metric (168 lines)
3. `src/metrics/treescore.py` - Treescore metric (158 lines)
4. `src/storage/file_storage.py` - Storage utilities (251 lines)
5. `src/storage/__init__.py` - Module init (1 line)
6. `tests/test_milestone2_features.py` - Test suite (325 lines)

### Testing

#### Run Tests
```bash
# All tests
pytest -v

# Milestone 2 only
pytest tests/test_milestone2_features.py -v

# With coverage
pytest --cov=src --cov=app --cov-report=term-missing
```

#### Manual Testing
```bash
# Start server
python -m uvicorn app:app --reload

# Upload model
curl -X POST http://localhost:8000/models/upload \
  -H "X-Authorization: bearer <TOKEN>" \
  -F "file=@model.zip" \
  -F "name=test_model"

# Download model (get ID from upload response)
curl -X GET "http://localhost:8000/models/model-1-123/download?aspect=full" \
  -H "X-Authorization: bearer <TOKEN>" \
  -o downloaded.zip

# Check rating with new metrics
curl -X GET "http://localhost:8000/artifact/model/model-1-123/rate" \
  -H "X-Authorization: bearer <TOKEN>"

# Ingest HuggingFace model (with threshold validation)
curl -X POST "http://localhost:8000/models/ingest?model_name=google/gemma-2-2b" \
  -H "X-Authorization: bearer <TOKEN>"

# Enumerate models (cursor-based pagination)
curl -X GET "http://localhost:8000/models?limit=25" \
  -H "X-Authorization: bearer <TOKEN>"

# Enumerate next page using cursor
curl -X GET "http://localhost:8000/models?cursor=model-2-1234567890&limit=25" \
  -H "X-Authorization: bearer <TOKEN>"
```

### Key Features

#### Upload Features
- ✅ ZIP validation
- ✅ Automatic extraction
- ✅ Model card parsing (README.md)
- ✅ SHA256 checksum calculation
- ✅ Metric calculation trigger
- ✅ SQLite metadata storage
- ✅ Audit logging

#### Download Features
- ✅ Sub-aspect filtering (full/weights/datasets/code)
- ✅ SHA256 integrity checks
- ✅ File streaming for large files
- ✅ ZIP compression
- ✅ Proper error handling
- ✅ Audit logging

#### Ingest Features
- ✅ HuggingFace model name parsing
- ✅ Threshold validation (≥0.5 on all non-latency metrics)
- ✅ Automatic artifact creation on success
- ✅ Detailed error messages for threshold failures
- ✅ Permission-based access control
- ✅ Audit logging

#### Enumerate Features
- ✅ Cursor-based pagination (not offset-based)
- ✅ Stable sorting by artifact ID
- ✅ Permission-based access control

#### Artifact Search & Retrieval Features ✅
**Endpoint**: `GET /artifact/byName/{name}` (NON-BASELINE)
- **Location**: `app.py` lines 1158-1194
- **Features**:
  - Exact name matching for stored artifact names
  - Support for full HuggingFace model names (e.g., `org/model-name`)
  - Returns all artifacts matching the name (multiple artifacts can share a name)
  - Works with both ingested models and uploaded artifacts

**Endpoint**: `POST /artifact/byRegEx` (BASELINE)
- **Location**: `app.py` lines 1197-1284
- **Features**:
  - Case-insensitive regex matching
  - Searches artifact names individually (for exact matches)
  - Searches both stored name and full HuggingFace model name (`hf_model_name`)
  - Searches README content from HuggingFace models
  - Supports exact match patterns like `^model-name$`
  - Supports partial match patterns like `.*model.*`
  - Maximum regex length of 500 characters to prevent ReDoS attacks
- **Behavior**:
  - Checks `name`, `hf_model_name`, and `readme_text` individually for exact matches
  - Also checks concatenated text for partial matches
  - Returns empty list if pattern is too long (500+ chars) or invalid

**Endpoint**: `GET /artifacts/{artifact_type}/{id}` (BASELINE)
- **Location**: `app.py` lines 1400-1435
- **Features**:
  - Retrieves artifact by ID and type
  - Type validation (returns 400 if type mismatch)
  - 404 if artifact doesn't exist
  - Debug logging for troubleshooting (CloudWatch logs)
  - Works with both SQLite and in-memory storage
- ✅ Efficient for large datasets (millions of models)
- ✅ Stable sorting for consistent pagination
- ✅ Configurable page size (1-100, default 25)
- ✅ Permission-based access control

#### Metric Features
- ✅ Reproducibility: Code execution testing
- ✅ Reviewedness: GitHub PR analysis
- ✅ Treescore: Lineage graph traversal
- ✅ All metrics in `/rate` endpoint
- ✅ Error handling with safe defaults
- ✅ Threshold logic excludes latency metrics correctly

### API Changes

#### Modified Endpoints
- `GET /models/{id}/download` - Now fully implemented (was 501 placeholder)
- `POST /models/upload` - Now fully implemented (was 501 placeholder)
- `GET /artifact/model/{id}/rate` - Now returns 11 metrics (was 8)
- `POST /artifact/{artifact_type}` - Fixed threshold logic to exclude latency metrics

#### New Endpoints
- `POST /models/ingest` - Ingest HuggingFace models with threshold validation
- `GET /models` - Enumerate models with cursor-based pagination

#### New Response Fields
Rating endpoint now includes:
```json
{
  "reproducibility": 1.0,
  "reviewedness": 0.75,
  "tree_score": 0.68,
  // ... 8 other Phase 1 metrics
}
```

### Configuration

#### Environment Variables (Optional)
- `GITHUB_TOKEN` - For reviewedness metric (avoids rate limits)
- `USE_SQLITE=1` - Enable SQLite storage (default: enabled in production, disabled locally)
- `USE_S3` - Enable S3 storage (default: auto-enabled in production/Lambda)
- `S3_BUCKET_NAME` - S3 bucket name for artifact storage (required if USE_S3=1)
- `AWS_REGION` - AWS region for S3 bucket (default: us-east-1)
- `USE_SQLITE` - Enable SQLite storage (default: disabled in production, enabled locally)
- `SQLALCHEMY_DATABASE_URL` - Override database path (default: `./registry.db` locally, not used in production)

#### Storage Location
- **Production (Lambda):**
  - **S3:** `s3://trustworthy-registry-artifacts/artifacts/{artifact_id}/metadata.json`
  - **SQLite:** Disabled automatically in production
- **Local Development:**
  - **SQLite Database:** `./registry.db` (current directory)
  - **Uploaded Files:** `uploads/{artifact_id}/` (local development only)

## 🎯 Next Steps

### Milestone 2 Completion Status ✅
1. **Full CRUD Implementation**: ✅ Complete
   - Upload: ZIP file upload with extraction and validation
   - Download: File streaming with sub-aspect filtering (full/weights/datasets/code)
   - Rate: All 11 metrics (8 Phase 1 + 3 Phase 2)
   - Delete: Full implementation with audit logging
2. **Enumerate**: ✅ `GET /models` with cursor-based pagination
3. **Ingest**: ✅ Threshold gating (≥0.5 non-latency) then auto-upload
4. **Database Integration**: ✅ SQLite fully integrated
5. **Enhanced Authentication**: ✅ JWT with proper token validation
6. **New Metrics**: ✅ Reproducibility, Reviewedness, Treescore implemented
7. **Integration Testing**: ✅ Comprehensive test suite with 40%+ coverage
8. **Local File Storage**: ✅ uploads/ directory with SHA256 integrity checks
9. **OpenAPI Compliance**: ✅ `/docs` available; responses align with schema

### Recent Improvements (Latest Updates)

#### Artifact Search Enhancements ✅
- **Improved Regex Matching**: Exact match patterns (e.g., `^model-name$`) now work correctly by checking name, HuggingFace model name, and README content individually
- **Case-Insensitive Search**: All regex searches are now case-insensitive for better user experience
- **Enhanced Name Matching**: `/artifact/byName/{name}` now supports both display names and full HuggingFace model names (e.g., `google/gemma-2-2b`)
- **Debug Logging**: Added comprehensive logging to artifact retrieval endpoint for troubleshooting autograder issues

#### Performance & Reliability
- **Better Error Handling**: Improved error messages for artifact retrieval failures
- **Type Safety**: Enhanced type validation for artifact type matching
- **Storage Consistency**: Ensured artifacts created via different endpoints (ingest, upload, create) are all retrievable consistently

### Remaining for Deployment
- **AWS Deployment**: Lambda + API Gateway configuration (handled separately)
- **CloudWatch Logging**: Production monitoring setup

### Future Enhancements
- **Security Track Features**: Sensitive models, package confusion detection
- **Performance Optimization**: Caching, parallel processing
- **Enhanced UI**: Advanced search, model comparison
- **S3 Storage**: Replace local storage with AWS S3 for production

---

## 📞 Support

For questions or issues:
- Check the GitHub Issues
- Review the API documentation at `/docs`
- Test API compatibility
- Contact team members via GitHub

---

## 🏆 Success Metrics

- **API Endpoints**: 20+ endpoints implemented ✅
- **CRUD Operations**: Upload, Download, Rate, Delete ✅
- **Phase 1 Metrics**: 8 metrics integrated ✅
- **Phase 2 Metrics**: 3 new metrics (reproducibility, reviewedness, treescore) ✅
- **Authentication**: JWT with bcrypt ✅
- **File Storage**: Local uploads/ with SHA256 checksums ✅
- **Sub-aspect Downloads**: full/weights/datasets/code filtering ✅
- **Health Monitoring**: /health and /health/components ✅
- **Error Handling**: Comprehensive 400/401/403/404/500 responses ✅
- **CORS Configuration**: Frontend integration ready ✅
- **Database**: SQLite with full schema ✅
- **CI/CD Pipeline**: 5-stage pipeline with auto-testing ✅
- **Test Coverage**: 40%+ with comprehensive test suite ✅
- **Documentation**: Complete API docs ✅

---

**🎉 MILESTONE 2 STATUS: ✅ COMPLETE**

All CRUD operations, new metrics, upload/download functionality, and testing are fully implemented and ready for production deployment.
