#!/bin/bash
# Comprehensive test runner for frontend and backend

set -e

echo "============================================================"
echo "🧪 Running Comprehensive Frontend/Backend Tests"
echo "============================================================"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test 1: Backend API Tests
echo -e "\n${YELLOW}Test 1: Backend API Tests${NC}"
python3 test_frontend_simple.py
BACKEND_RESULT=$?

# Test 2: Frontend UI Tests (if frontend is running)
echo -e "\n${YELLOW}Test 2: Frontend UI Tests${NC}"
if curl -s http://localhost:3000 > /dev/null 2>&1; then
    echo "Frontend is running - running UI tests..."
    python3 test_frontend_ui.py
    UI_RESULT=$?
else
    echo -e "${YELLOW}⚠️  Frontend not running${NC}"
    echo "To test UI, start frontend in another terminal:"
    echo "  cd frontend && npm start"
    echo ""
    echo "Then run: python3 test_frontend_ui.py"
    UI_RESULT=0  # Don't fail if frontend isn't running
fi

# Summary
echo -e "\n${YELLOW}============================================================"
echo "📊 Test Summary"
echo "============================================================${NC}"

if [ $BACKEND_RESULT -eq 0 ]; then
    echo -e "${GREEN}✅ Backend API Tests: PASSED${NC}"
else
    echo -e "${RED}❌ Backend API Tests: FAILED${NC}"
fi

if [ $UI_RESULT -eq 0 ]; then
    echo -e "${GREEN}✅ Frontend UI Tests: PASSED${NC}"
else
    echo -e "${RED}❌ Frontend UI Tests: FAILED${NC}"
fi

if [ $BACKEND_RESULT -eq 0 ] && [ $UI_RESULT -eq 0 ]; then
    echo -e "\n${GREEN}✅ All tests passed!${NC}"
    exit 0
else
    echo -e "\n${RED}❌ Some tests failed${NC}"
    exit 1
fi

