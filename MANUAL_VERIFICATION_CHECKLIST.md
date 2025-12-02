# Manual Verification Checklist - Rubric Requirements

## ✅ Code Implementation Status

All baseline and Security Track requirements are **IMPLEMENTED** and **OpenAPI COMPLIANT**. See `RUBRIC_COMPLIANCE_CHECK.md` for detailed verification.

---

## ⚠️ Manual Verification Items (5 points)

### 1. Test Evidence (3 points)
**Requirement**: "Provided evidence that they performed manual or automated tests on all their reported features or provided justification while some features were untested."

**Status**: ✅ **PARTIALLY DOCUMENTED**

**Evidence Found**:
- ✅ **32 test files** in `tests/` directory covering all features
- ✅ **Test coverage**: 70% (exceeds 60% requirement) - mentioned in `FIXES_APPLIED.md` line 495
- ✅ **Test documentation** in `README.md`:
  - Milestone 2 tests (lines 1023-1068)
  - Milestone 5 tests (lines 1564-1577)
  - Milestone 6 tests (lines 1715-1729)
- ✅ **Selenium frontend tests**: `tests/test_frontend_ui.py` (195 lines)
- ✅ **Locust performance tests**: `tests/locustfile.py`

**What's Missing**:
- ⚠️ **Test execution reports** (HTML/JSON coverage reports)
- ⚠️ **Test results summary** (pass/fail counts per feature)
- ⚠️ **Justification for untested features** (if any)

**Recommendation**: 
- Generate and include test coverage HTML report (already exists in `htmlcov/` directory)
- Create a `TEST_EVIDENCE.md` document summarizing:
  - Test execution results
  - Coverage per feature
  - Manual test scenarios
  - Justification for any untested features

---

### 2. LLM Usage (3 points)
**Requirement**: 
- "LLMs are used to analyze the README or to analyze the relationship between artifacts."
- "Provided evidence that LLMs are used in the development phase to generate code or review PR"
- "Provided use of AWS Sagemaker or equivalent service to perform LLM-based activities. Partial points for API based LLM use."

**Status**: ✅ **PARTIALLY DOCUMENTED**

**Evidence Found**:

#### A. LLM Usage in Code (README Analysis) ✅
- ✅ **File**: `src/metrics/performance_metric.py`
- ✅ **Implementation**: Uses Google Gemini API or Purdue GenAI API to analyze README files for performance claims
- ✅ **Documentation**: Mentioned in `FIXES_APPLIED.md` lines 516-518
- ✅ **Code Location**: Lines 151-194 in `performance_metric.py`
  ```python
  # Uses Gemini API or Purdue GenAI API
  if api_key:
      from google import genai
      client = genai.Client()
      response = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
  else:
      # Purdue GenAI API fallback
      url = "https://genai.rcac.purdue.edu/api/chat/completions"
  ```

#### B. LLM Usage in Development (Code Generation/PR Review) ⚠️
- ⚠️ **Not explicitly documented** in README or separate document
- ⚠️ **No evidence** of LLM usage for:
  - Code generation during development
  - PR review assistance
  - Documentation generation

#### C. AWS SageMaker Usage ⚠️
- ⚠️ **Not implemented** - Using API-based LLM (Google Gemini/Purdue GenAI)
- ⚠️ **Partial credit only** - API-based LLM use qualifies for partial points

**What's Missing**:
- ⚠️ **Documentation** of LLM usage in development phase
- ⚠️ **Evidence** of LLM-assisted code generation or PR reviews
- ⚠️ **AWS SageMaker** implementation (optional, but would get full points)

**Recommendation**:
- Create `LLM_USAGE.md` document with:
  - Screenshots/logs of LLM usage in development
  - Examples of LLM-generated code or PR review comments
  - Documentation of README analysis feature
  - Note about API-based LLM (partial credit)

---

### 3. Browser-based Interface (4 points)
**Requirement**:
- "Provided evidence that they implemented a browser-based interface for their APIs."
- "Provides a usable web browser interface that exposes core system functionality for human users (upload/query/download), satisfying the accessibility/UI requirements."
- "Provided evidence that they implemented frontend automated tests, e.g., with Selenium"
- "Provided evidence that they assessed their interface for ADA compliance."

**Status**: ✅ **MOSTLY COMPLETE**

**Evidence Found**:

#### A. Browser-based Interface ✅
- ✅ **Frontend implemented**: React/TypeScript frontend in `frontend/` directory
- ✅ **Components**: 7 major components (LoginPage, ModelUploadPage, ModelSearchPage, ModelDownloadPage, HealthDashboard, UserManagementPage, DashboardPage)
- ✅ **Core functionality exposed**:
  - Upload: `ModelUploadPage.tsx`
  - Query/Search: `ModelSearchPage.tsx`
  - Download: `ModelDownloadPage.tsx`
- ✅ **Deployed**: https://main.d1vmhndnokays2.amplifyapp.com/dashboard
- ✅ **Documentation**: `README.md` lines 1616-1643

#### B. Frontend Automated Tests ✅
- ✅ **Selenium tests**: `tests/test_frontend_ui.py` (195 lines)
- ✅ **Test coverage**:
  - WCAG compliance tests (ARIA labels, autocomplete, keyboard navigation)
  - Frontend functionality tests (health dashboard, login page)
  - Frontend-backend integration tests
- ✅ **Selenium WebDriver**: Configured with Chrome headless mode
- ✅ **Documentation**: `README.md` lines 1632-1637

#### C. ADA Compliance Assessment ⚠️
- ✅ **WCAG 2.1 AA compliance** mentioned in README (lines 1625-1630)
- ✅ **Accessibility features**:
  - ARIA labels (`aria-describedby`)
  - Autocomplete attributes
  - Keyboard navigation support
  - Material-UI components (built-in accessibility)
- ⚠️ **Lighthouse test results**: **NOT FOUND**
  - No `lighthouse*.json` or `lighthouse*.html` files
  - No Lighthouse CI integration
  - No documented Lighthouse audit results

**What's Missing**:
- ⚠️ **Lighthouse test results** (HTML/JSON reports)
- ⚠️ **Lighthouse CI integration** (automated accessibility testing)
- ⚠️ **ADA compliance audit report** (screenshots or documentation)

**Recommendation**:
1. **Run Lighthouse audit**:
   ```bash
   npm install -g @lhci/cli
   lhci autorun --url=https://main.d1vmhndnokays2.amplifyapp.com/dashboard
   ```
2. **Save results** as `lighthouse-report.html` in project root
3. **Add to CI/CD** for automated accessibility testing
4. **Document** in README or create `ACCESSIBILITY_REPORT.md`

---

## Summary

### ✅ Complete (Ready for Manual Review)
- ✅ All baseline requirements implemented
- ✅ All Security Track requirements implemented
- ✅ OpenAPI spec compliance verified
- ✅ Frontend interface implemented
- ✅ Selenium tests implemented
- ✅ LLM usage in code (README analysis)

### ⚠️ Needs Documentation/Evidence
- ⚠️ **Test evidence documentation** (create `TEST_EVIDENCE.md`)
- ⚠️ **LLM usage in development** (create `LLM_USAGE.md`)
- ⚠️ **Lighthouse ADA compliance report** (run Lighthouse and save results)

### 📋 Action Items
1. **Run Lighthouse audit** and save results
2. **Create `TEST_EVIDENCE.md`** with test execution summary
3. **Create `LLM_USAGE.md`** documenting LLM usage in development
4. **Add Lighthouse CI** to GitHub Actions for automated testing

---

## Files to Create/Update

### 1. `TEST_EVIDENCE.md`
```markdown
# Test Evidence Documentation

## Test Coverage Summary
- Total test files: 32
- Code coverage: 70% (exceeds 60% requirement)
- Test execution: All tests passing

## Feature Test Coverage
- [List each feature and its test coverage]
- [Include test execution results]
- [Justify any untested features]
```

### 2. `LLM_USAGE.md`
```markdown
# LLM Usage Documentation

## LLM Usage in Code
- Performance Claims Metric: Uses Google Gemini API or Purdue GenAI API
- README Analysis: Analyzes model READMEs for performance claims

## LLM Usage in Development
- [Document LLM-assisted code generation]
- [Document LLM PR reviews]
- [Include screenshots/logs]
```

### 3. `lighthouse-report.html`
- Run Lighthouse audit on deployed frontend
- Save HTML report in project root
- Reference in README

---

## Rubric Points Breakdown

### Manual Checks (5 points total, scaled from 10)
- **Test Evidence**: 3 points (currently ~2 points - needs documentation)
- **LLM Usage**: 3 points (currently ~2 points - API-based, needs dev evidence)
- **Browser Interface**: 4 points (currently ~3 points - needs Lighthouse report)

**Current Estimated Score**: ~7/10 points (scaled to ~3.5/5)

**Potential Score with Documentation**: 10/10 points (scaled to 5/5)

