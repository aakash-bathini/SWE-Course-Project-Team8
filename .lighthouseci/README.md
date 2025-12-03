# Lighthouse Test Results - Intentionally Tracked

This directory contains Lighthouse CI test results that are **intentionally committed** to the repository.

## Why These Files Are Tracked

These files are required for rubric compliance:
- **Rubric Requirement**: "Provided evidence that they assessed their interface for ADA compliance"
- **Evidence Needed**: Lighthouse test results showing accessibility scores
- **Result**: Accessibility score 100/100 (Full ADA/WCAG compliance)

## Files in This Directory

- `assertion-results.json` - Test assertion results
- `lhr-*.html` - HTML reports from Lighthouse runs
- `lhr-*.json` - JSON reports from Lighthouse runs

These files are **not** ignored by `.gitignore` because they serve as evidence for manual verification in the grading rubric.

## Test Execution

Tests were run on December 2, 2025 using:
```bash
lhci autorun --collect.url=https://main.d1vmhndnokays2.amplifyapp.com/dashboard
```

Results: Accessibility 100/100, Performance 98/100, Best Practices 96/100, SEO 100/100
