# Autograder Fixes Applied - November 18, 2025

## Issues Identified and Fixed

### Issue 1: Reset Endpoint Not Clearing In-Memory Artifacts in S3 Mode
**Problem**: When `USE_S3=True` (Lambda production), the reset endpoint was NOT clearing the in-memory `artifacts_db` dictionary. This caused:
- After reset, old artifacts persisted in `artifacts_db` in-memory
- Subsequent uploads could see stale artifacts from previous requests
- Potential conflicts when artifacts exist in both S3 and in-memory with different states

**Root Cause**: The reset endpoint had this logic:
```python
if USE_S3 and s3_storage:
    s3_storage.clear_all_artifacts()  # Clears S3
elif USE_SQLITE:
    db_crud.reset_registry(_db)  # Clears SQLite
else:
    artifacts_db.clear()  # Only clears in-memory as fallback!
```

**Fix Applied** (app.py, line ~1703-1710): 
Always clear `artifacts_db` regardless of storage backend:
```python
# CRITICAL: Always clear in-memory artifacts_db regardless of storage backend
artifacts_db.clear()

if USE_S3 and s3_storage:
    s3_storage.clear_all_artifacts()
elif USE_SQLITE:
    db_crud.reset_registry(_db)
```

### Issue 2: S3 Bucket Creation Fails for us-east-1 Region
**Problem**: When creating S3 buckets in us-east-1 (default region), the code was incorrectly specifying `CreateBucketConfiguration`, which is not allowed for the default region. This could cause:
- Bucket creation to fail with `InvalidLocationConstraint` error
- Artifact uploads failing silently when trying to save to S3
- All read operations returning 404 because artifacts weren't persisted

**Root Cause** (s3_storage.py, line ~176-180):
```python
self.s3_client.create_bucket(
    Bucket=self.bucket_name,
    CreateBucketConfiguration={"LocationConstraint": self.region},  # ERROR for us-east-1!
)
```

For us-east-1, AWS requires:
- NO `CreateBucketConfiguration` parameter (the region is implicit)
- For OTHER regions, `CreateBucketConfiguration` is required

**Fix Applied** (s3_storage.py, line ~165-195):
```python
if self.region and self.region != "us-east-1":
    self.s3_client.create_bucket(
        Bucket=self.bucket_name,
        CreateBucketConfiguration={"LocationConstraint": self.region},
    )
else:
    # us-east-1 is the default, no config needed
    self.s3_client.create_bucket(Bucket=self.bucket_name)
```

## Test Failure Analysis

### Artifact Read Tests: 16/49 Passing (~67% failure)
**Expected Impact**: Both fixes should improve this:
1. Reset clearing `artifacts_db` prevents stale data
2. S3 bucket creation fix ensures artifacts are actually persisted

### Rate Tests: 0/11 Passing (Complete failure)
**Expected Impact**: Rate tests depend on artifact retrieval, so S3 fix should help these

### Regex Tests: 4/6 Passing
**Expected Impact**: Regex tests also depend on artifact retrieval, should improve

### Upload Tests: 29/29 Passing ✅
**No changes needed** - these were already working

### Setup/Reset Tests: 6/6 Passing ✅
**Expected Impact**: Our reset fix might improve reliability

## Files Modified

1. **app.py** (line ~1703-1710)
   - Fixed reset endpoint to always clear `artifacts_db`

2. **src/storage/s3_storage.py** (line ~165-195)
   - Fixed S3 bucket creation for us-east-1 region

## Testing Recommendations

To verify these fixes:

1. **Run reset test**: Ensure `/reset` clears all storage layers
2. **Run upload test**: Upload artifacts via `/models/ingest` 
3. **Run read test**: Immediately read same artifact in new Lambda instance
4. **Run concurrent tests**: Test with multiple simultaneous read/rate requests

## Key Insights

The main issue was that **Lambda instances are ephemeral** - each cold start gets a fresh `artifacts_db` dictionary. The code correctly handles this by checking S3 as a fallback, but:

1. **Reset not clearing in-memory caused cache pollution** - artifacts from previous test runs could interfere
2. **S3 bucket creation issue prevented persistence** - without a working S3 bucket, nothing gets saved

With these fixes, the artifact lifecycle should work correctly:
```
Request A: Upload artifact → saves to artifacts_db (same instance) + S3 (durable)
Request B: Read artifact (different instance) → artifacts_db empty → checks S3 → finds it ✅
Reset: Clears artifacts_db (this instance) + S3 (durable) ✅
```

## Next Steps If Tests Still Fail

1. Check CloudWatch logs for S3 errors or permission issues
2. Verify IAM role has `s3:*` permissions for the artifact bucket
3. Check if bucket is actually being created (might already exist and be inaccessible)
4. Look for eventual consistency issues with S3 (very unlikely but possible in high-concurrency scenarios)
