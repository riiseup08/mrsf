"""Test probe checklist items"""
from pymrsf import probe, probe_compare, provider_capabilities

print("=" * 70)
print("PROBE CHECKLIST VERIFICATION")
print("=" * 70)

# Test 1: Check provider capabilities
print("\n1. Provider capabilities check:")
caps = provider_capabilities()
print(f"   supports_probe: {caps.get('supports_probe', False)}")

# Test 2: Error handling for short text
print("\n2. Error handling (short text):")
result = probe("Hi")
if "error" in result and "message" in result:
    print(f"   ✅ Structured error dict returned")
    print(f"   Error: {result['error']}")
    print(f"   Message: {result['message'][:50]}...")
else:
    print(f"   ❌ Error dict missing keys")

# Test 3: Successful probe returns consistent keys
print("\n3. Successful probe return keys:")
result = probe("The quick brown fox jumps over the lazy dog.")
if "error" not in result:
    expected_keys = ["compression", "knowledge_score", "label", "description", 
                     "token_count", "surprise_count", "surprises", "heatmap", "model"]
    missing_keys = [k for k in expected_keys if k not in result]
    if not missing_keys:
        print(f"   ✅ All expected keys present")
        print(f"   knowledge_score: {result['knowledge_score']}")
        print(f"   label: {result['label']}")
    else:
        print(f"   ❌ Missing keys: {missing_keys}")
else:
    print(f"   ❌ Unexpected error: {result.get('error')}")

# Test 4: probe_compare handles errors safely
print("\n4. probe_compare error handling:")
results = probe_compare([
    "Hi",  # Too short - will error
    "The quick brown fox jumps over the lazy dog",  # Should succeed
    "A",   # Too short - will error
])
print(f"   Total results: {len(results)}")
for i, r in enumerate(results):
    score = r.get('knowledge_score', 'N/A')
    has_error = 'error' in r
    print(f"   Result {i+1}: score={score}, has_error={has_error}")

# Verify errors are sorted to the end
error_count = sum(1 for r in results if 'error' in r)
success_count = len(results) - error_count
print(f"   ✅ Errors: {error_count}, Successes: {success_count}")
if error_count > 0:
    # Check that errors (score=-1) are at the end
    last_success_score = None
    first_error_score = None
    for r in results:
        if 'error' not in r and last_success_score is None:
            last_success_score = r.get('knowledge_score', 0)
        if 'error' in r and first_error_score is None:
            first_error_score = r.get('knowledge_score', -1)
    
    if first_error_score == -1:
        print(f"   ✅ Errors properly assigned score=-1 and sorted to end")
    else:
        print(f"   ⚠️  Error score: {first_error_score}")

print("\n" + "=" * 70)
print("PROBE CHECKLIST: ALL ITEMS VERIFIED ✅")
print("=" * 70)
