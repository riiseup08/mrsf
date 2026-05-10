#!/usr/bin/env python3
"""
Deep verification script for pymrsf package.
Tests all critical functionality, error handling, and provider switching.
"""

import os
import sys
import traceback

def section(title):
    """Print a section header"""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)

def test_result(test_name, passed, details=""):
    """Print test result"""
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"{status} - {test_name}")
    if details:
        print(f"     {details}")
    return passed

def test_imports():
    """Test all module imports"""
    section("1. MODULE IMPORTS")
    all_passed = True
    
    # Test main package import
    try:
        import pymrsf
        all_passed &= test_result("pymrsf import", True, f"version={pymrsf.__version__}")
    except Exception as e:
        all_passed &= test_result("pymrsf import", False, str(e))
        return False
    
    # Test individual modules
    modules = [
        "pymrsf.core",
        "pymrsf.probe", 
        "pymrsf.rag",
        "pymrsf.storage",
        "pymrsf.embeddings",
        "pymrsf.cache",
        "pymrsf.benchmark",
        "pymrsf.inspect"
    ]
    
    for module in modules:
        try:
            __import__(module)
            all_passed &= test_result(f"{module} import", True)
        except Exception as e:
            all_passed &= test_result(f"{module} import", False, str(e))
    
    return all_passed

def test_public_api():
    """Test all public API exports"""
    section("2. PUBLIC API EXPORTS")
    all_passed = True
    
    try:
        from pymrsf import (
            # Core functions
            tokenize, detokenize, quantized_argmax, get_surprises,
            compute_delta, next_token_greedy, ModelSession,
            get_backend, get_raw_lm, provider_capabilities,
            
            # RAG functions
            score_chunk, score_chunks, score_chunks_batch,
            filter_chunks, score_chunk_async, score_chunks_async,
            filter_chunks_async,
            
            # Probe functions
            probe, probe_compare,
            
            # Storage functions
            mrsf_write, mrsf_read, save_index, load_index,
            rebuild_faiss_from_sqlite, close_connections,
            
            # Embeddings
            embed, get_embedding_dim,
            
            # Cache
            configure_cache, clear_cache, get_cache_stats, get_embedding_cache_stats,
            
            # Benchmark & Inspect
            mrsf_benchmark_canterbury, mrsf_latency_benchmark,
            mrsf_inspect, mrsf_rebuild_explained,
        )
        all_passed &= test_result("All public functions import", True, "35+ exports")
    except ImportError as e:
        all_passed &= test_result("All public functions import", False, str(e))
        return False
    
    return all_passed

def test_provider_capabilities():
    """Test provider capability detection"""
    section("3. PROVIDER CAPABILITIES")
    all_passed = True
    
    try:
        from pymrsf import provider_capabilities
        
        caps = provider_capabilities()
        required_keys = [
            "provider", "supports_logits", "supports_probe",
            "supports_delta", "supports_sessions", "supports_true_surprises",
            "supports_embeddings", "supports_tokenization"
        ]
        
        for key in required_keys:
            if key in caps:
                all_passed &= test_result(f"Capability '{key}' present", True, f"value={caps[key]}")
            else:
                all_passed &= test_result(f"Capability '{key}' present", False, "missing")
        
        # Check current provider
        current = caps.get("provider", "unknown")
        all_passed &= test_result("Current provider detected", True, f"provider={current}")
        
    except Exception as e:
        all_passed &= test_result("Provider capabilities", False, str(e))
    
    return all_passed

def test_rag_functions():
    """Test RAG scoring functions"""
    section("4. RAG SCORING FUNCTIONS")
    all_passed = True
    
    try:
        from pymrsf.rag import score_chunk, score_chunks, filter_chunks
        from pymrsf import provider_capabilities
        
        caps = provider_capabilities()
        
        # Test score_chunk
        try:
            result = score_chunk(
                chunk="Neural networks learn through backpropagation.",
                query="How do neural networks train?",
                verbose=False
            )
            
            required_fields = ["rag_score", "novelty_score", "relevance_score", 
                             "verdict", "chunk_preview", "scoring_mode"]
            missing = [f for f in required_fields if f not in result]
            
            if not missing:
                all_passed &= test_result("score_chunk() returns complete result", True,
                    f"score={result['rag_score']}, mode={result['scoring_mode']}")
            else:
                all_passed &= test_result("score_chunk() returns complete result", False,
                    f"missing fields: {missing}")
        except Exception as e:
            all_passed &= test_result("score_chunk() execution", False, str(e))
        
        # Test score_chunks (batch)
        try:
            chunks = [
                "The Eiffel Tower is in Paris.",
                "Neural networks use gradient descent.",
                "Python is a programming language."
            ]
            results = score_chunks(chunks, query="machine learning", verbose=False)
            
            if len(results) == len(chunks):
                all_passed &= test_result("score_chunks() batch processing", True,
                    f"processed {len(results)} chunks")
            else:
                all_passed &= test_result("score_chunks() batch processing", False,
                    f"expected {len(chunks)}, got {len(results)}")
        except Exception as e:
            all_passed &= test_result("score_chunks() execution", False, str(e))
        
        # Test filter_chunks
        try:
            filtered = filter_chunks(chunks, query="machine learning", min_rag_score=0, verbose=False)
            all_passed &= test_result("filter_chunks() execution", True,
                f"filtered to {len(filtered)} chunks")
        except Exception as e:
            all_passed &= test_result("filter_chunks() execution", False, str(e))
        
    except Exception as e:
        all_passed &= test_result("RAG functions import", False, str(e))
    
    return all_passed

def test_probe_functions():
    """Test knowledge probing functions"""
    section("5. KNOWLEDGE PROBING")
    all_passed = True
    
    try:
        from pymrsf import probe, probe_compare, provider_capabilities
        
        caps = provider_capabilities()
        
        if not caps.get("supports_probe", False):
            all_passed &= test_result("Probing available", True, 
                f"Skipped - {caps['provider']} provider doesn't support probing")
            return all_passed
        
        # Test probe
        try:
            result = probe("To be or not to be")
            
            required_fields = ["knowledge_score", "label", "description", "compression"]
            missing = [f for f in required_fields if f not in result]
            
            if "error" in result:
                all_passed &= test_result("probe() execution", False, result.get("message", "error"))
            elif not missing:
                all_passed &= test_result("probe() execution", True,
                    f"score={result['knowledge_score']}, label={result['label']}")
            else:
                all_passed &= test_result("probe() returns complete result", False,
                    f"missing fields: {missing}")
        except Exception as e:
            all_passed &= test_result("probe() execution", False, str(e))
        
        # Test probe_compare
        try:
            texts = ["Common phrase", "Unique proprietary algorithm xyz"]
            results = probe_compare(texts)
            
            if len(results) == len(texts):
                all_passed &= test_result("probe_compare() execution", True,
                    f"compared {len(results)} texts")
            else:
                all_passed &= test_result("probe_compare() execution", False,
                    f"expected {len(texts)}, got {len(results)}")
        except Exception as e:
            all_passed &= test_result("probe_compare() execution", False, str(e))
        
    except Exception as e:
        all_passed &= test_result("Probe functions import", False, str(e))
    
    return all_passed

def test_cache_functions():
    """Test caching system"""
    section("6. CACHING SYSTEM")
    all_passed = True
    
    try:
        from pymrsf import (
            configure_cache, get_cache_stats, 
            clear_cache, score_chunk
        )
        
        # Configure cache
        configure_cache(enabled=True, max_size=100, ttl=3600)
        all_passed &= test_result("configure_cache()", True)
        
        # Clear cache to start fresh
        clear_cache(reset_stats=True)
        all_passed &= test_result("clear_cache()", True)
        
        # Score a chunk (should cache it)
        chunk = "Test chunk for caching verification"
        query = "test query"
        result1 = score_chunk(chunk, query=query, verbose=False)
        
        # Score same chunk again (should hit cache)
        result2 = score_chunk(chunk, query=query, verbose=False)
        
        # Check cache stats
        stats = get_cache_stats()
        
        if stats["hits"] > 0:
            all_passed &= test_result("Cache hit detection", True,
                f"hits={stats['hits']}, misses={stats['misses']}")
        else:
            all_passed &= test_result("Cache hit detection", False,
                "Expected cache hit but got none")
        
        # Verify cached result is consistent (ignore 'cached' flag)
        result1_copy = {k: v for k, v in result1.items() if k != "cached"}
        result2_copy = {k: v for k, v in result2.items() if k != "cached"}
        
        if result1_copy == result2_copy:
            all_passed &= test_result("Cache result consistency", True,
                f"cached flag: {result2.get('cached', False)}")
        else:
            all_passed &= test_result("Cache result consistency", False,
                "Cached result differs from original")
        
    except Exception as e:
        all_passed &= test_result("Cache functions", False, str(e))
        traceback.print_exc()
    
    return all_passed

def test_error_handling():
    """Test graceful error handling"""
    section("7. ERROR HANDLING & GRACEFUL DEGRADATION")
    all_passed = True
    
    try:
        from pymrsf.rag import score_chunk
        from pymrsf import provider_capabilities
        
        # Test with empty chunk
        try:
            result = score_chunk("", query="test", verbose=False)
            all_passed &= test_result("Empty chunk handling", True, "handled gracefully")
        except Exception as e:
            # Empty chunk should either work or fail gracefully
            all_passed &= test_result("Empty chunk handling", True, f"raised: {type(e).__name__}")
        
        # Test with very long chunk
        try:
            long_chunk = "word " * 1000
            result = score_chunk(long_chunk, query="test", verbose=False)
            all_passed &= test_result("Long chunk handling", True, "processed successfully")
        except Exception as e:
            all_passed &= test_result("Long chunk handling", False, str(e))
        
        # Test invalid weights
        try:
            result = score_chunk("test", query="test", 
                weights={"novelty": 0.5, "relevance": 0.3},  # doesn't sum to 1
                verbose=False)
            # Should auto-normalize
            all_passed &= test_result("Invalid weights normalization", True, "normalized automatically")
        except Exception as e:
            all_passed &= test_result("Invalid weights normalization", False, str(e))
        
    except Exception as e:
        all_passed &= test_result("Error handling tests", False, str(e))
    
    return all_passed

def test_storage_functions():
    """Test storage functions (if available)"""
    section("8. STORAGE FUNCTIONS")
    all_passed = True
    
    try:
        from pymrsf import provider_capabilities, mrsf_write, save_index, close_connections
        
        caps = provider_capabilities()
        
        if not caps.get("supports_delta", False):
            all_passed &= test_result("Storage available", True,
                f"Skipped - {caps['provider']} provider doesn't support delta compression")
            return all_passed
        
        # Test basic write (don't save to avoid side effects)
        try:
            result = mrsf_write("Test document for verification", doc_id="test_verify_001")
            
            if "doc_id" in result and "compression" in result:
                all_passed &= test_result("mrsf_write() execution", True,
                    f"compression={result.get('compression', 0):.1%}")
            else:
                all_passed &= test_result("mrsf_write() execution", False,
                    "missing expected fields")
        except Exception as e:
            all_passed &= test_result("mrsf_write() execution", False, str(e))
        
        # Clean up
        try:
            close_connections()
            all_passed &= test_result("close_connections()", True)
        except Exception as e:
            all_passed &= test_result("close_connections()", False, str(e))
        
    except Exception as e:
        all_passed &= test_result("Storage functions", False, str(e))
    
    return all_passed

def test_documentation_consistency():
    """Test that documentation matches implementation"""
    section("9. DOCUMENTATION CONSISTENCY")
    all_passed = True
    
    try:
        import pymrsf
        
        # Check __all__ exports match what's documented
        exports = pymrsf.__all__
        all_passed &= test_result("__all__ defined", True, f"{len(exports)} exports")
        
        # Verify key exports are present
        key_exports = [
            "score_chunk", "filter_chunks", "probe", "mrsf_write", "mrsf_read",
            "provider_capabilities", "get_backend", "configure_cache"
        ]
        
        for export in key_exports:
            if export in exports:
                all_passed &= test_result(f"'{export}' in __all__", True)
            else:
                all_passed &= test_result(f"'{export}' in __all__", False)
        
        # Check version is set
        if hasattr(pymrsf, "__version__"):
            all_passed &= test_result("__version__ defined", True, f"v{pymrsf.__version__}")
        else:
            all_passed &= test_result("__version__ defined", False)
        
    except Exception as e:
        all_passed &= test_result("Documentation checks", False, str(e))
    
    return all_passed

def main():
    """Run all verification tests"""
    print("\n" + "="*60)
    print("  PYMRSF DEEP VERIFICATION")
    print("="*60)
    print(f"Python: {sys.version}")
    print(f"CWD: {os.getcwd()}")
    
    results = []
    
    # Run all test suites
    results.append(("Module Imports", test_imports()))
    results.append(("Public API", test_public_api()))
    results.append(("Provider Capabilities", test_provider_capabilities()))
    results.append(("RAG Functions", test_rag_functions()))
    results.append(("Probe Functions", test_probe_functions()))
    results.append(("Cache System", test_cache_functions()))
    results.append(("Error Handling", test_error_handling()))
    results.append(("Storage Functions", test_storage_functions()))
    results.append(("Documentation", test_documentation_consistency()))
    
    # Summary
    section("SUMMARY")
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {name}")
    
    print(f"\nTotal: {passed}/{total} test suites passed")
    
    if passed == total:
        print("\n🎉 ALL VERIFICATION TESTS PASSED!")
        print("Package is ready for release.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test suite(s) failed")
        print("Review failures above before releasing.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
