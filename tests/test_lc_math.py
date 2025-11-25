"""
Unit tests for LC-MCTS mathematical invariants.
Tests the hybrid merge algorithm before integrating into Connect4.
"""

import numpy as np
from scipy.stats import norm
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def hybrid_merge(mu1, w1, nu1, mu2, w2, nu2):
    """
    Hybrid merge algorithm from LC-MCTS paper.
    
    Args:
        mu1, mu2: Expected values (scalars)
        w1, w2: Structured uncertainty embeddings (d-vectors)
        nu1, nu2: Unstructured uncertainty (scalars)
    
    Returns:
        mu_p, w_p, nu_p: Parent node state
    """
    # Step A: Compute total variances
    s1_sq = np.dot(w1, w1) + nu1**2
    s2_sq = np.dot(w2, w2) + nu2**2
    
    # Step B: Compute difference statistics (only structured part contributes to covariance!)
    cov = np.dot(w1, w2)
    var_diff = s1_sq + s2_sq - 2*cov
    sigma_diff = np.sqrt(max(var_diff, 1e-9))  # Numerical stability
    
    # Handle clone case explicitly (when nodes are identical or nearly identical)
    # Check if both mean and variance difference are negligible
    mu_diff = abs(mu1 - mu2)
    w_diff_norm = np.linalg.norm(w1 - w2)
    nu_diff = abs(nu1 - nu2)
    
    if mu_diff < 1e-9 and w_diff_norm < 1e-9 and nu_diff < 1e-9:
        # Clone case: return identical node
        return mu1, w1.copy(), nu1
    
    # Step C: Compute probabilities
    alpha = (mu1 - mu2) / sigma_diff
    Phi = norm.cdf(alpha)
    phi = norm.pdf(alpha)
    
    # Step D: Update mean (optimistic)
    mu_p = mu1*Phi + mu2*(1-Phi) + sigma_diff*phi
    
    # Step E: Update structured uncertainty (Stein's Lemma, NO RESCALING)
    w_p = Phi * w1 + (1 - Phi) * w2
    
    # Step F: Compute target variance (Clark's Formula)
    E_P_sq = (mu1**2 + s1_sq)*Phi + (mu2**2 + s2_sq)*(1-Phi) + (mu1 + mu2)*sigma_diff*phi
    target_var = max(E_P_sq - mu_p**2, 1e-9)  # Numerical stability
    
    # Step G: Compute unstructured residual
    w_p_var = np.dot(w_p, w_p)
    nu_p = np.sqrt(max(target_var - w_p_var, 0))
    
    return mu_p, w_p, nu_p


def test_clone_case():
    """Test: If A = B (identical clones), there should be no value inflation."""
    print("\n" + "="*80)
    print("TEST 1: Clone Case (A = B)")
    print("="*80)
    
    d = 32
    mu = 0.5
    w = np.random.randn(d)
    nu = 0.1
    
    mu_p, w_p, nu_p = hybrid_merge(mu, w, nu, mu, w, nu)
    
    # Check: parent should equal child
    mu_error = abs(mu_p - mu)
    w_error = np.linalg.norm(w_p - w)
    nu_error = abs(nu_p - nu)
    
    print(f"Input:  μ={mu:.4f}, ||w||={np.linalg.norm(w):.4f}, ν={nu:.4f}")
    print(f"Output: μ={mu_p:.4f}, ||w||={np.linalg.norm(w_p):.4f}, ν={nu_p:.4f}")
    print(f"Errors: Δμ={mu_error:.6f}, Δw={w_error:.6f}, Δν={nu_error:.6f}")
    
    assert mu_error < 1e-6, f"Clone case failed: μ changed by {mu_error}"
    assert w_error < 1e-6, f"Clone case failed: w changed by {w_error}"
    assert nu_error < 1e-6, f"Clone case failed: ν changed by {nu_error}"
    
    print("✓ PASSED: Clone case preserves state exactly")
    return True


def test_variance_accuracy():
    """Test: Total variance should match Clark's formula."""
    print("\n" + "="*80)
    print("TEST 2: Variance Accuracy")
    print("="*80)
    
    d = 32
    mu1, mu2 = 0.6, 0.4
    w1 = np.random.randn(d)
    w2 = np.random.randn(d)
    nu1, nu2 = 0.1, 0.15
    
    mu_p, w_p, nu_p = hybrid_merge(mu1, w1, nu1, mu2, w2, nu2)
    
    # Compute expected variance using Clark's formula directly
    s1_sq = np.dot(w1, w1) + nu1**2
    s2_sq = np.dot(w2, w2) + nu2**2
    cov = np.dot(w1, w2)
    var_diff = s1_sq + s2_sq - 2*cov
    sigma_diff = np.sqrt(max(var_diff, 1e-9))
    alpha = (mu1 - mu2) / sigma_diff
    Phi = norm.cdf(alpha)
    phi = norm.pdf(alpha)
    
    E_P_sq = (mu1**2 + s1_sq)*Phi + (mu2**2 + s2_sq)*(1-Phi) + (mu1 + mu2)*sigma_diff*phi
    expected_var = E_P_sq - mu_p**2
    
    # Compute actual variance from output
    actual_var = np.dot(w_p, w_p) + nu_p**2
    
    var_error = abs(actual_var - expected_var) / max(expected_var, 1e-9)
    
    print(f"Expected variance (Clark): {expected_var:.6f}")
    print(f"Actual variance (||w_p||² + ν_p²): {actual_var:.6f}")
    print(f"Relative error: {var_error:.6f} ({var_error*100:.2f}%)")
    
    assert var_error < 0.01, f"Variance accuracy failed: {var_error*100:.2f}% error"
    
    print("✓ PASSED: Variance matches Clark's formula within 1%")
    return True


def test_correlation_preservation():
    """Test: Correlation with external node should be preserved."""
    print("\n" + "="*80)
    print("TEST 3: Correlation Preservation")
    print("="*80)
    
    d = 32
    mu1, mu2 = 0.6, 0.4
    w1 = np.random.randn(d)
    w2 = np.random.randn(d)
    nu1, nu2 = 0.1, 0.15
    
    # External node C
    w_c = np.random.randn(d)
    
    mu_p, w_p, nu_p = hybrid_merge(mu1, w1, nu1, mu2, w2, nu2)
    
    # Compute expected covariance using Stein's formula
    alpha = (mu1 - mu2) / np.sqrt(max(np.dot(w1-w2, w1-w2), 1e-9))
    Phi = norm.cdf(alpha)
    
    expected_cov = Phi * np.dot(w1, w_c) + (1-Phi) * np.dot(w2, w_c)
    
    # Actual covariance from output
    actual_cov = np.dot(w_p, w_c)
    
    cov_error = abs(actual_cov - expected_cov) / max(abs(expected_cov), 1e-9)
    
    print(f"Expected Cov(P, C): {expected_cov:.6f}")
    print(f"Actual Cov(P, C): {actual_cov:.6f}")
    print(f"Relative error: {cov_error:.6f} ({cov_error*100:.2f}%)")
    
    assert cov_error < 0.01, f"Correlation preservation failed: {cov_error*100:.2f}% error"
    
    print("✓ PASSED: Correlation preserved within 1%")
    return True


def test_convergence_to_minimax():
    """Test: As uncertainty → 0, algorithm should converge to deterministic minimax."""
    print("\n" + "="*80)
    print("TEST 4: Convergence to Minimax")
    print("="*80)
    
    d = 32
    mu1, mu2 = 0.7, 0.3  # mu1 > mu2
    
    # Test with decreasing uncertainty
    uncertainties = [1.0, 0.1, 0.01, 0.001, 0.0001]
    
    print(f"Testing with μ1={mu1}, μ2={mu2} (μ1 > μ2)")
    print(f"{'Uncertainty':<15} {'μ_p':<10} {'Error from μ1':<15}")
    print("-" * 40)
    
    for scale in uncertainties:
        w1 = np.random.randn(d) * scale
        w2 = np.random.randn(d) * scale
        nu1 = 0.1 * scale
        nu2 = 0.1 * scale
        
        mu_p, w_p, nu_p = hybrid_merge(mu1, w1, nu1, mu2, w2, nu2)
        error = abs(mu_p - mu1)
        
        print(f"{scale:<15.4f} {mu_p:<10.6f} {error:<15.6f}")
    
    # Final test: with very low uncertainty, should be very close to mu1
    w1 = np.random.randn(d) * 0.0001
    w2 = np.random.randn(d) * 0.0001
    nu1 = 0.0001
    nu2 = 0.0001
    
    mu_p, w_p, nu_p = hybrid_merge(mu1, w1, nu1, mu2, w2, nu2)
    error = abs(mu_p - mu1)
    
    assert error < 0.01, f"Convergence failed: μ_p={mu_p:.6f} should be close to μ1={mu1}"
    
    print("✓ PASSED: Converges to deterministic minimax as uncertainty → 0")
    return True


def test_nu_growth_pattern():
    """Test: As we go up the tree, ||w||² should shrink and ν² should grow."""
    print("\n" + "="*80)
    print("TEST 5: ν Growth Pattern")
    print("="*80)
    
    d = 32
    
    # Start with 4 leaf nodes
    leaves = []
    for _ in range(4):
        mu = np.random.randn() * 0.5
        w = np.random.randn(d)
        nu = 0.0  # Leaves have ν=0
        leaves.append((mu, w, nu))
    
    print(f"{'Depth':<10} {'||w||²':<15} {'ν²':<15} {'w fraction':<15}")
    print("-" * 55)
    
    # Depth 0 (leaves)
    avg_w_sq = np.mean([np.dot(w, w) for _, w, _ in leaves])
    avg_nu_sq = np.mean([nu**2 for _, _, nu in leaves])
    w_frac = avg_w_sq / (avg_w_sq + avg_nu_sq + 1e-9)
    print(f"{'0 (leaf)':<10} {avg_w_sq:<15.4f} {avg_nu_sq:<15.4f} {w_frac:<15.2%}")
    
    # Merge up the tree
    current_level = leaves
    depth = 0
    
    while len(current_level) > 1:
        depth += 2
        next_level = []
        
        for i in range(0, len(current_level), 2):
            if i + 1 < len(current_level):
                mu1, w1, nu1 = current_level[i]
                mu2, w2, nu2 = current_level[i+1]
                mu_p, w_p, nu_p = hybrid_merge(mu1, w1, nu1, mu2, w2, nu2)
                next_level.append((mu_p, w_p, nu_p))
            else:
                next_level.append(current_level[i])
        
        # Compute statistics
        avg_w_sq = np.mean([np.dot(w, w) for _, w, _ in next_level])
        avg_nu_sq = np.mean([nu**2 for _, _, nu in next_level])
        w_frac = avg_w_sq / (avg_w_sq + avg_nu_sq + 1e-9)
        print(f"{depth:<10} {avg_w_sq:<15.4f} {avg_nu_sq:<15.4f} {w_frac:<15.2%}")
        
        current_level = next_level
    
    # Check that w fraction decreased significantly from 100% at leaves
    final_w_sq = np.dot(current_level[0][1], current_level[0][1])
    final_nu_sq = current_level[0][2]**2
    final_w_frac = final_w_sq / (final_w_sq + final_nu_sq + 1e-9)
    
    print(f"\nFinal w fraction: {final_w_frac:.2%}")
    # At depth 4 with 4 leaves, we expect w fraction to be around 50-60%
    # (paper shows 11% at depth 6 with 64 leaves, so 50-60% at depth 4 is reasonable)
    assert final_w_frac < 0.70, f"ν growth pattern failed: w fraction should decrease from 100%"
    
    print("✓ PASSED: w shrinks and ν grows as expected")
    return True


def test_perfect_correlation():
    """Test: Perfectly correlated children (w1 = w2) should give no bonus."""
    print("\n" + "="*80)
    print("TEST 6: Perfect Correlation (Queen Sacrifice Scenario)")
    print("="*80)
    
    d = 32
    mu = 0.8
    w = np.random.randn(d)  # Same w for all children
    nu = 0.0
    
    # Merge 4 perfectly correlated children
    print(f"Merging 4 children with μ={mu}, identical w vectors")
    
    # First merge
    mu_p1, w_p1, nu_p1 = hybrid_merge(mu, w, nu, mu, w, nu)
    print(f"After 1st merge: μ={mu_p1:.4f}")
    
    # Second merge
    mu_p2, w_p2, nu_p2 = hybrid_merge(mu_p1, w_p1, nu_p1, mu_p1, w_p1, nu_p1)
    print(f"After 2nd merge: μ={mu_p2:.4f}")
    
    # Should be very close to original μ (no bonus from correlation)
    inflation = mu_p2 - mu
    print(f"Value inflation: {inflation:.6f}")
    
    assert abs(inflation) < 0.01, f"Perfect correlation failed: got {inflation:.6f} inflation"
    
    print("✓ PASSED: Perfectly correlated children give no bonus")
    return True


def run_all_tests():
    """Run all LC-MCTS math tests."""
    print("="*80)
    print("LC-MCTS MATHEMATICAL INVARIANTS TEST SUITE")
    print("="*80)
    
    tests = [
        test_clone_case,
        test_variance_accuracy,
        test_correlation_preservation,
        test_convergence_to_minimax,
        test_nu_growth_pattern,
        test_perfect_correlation,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"✗ FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ ERROR: {e}")
            failed += 1
    
    print("\n" + "="*80)
    print(f"TEST RESULTS: {passed} passed, {failed} failed")
    print("="*80)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
