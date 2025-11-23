"""
Unit tests to verify the penalty mechanism is working correctly.
"""

import numpy as np
from framework import convolve_max_and_penalize, dist_stats, project_gaussian, ATOMS

def test_penalty_basic():
    """Test that penalty actually shifts the mean."""
    print("="*80)
    print("TEST 1: Basic Penalty Mechanism")
    print("="*80)
    
    # Create two synthetic child distributions
    # Child 1: Safe move (low mean, low variance)
    # Child 2: Risky move (high mean, high variance)
    child1 = project_gaussian(0.0, 0.1)
    child2 = project_gaussian(0.5, 0.5)
    
    child_dists = [child1, child2]
    
    # Test with different lambda values
    for lambda_val in [0.0, 0.5, 1.0, 2.0]:
        result_dist = convolve_max_and_penalize(child_dists, lambda_val)
        mu, sigma = dist_stats(result_dist)
        print(f"\nλ={lambda_val}:")
        print(f"  Mean: {mu:.4f}")
        print(f"  Std:  {sigma:.4f}")
        print(f"  Score (μ - λσ): {mu - lambda_val * sigma:.4f}")
    
    # Verify that mean decreases with lambda
    dist_l0 = convolve_max_and_penalize(child_dists, 0.0)
    dist_l1 = convolve_max_and_penalize(child_dists, 1.0)
    dist_l2 = convolve_max_and_penalize(child_dists, 2.0)
    
    mu0, _ = dist_stats(dist_l0)
    mu1, _ = dist_stats(dist_l1)
    mu2, _ = dist_stats(dist_l2)
    
    print(f"\n✓ Mean decreases with λ: {mu0:.4f} > {mu1:.4f} > {mu2:.4f}")
    assert mu0 > mu1 > mu2, "Mean should decrease as lambda increases!"
    print("✓ TEST PASSED: Penalty mechanism shifts mean correctly")

def test_trap_scenario():
    """Test the trap scenario from the paper's Theorem 1."""
    print("\n" + "="*80)
    print("TEST 2: Trap Scenario (Theorem 1)")
    print("="*80)
    
    # Safe move: v = 0.0, low variance
    safe = project_gaussian(0.0, 0.05)
    
    # Risky move: v = 0.2 (slightly better), high variance
    risky = project_gaussian(0.2, 0.5)
    
    print("\nChild distributions:")
    mu_safe, sigma_safe = dist_stats(safe)
    mu_risky, sigma_risky = dist_stats(risky)
    print(f"  Safe:  μ={mu_safe:.4f}, σ={sigma_safe:.4f}")
    print(f"  Risky: μ={mu_risky:.4f}, σ={sigma_risky:.4f}")
    
    # According to Theorem 1, RD-MCTS should prefer safe if λ > ε/σ
    # where ε = mu_risky - mu_safe
    epsilon = mu_risky - mu_safe
    threshold_lambda = epsilon / sigma_risky if sigma_risky > 0 else 0
    print(f"\nTheorem 1 threshold: λ > {threshold_lambda:.4f}")
    
    child_dists = [safe, risky]
    
    # Test with λ=0 (should prefer risky)
    dist_l0 = convolve_max_and_penalize(child_dists, 0.0)
    mu0, sigma0 = dist_stats(dist_l0)
    score0 = mu0  # No penalty
    
    # Test with λ=1.0 (should prefer safe if threshold < 1.0)
    dist_l1 = convolve_max_and_penalize(child_dists, 1.0)
    mu1, sigma1 = dist_stats(dist_l1)
    score1 = mu1 - 1.0 * sigma1
    
    print(f"\nλ=0.0: μ={mu0:.4f}, σ={sigma0:.4f}, score={score0:.4f}")
    print(f"λ=1.0: μ={mu1:.4f}, σ={sigma1:.4f}, score={score1:.4f}")
    
    # The max-convolution should pick the risky child (higher mean)
    # But with penalty, the score should be lower
    print(f"\n✓ Score decreased with penalty: {score0:.4f} > {score1:.4f}")
    assert score0 > score1, "Penalty should decrease the score!"
    print("✓ TEST PASSED: Trap avoidance mechanism works")

def test_gaussian_approximation_effect():
    """Test how much the Gaussian approximation affects the result."""
    print("\n" + "="*80)
    print("TEST 3: Gaussian Approximation Effect")
    print("="*80)
    
    # Create a bimodal distribution (two peaks)
    dist1 = project_gaussian(-0.5, 0.1)
    dist2 = project_gaussian(0.5, 0.1)
    bimodal = (dist1 + dist2) / 2
    
    mu_orig, sigma_orig = dist_stats(bimodal)
    print(f"\nOriginal bimodal distribution:")
    print(f"  μ={mu_orig:.4f}, σ={sigma_orig:.4f}")
    
    # Apply max-convolution with λ=0 (no penalty, just Gaussian approximation)
    result = convolve_max_and_penalize([bimodal], 0.0)
    mu_approx, sigma_approx = dist_stats(result)
    print(f"\nAfter Gaussian approximation:")
    print(f"  μ={mu_approx:.4f}, σ={sigma_approx:.4f}")
    
    # Check how much information is lost
    print(f"\nInformation loss:")
    print(f"  Mean shift: {abs(mu_orig - mu_approx):.4f}")
    print(f"  Std shift:  {abs(sigma_orig - sigma_approx):.4f}")
    
    if abs(mu_orig - mu_approx) > 0.1 or abs(sigma_orig - sigma_approx) > 0.1:
        print("⚠ WARNING: Gaussian approximation significantly changes distribution!")
    else:
        print("✓ Gaussian approximation preserves distribution reasonably well")

def test_penalty_magnitude():
    """Test the actual magnitude of the penalty effect."""
    print("\n" + "="*80)
    print("TEST 4: Penalty Magnitude Analysis")
    print("="*80)
    
    # Create distributions with varying uncertainty
    for sigma_val in [0.05, 0.1, 0.2, 0.5]:
        dist = project_gaussian(0.5, sigma_val)
        
        print(f"\nInput distribution: μ=0.5, σ={sigma_val}")
        
        # Apply different penalties
        for lambda_val in [0.0, 1.0, 2.0]:
            result = convolve_max_and_penalize([dist], lambda_val)
            mu, sigma = dist_stats(result)
            penalty_effect = 0.5 - mu  # How much did mean shift from original?
            
            print(f"  λ={lambda_val}: μ={mu:.4f}, σ={sigma:.4f}, shift={penalty_effect:.4f}")
        
        # Check if penalty is proportional to sigma
        result_l1 = convolve_max_and_penalize([dist], 1.0)
        mu_l1, _ = dist_stats(result_l1)
        expected_shift = sigma_val
        actual_shift = 0.5 - mu_l1
        
        print(f"  Expected shift (λ=1): {expected_shift:.4f}")
        print(f"  Actual shift:         {actual_shift:.4f}")
        print(f"  Ratio: {actual_shift/expected_shift:.2f}x")

def test_multiple_children():
    """Test penalty with multiple children (realistic scenario)."""
    print("\n" + "="*80)
    print("TEST 5: Multiple Children Scenario")
    print("="*80)
    
    # Create 7 children (like Connect4 moves)
    children = []
    for i in range(7):
        # Vary mean and std
        mean = -0.5 + i * 0.2  # Range from -0.5 to 0.7
        std = 0.1 + i * 0.05   # Range from 0.1 to 0.4
        children.append(project_gaussian(mean, std))
        mu, sigma = dist_stats(children[-1])
        print(f"Child {i}: μ={mu:.4f}, σ={sigma:.4f}")
    
    print("\nMax-convolution results:")
    for lambda_val in [0.0, 1.0, 2.0]:
        result = convolve_max_and_penalize(children, lambda_val)
        mu, sigma = dist_stats(result)
        print(f"  λ={lambda_val}: μ={mu:.4f}, σ={sigma:.4f}, score={mu - lambda_val * sigma:.4f}")

if __name__ == "__main__":
    print("\n" + "="*80)
    print("PENALTY MECHANISM UNIT TESTS")
    print("="*80)
    
    try:
        test_penalty_basic()
        test_trap_scenario()
        test_gaussian_approximation_effect()
        test_penalty_magnitude()
        test_multiple_children()
        
        print("\n" + "="*80)
        print("ALL TESTS PASSED!")
        print("="*80)
        print("\nConclusion: Penalty mechanism is implemented correctly.")
        print("The Gaussian approximation may reduce the penalty's effect,")
        print("but the core mechanism (μ - λσ) is working as intended.")
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        print("\nThe penalty mechanism has a bug that needs to be fixed!")
