# Robust Distributional MCTS (RD-MCTS)

Implementation and evaluation of **Robust Distributional Monte Carlo Tree Search**, a novel MCTS variant that uses distributional value representations with penalized max-convolution backpropagation for risk-sensitive game playing.

## Overview

RD-MCTS extends traditional MCTS with two key innovations:

1. **Thompson Sampling** for node selection - efficiently explores high-variance branches without manual UCB tuning
2. **Penalized Max-Convolution** for backpropagation - propagates full value distributions using the distributional Bellman optimality equation with a variance penalty term

This approach enables the agent to act as a "Risk-Sensitive Minimax" player, avoiding volatility traps that standard MCTS fails to detect.

## Key Results

- **80% win rate** (16W-1L-3D) against standard scalar MCTS baseline
- Successfully identifies and avoids high-uncertainty "trap" positions
- Efficient O(K×M) computation using CDF trick for distribution operations
- Only ~14% computational overhead compared to baseline

## Project Structure

```
.
├── rd_mcts_experiment.py    # Core RD-MCTS implementation
├── baseline_mcts.py          # Standard scalar MCTS baseline
├── evaluation.py             # Comprehensive evaluation framework
├── test_implementation.py    # Unit and integration tests
├── visualize_results.py      # Results visualization
├── EVALUATION_REPORT.md      # Detailed analysis and findings
├── evaluation_results.json   # Raw numerical results
└── evaluation_results.png    # Visual comparison charts
```

## Installation

```bash
pip install torch numpy matplotlib
```

Note: If you encounter numpy compatibility issues with PyTorch, downgrade numpy:
```bash
pip install 'numpy<2'
```

## Usage

### Run Basic RD-MCTS Demo

```bash
python3 rd_mcts_experiment.py
```

This runs 100 MCTS simulations on an empty Connect 4 board and displays the estimated value and uncertainty for each move.

### Run Comprehensive Evaluation

```bash
python3 evaluation.py
```

This performs:
- Training of both RD-MCTS and baseline models via self-play
- Trap avoidance analysis on constructed positions
- Head-to-head match (20 games)
- Performance metrics and comparison

Expected runtime: ~3-5 minutes on CPU

### Run Tests

```bash
python3 test_implementation.py
```

Validates all components including:
- Connect 4 game logic
- Network architectures
- Distribution operations
- MCTS search algorithms
- Thompson Sampling exploration

### Generate Visualizations

```bash
python3 visualize_results.py
```

Creates comparison charts from evaluation results.

## Algorithm Details

### Distributional Representation

The value network outputs a categorical distribution over 51 atoms spanning [-1, 1]:

```
Z_θ(s) = Σ p_i(s) δ_{z_i}
```

### Selection: Thompson Sampling

At each node, sample a value from each child's distribution and select the child with the highest sampled value plus exploration bonus:

```
a* = argmax_a { ṽ_a + c_puct × P(a) × √N / (1 + N_a) }
where ṽ_a ~ Z(s, a)
```

### Backpropagation: Penalized Max-Convolution

**Step 1: Max-Convolution** (using CDF trick for efficiency)
```
D_raw(s) = Dist(max(X_1, X_2, ..., X_M)) where X_i ~ D_i
F_max(x) = ∏ F_i(x)
```

**Step 2: Variance Penalty**
```
μ_final = μ_raw - λ × σ_raw
```

The penalty parameter λ controls risk aversion (default: 1.0).

## Configuration

Key hyperparameters in `rd_mcts_experiment.py`:

```python
SUPPORT_SIZE = 51        # Number of atoms in distribution
V_MIN = -1.0            # Minimum value
V_MAX = 1.0             # Maximum value
LAMBDA_PENALTY = 1.0    # Risk aversion parameter
```

## Performance

### Training Convergence

Both algorithms converge over 5 training iterations with 10 self-play games each. RD-MCTS shows higher absolute loss due to distributional cross-entropy vs MSE, but this is expected and not directly comparable.

### Trap Avoidance

In constructed trap positions, RD-MCTS successfully penalizes high-variance moves:
- High-variance moves (σ > 0.3) receive lower scores after penalty
- Move selection correlates with penalized score (mean - λ×std) rather than raw mean

### Head-to-Head Performance

20-game match results:
- RD-MCTS: 16 wins (80%)
- Baseline: 1 win (5%)
- Draws: 3 (15%)

## Limitations

- Tested only on Connect 4 (relatively simple domain)
- Limited training (50 total self-play games)
- Fixed hyperparameters (λ not tuned)
- CPU-only implementation

## Future Work

- Adaptive penalty parameter λ(s) as function of game state
- Validation on larger domains (Go, Chess, Shogi)
- Opponent modeling with risk preferences
- GPU acceleration for larger-scale training
- Extension to continuous action spaces

## References

This implementation is based on the research paper "Robust Distributional MCTS via Penalized Max-Convolution (RD-MCTS)" which introduces the theoretical framework for distributional MCTS with variance penalties.

## Citation

If you use this implementation in your research, please cite:

```
@article{rdmcts2024,
  title={Robust Distributional MCTS via Penalized Max-Convolution},
  year={2024}
}
```

## License

This is a research implementation provided as-is for educational and research purposes.
