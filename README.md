# Sophisticated-Electronic-Trading-Surveillance-Dataset
# Sophisticated Electronic Trading Surveillance Dataset

## Overview

This dataset simulates a realistic electronic trading environment designed for:

- Trade Surveillance model development  
- Market abuse (spoofing / layering) detection  
- Threshold calibration exercises  
- False Positive / False Negative trade-off analysis  
- ML-based behavioural classification  
- Model governance and audit demonstration  

It is engineered to withstand scrutiny from:

- Head of Surveillance  
- Model Risk Management (MRM)  
- Internal Audit  
- Regulatory review environments  

Unlike simplistic synthetic datasets, this version intentionally incorporates **grey behaviour and statistical overlap** to create realistic calibration challenges.

---

# 1. Market Microstructure Design

The dataset simulates a Level 1 (L1) electronic market structure including:

- `best_bid`
- `best_offer`
- `mid_price`
- Dynamic bid-ask spread

### Key Properties

- Prices follow a stochastic random walk with volatility clustering.
- The bid-ask spread varies dynamically.
- No deterministic price jumps.
- Large visible liquidity may exert probabilistic short-lived price pressure.
- Price impact decays following spoof cancellation.

This ensures the data resembles a realistic electronic trading environment rather than a static simulation.

---

# 2. Trader Archetypes

The dataset includes three behavioural personas.

## Type A – Retail / Noise Traders

- Low frequency  
- Gaussian-distributed order sizes  
- Long order lifetimes (Gamma-distributed, mean approx. 8–15s)  
- Cancel ratio: ~20–40%  
- Occasional short-lived orders to introduce noise  

These traders represent background market flow.

---

## Type B – Institutional / Market Makers

- High frequency  
- Small order sizes (fat-tailed distribution)  
- Cancel ratio: ~70–90%  
- Order lifetime overlapping with spoofers  
- Balanced buy/sell activity  
- Liquidity provision near the touch  

These traders create realistic grey behaviour, increasing model complexity and reducing trivial separability.

---

## Type C – Spoofers (Target Behaviour)

Spoofers are intentionally **not trivially separable**.

### Behavioural Characteristics

**Size Asymmetry**
- Spoof orders are significantly larger (10x–50x median market size).
- Executed “flip” trades remain small.
- Some normal traders occasionally submit large trades to introduce ambiguity.

**Directional Layering**
- 3–5 large same-side orders placed progressively further from touch.
- Small opposite-side execution.
- Rapid cancellation of layered spoof orders.

**Burst Clustering**
- Spoofing occurs in stochastic clusters.
- Not evenly distributed over time.
- Periods of inactivity between bursts.

**Distribution Overlap**
- No hard thresholds on lifetime or size.
- Significant statistical overlap between spoofers and market makers.

Spoofing activity represents approximately **2–4% of total rows**.

---

# 3. Statistical Realism

## Order Size

- Base market volume follows a Power Law / Pareto distribution.
- Fat tails simulate realistic electronic market volume dynamics.

## Volatility Regimes

The dataset includes both low and high volatility regimes.

- Volatility clustering is present.
- Spoofing does not exclusively occur during high volatility.
- No artificial correlation between volatility spikes and abuse events.

## Logical Consistency

- `cancel_flag` and `execution_flag` are mutually consistent.
- A small percentage (2–5%) of open orders may exist.
- Order lifecycles are traceable via `order_id`.

---

# 4. Schema

| Column | Description |
|--------|------------|
| timestamp | Microsecond precision event time |
| session_id | Trading session identifier |
| order_id | Unique order identifier |
| trader_id | Trader identifier |
| trader_type | A / B / C |
| side | buy / sell |
| order_size | Order volume |
| price | Order price |
| best_bid | Best bid at event time |
| best_offer | Best offer at event time |
| spread | best_offer - best_bid |
| cancel_flag | Whether order was cancelled |
| execution_flag | Whether order was executed |
| order_lifetime | Duration before terminal state |
| distance_from_touch | Price distance from BBO |
| volatility_regime | Low / High regime label (if included) |

---

# 5. Surveillance Use Cases

This dataset supports:

## Rule-Based Alerting

- Cancel ratio thresholds  
- Order velocity detection  
- Size imbalance detection  
- Layering pattern detection  
- Directional clustering detection  

## Calibration Exercises

- Threshold optimisation  
- Alert volume control  
- False Positive / False Negative trade-offs  
- Sensitivity analysis  

## ML-Based Modelling

- Logistic Regression  
- Random Forest  
- Gradient Boosting  
- Behavioural clustering  

## Model Governance

- Feature stability review  
- Distribution drift analysis  
- Stress testing under volatility shifts  
- Audit trail reconstruction  

---

# 6. Design Philosophy

This dataset intentionally avoids:

- Perfect separability  
- Hard thresholds  
- Artificially clean patterns  
- Deterministic price reactions  

The objective is to simulate the **ambiguity and grey behaviour** present in real electronic markets.

A surveillance model built on this dataset must demonstrate:

- Robust precision-recall trade-offs  
- Calibration sensitivity  
- Behavioural interpretability  
- Governance readiness  

---

# 7. Intended Audience

This dataset is suitable for:

- Surveillance Analysts  
- Quant Developers  
- Compliance Technology teams  
- Model Risk Management  
- Academic research in market abuse detection  

---

# 8. Limitations

While statistically realistic, this remains a synthetic dataset and does not replicate:

- Full Level 2 order book depth  
- Hidden liquidity  
- Cross-venue arbitrage  
- Real-world latency infrastructure  
- Cross-asset behaviour  

It is intended for controlled surveillance model experimentation and demonstration, not production deployment.

---

# Final Note

This dataset was engineered to move beyond textbook spoofing detection and toward a realistic calibration challenge reflective of a modern CIB electronic trading environment.

It is designed to test not only detection capability, but also governance robustness and decision-making under behavioural ambiguity.
