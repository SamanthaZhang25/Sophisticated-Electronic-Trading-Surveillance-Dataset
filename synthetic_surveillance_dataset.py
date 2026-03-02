import numpy as np
import pandas as pd

# =====================================
# Configuration
# =====================================
SEED = 20260302
N_ROWS = 100_000
N_TRADERS = 96
TICK_SIZE = 0.01
BASE_PRICE = 100.0
INSTRUMENT = "EURUSD"
OUTPUT_PATH = "sophisticated_trading_dataset.csv"

rng = np.random.default_rng(SEED)


# =====================================
# Trader population and personas
# =====================================
trader_ids = np.arange(1, N_TRADERS + 1)

# Persona mix with overlap-friendly design
n_type_c = 3  # ~3.1% of traders
n_type_b = 40
n_type_a = N_TRADERS - n_type_b - n_type_c

perm = rng.permutation(trader_ids)
type_c_traders = perm[:n_type_c]
type_b_traders = perm[n_type_c : n_type_c + n_type_b]
type_a_traders = perm[n_type_c + n_type_b :]

trader_type_map = {}
for t in type_a_traders:
    trader_type_map[int(t)] = "A"
for t in type_b_traders:
    trader_type_map[int(t)] = "B"
for t in type_c_traders:
    trader_type_map[int(t)] = "C"

# Activity intensities (Poisson-like heterogeneity)
intensity = pd.Series(0.0, index=trader_ids, dtype=float)
intensity.loc[type_a_traders] = rng.lognormal(mean=-0.85, sigma=0.45, size=n_type_a)  # lower frequency
intensity.loc[type_b_traders] = rng.lognormal(mean=0.65, sigma=0.35, size=n_type_b)   # higher frequency
intensity.loc[type_c_traders] = rng.lognormal(mean=0.15, sigma=0.40, size=n_type_c)    # overlaps with A/B
intensity = (intensity / intensity.sum()).values

trader_id = rng.choice(trader_ids, size=N_ROWS, p=intensity)
trader_type = np.array([trader_type_map[int(t)] for t in trader_id], dtype=object)


# =====================================
# Volatility regimes + event-time process
# =====================================
# Markov regime: low vol and high vol bursts
regime = np.zeros(N_ROWS, dtype=int)
for i in range(1, N_ROWS):
    if regime[i - 1] == 0:
        regime[i] = 1 if rng.random() < 0.0028 else 0
    else:
        regime[i] = 0 if rng.random() < 0.028 else 1

volatility_regime = np.where(regime == 0, "low", "high")

# Uneven spacing in microseconds (faster in high-vol periods)
arr_scale_sec = np.where(regime == 0, 0.42, 0.17)
inter_arrival_us = np.maximum(1, rng.exponential(arr_scale_sec, size=N_ROWS) * 1_000_000).astype(np.int64)


# =====================================
# Spoof burst calendar (clustered and intermittent)
# =====================================
spoof_layer_mask = np.zeros(N_ROWS, dtype=bool)
spoof_flip_mask = np.zeros(N_ROWS, dtype=bool)
spoof_event_flag = np.zeros(N_ROWS, dtype=bool)
layer_rank = np.zeros(N_ROWS, dtype=int)  # 1..5 within burst layers
burst_side = np.array([""] * N_ROWS, dtype=object)

target_spoof_rows = int(rng.integers(2_700, 3_701))  # 2.7%-3.7%
used = np.zeros(N_ROWS, dtype=bool)
spoof_rows = 0
burst_starts = []

idx = int(rng.integers(150, 450))
while spoof_rows < target_spoof_rows and idx < N_ROWS - 12:
    gap = int(rng.integers(40, 260)) if rng.random() < 0.70 else int(rng.integers(900, 2600))
    idx += gap
    if idx >= N_ROWS - 12:
        break

    n_layers = int(rng.integers(3, 6))
    n_flips = int(rng.integers(1, 3))
    block_len = n_layers + n_flips

    if used[idx : idx + block_len].any():
        continue

    # Assign a spoofing trader for the full burst
    spoofer = int(rng.choice(type_c_traders))
    burst_direction = str(rng.choice(["buy", "sell"]))

    burst_rows = np.arange(idx, idx + block_len)
    layer_rows = burst_rows[:n_layers]
    flip_rows = burst_rows[n_layers:]

    # Override trader identity to enforce clustered activity
    trader_id[burst_rows] = spoofer
    trader_type[burst_rows] = "C"

    spoof_layer_mask[layer_rows] = True
    spoof_flip_mask[flip_rows] = True
    spoof_event_flag[burst_rows] = True
    burst_side[burst_rows] = burst_direction

    for j, ridx in enumerate(layer_rows, start=1):
        layer_rank[ridx] = j

    used[burst_rows] = True
    burst_starts.append((idx, n_layers, n_flips, burst_direction))
    spoof_rows += block_len

# Fallback fill to guarantee spoof activity stays in 2%-4% band
attempts = 0
while spoof_rows < target_spoof_rows and attempts < 50_000:
    attempts += 1
    start = int(rng.integers(120, N_ROWS - 10))
    n_layers = int(rng.integers(3, 6))
    n_flips = int(rng.integers(1, 3))
    block_len = n_layers + n_flips
    end = start + block_len
    if end >= N_ROWS:
        continue
    if used[start:end].any():
        continue

    spoofer = int(rng.choice(type_c_traders))
    burst_direction = str(rng.choice(["buy", "sell"]))
    burst_rows = np.arange(start, end)
    layer_rows = burst_rows[:n_layers]
    flip_rows = burst_rows[n_layers:]

    trader_id[burst_rows] = spoofer
    trader_type[burst_rows] = "C"
    spoof_layer_mask[layer_rows] = True
    spoof_flip_mask[flip_rows] = True
    spoof_event_flag[burst_rows] = True
    burst_side[burst_rows] = burst_direction
    for j, ridx in enumerate(layer_rows, start=1):
        layer_rank[ridx] = j
    used[burst_rows] = True
    burst_starts.append((start, n_layers, n_flips, burst_direction))
    spoof_rows += block_len

# Shape micro-time clustering for spoof bursts
for start, n_layers, n_flips, _ in burst_starts:
    end = start + n_layers + n_flips
    inter_arrival_us[start:end] = rng.integers(200, 18_000, size=end - start)  # rapid burst activity
    if end < N_ROWS:
        inter_arrival_us[end] += int(rng.integers(250_000, 2_000_000))  # intermittent inactivity after burst

# Build timestamps with microsecond precision
elapsed_us = np.cumsum(inter_arrival_us)
start_ts = np.datetime64("2026-03-02T08:00:00")
timestamp = start_ts + elapsed_us.astype("timedelta64[us]")

# Session labels
q1 = np.quantile(elapsed_us, 1 / 3)
q2 = np.quantile(elapsed_us, 2 / 3)
session_id = np.where(elapsed_us <= q1, "S1", np.where(elapsed_us <= q2, "S2", "S3"))


# =====================================
# Mid-price process with volatility clustering
# =====================================
# Regime-dependent GARCH-like variance process to avoid i.i.d. noise
ret = np.zeros(N_ROWS, dtype=float)
var = np.zeros(N_ROWS, dtype=float)
base_omega = np.where(regime == 0, 2.5e-8, 1.1e-7)
alpha = 0.08
beta = 0.90
var[0] = base_omega[0] / (1 - alpha - beta)

for i in range(1, N_ROWS):
    var[i] = base_omega[i] + alpha * (ret[i - 1] ** 2) + beta * var[i - 1]
    ret[i] = np.sqrt(max(var[i], 1e-10)) * rng.normal(0.0, 1.0)

mid_base = BASE_PRICE + np.cumsum(ret)

# Probabilistic, decaying endogenous impact around spoof layering
impact = np.zeros(N_ROWS, dtype=float)
for start, n_layers, n_flips, direction in burst_starts:
    if rng.random() < 0.68:  # probabilistic market reaction
        sign = 1.0 if direction == "buy" else -1.0
        amp = rng.uniform(0.002, 0.013)
        decay_horizon = int(rng.integers(8, 26))
        tau = rng.uniform(2.5, 7.0)
        burst_end = start + n_layers + n_flips

        for k in range(decay_horizon):
            ridx = burst_end + k
            if ridx >= N_ROWS:
                break
            local_noise = rng.normal(1.0, 0.18)
            impact[ridx] += sign * amp * np.exp(-k / tau) * max(local_noise, 0.2)

mid_price = mid_base + impact

# Dynamic spread process (stochastic, regime-sensitive)
log_spread = np.zeros(N_ROWS, dtype=float)
log_spread[0] = np.log(0.024)
for i in range(1, N_ROWS):
    target = np.log(0.021 if regime[i] == 0 else 0.039)
    shock = rng.normal(0.0, 0.075 if regime[i] == 0 else 0.11)
    jump = rng.normal(0.0, 0.20) if rng.random() < (0.002 if regime[i] == 0 else 0.006) else 0.0
    log_spread[i] = 0.965 * log_spread[i - 1] + 0.035 * target + shock + jump

spread = np.exp(log_spread)
spread = np.clip(spread, 0.01, 0.10)
spread = np.maximum(TICK_SIZE, np.round(spread / TICK_SIZE) * TICK_SIZE)

best_bid = np.round((mid_price - spread / 2) / TICK_SIZE) * TICK_SIZE
best_offer = best_bid + spread
mid_price = (best_bid + best_offer) / 2.0


# =====================================
# Order side, distance-from-touch, price, and size
# =====================================
# Trader-level directional bias with overlap
trader_bias = {int(t): rng.normal(0.0, 0.08) for t in trader_ids}
base_buy_p = np.array([0.5 + trader_bias[int(t)] for t in trader_id])
base_buy_p = np.clip(base_buy_p, 0.35, 0.65)
side = np.where(rng.random(N_ROWS) < base_buy_p, "buy", "sell")

# B-type traders quote near touch, A wider, C mixed
distance_ticks = np.zeros(N_ROWS, dtype=float)
mask_a = trader_type == "A"
mask_b = trader_type == "B"
mask_c = trader_type == "C"

distance_ticks[mask_a] = rng.gamma(shape=2.2, scale=1.7, size=mask_a.sum())
distance_ticks[mask_b] = rng.gamma(shape=1.3, scale=0.95, size=mask_b.sum())
distance_ticks[mask_c] = rng.gamma(shape=1.7, scale=1.35, size=mask_c.sum())

distance_ticks = np.clip(distance_ticks, 0.0, 18.0)

# Layered spoof orders: progressively further from touch
for start, n_layers, n_flips, direction in burst_starts:
    layer_rows = np.arange(start, start + n_layers)
    flip_rows = np.arange(start + n_layers, start + n_layers + n_flips)

    increments = rng.integers(1, 4, size=n_layers)
    layered_levels = np.cumsum(increments).astype(float)

    side[layer_rows] = direction
    distance_ticks[layer_rows] = layered_levels

    opposite = "sell" if direction == "buy" else "buy"
    side[flip_rows] = opposite
    distance_ticks[flip_rows] = rng.uniform(0.0, 1.2, size=n_flips)

# Base fat-tailed market sizes with persona overlays
base_pareto = rng.pareto(a=2.3, size=N_ROWS) + 1.0
order_size = np.zeros(N_ROWS, dtype=float)

# Type A: moderate Gaussian with occasional short-lived and large outliers
order_size[mask_a] = np.clip(rng.normal(28.0, 9.5, size=mask_a.sum()), 2.0, None) * (base_pareto[mask_a] ** 0.33)

# Type B: small, fat-tailed (power-law-like)
order_size[mask_b] = (2.3 + 4.6 * (base_pareto[mask_b] ** 0.85)) * rng.lognormal(0.0, 0.30, size=mask_b.sum())

# Type C: overlaps with B and A (not always extreme)
order_size[mask_c] = (4.5 + 6.0 * (base_pareto[mask_c] ** 0.78)) * rng.lognormal(0.0, 0.42, size=mask_c.sum())

# Grey behaviour: occasional large orders from non-spoof traders
normal_large = (trader_type != "C") & (rng.random(N_ROWS) < 0.009)
order_size[normal_large] *= rng.uniform(7.0, 22.0, size=normal_large.sum())

# Spoof layer size asymmetry (10x-50x median), flip trades are small
baseline_median = np.median(order_size)
order_size[spoof_layer_mask] = baseline_median * rng.uniform(10.0, 50.0, size=spoof_layer_mask.sum()) * rng.lognormal(
    mean=0.0, sigma=0.22, size=spoof_layer_mask.sum()
)
order_size[spoof_flip_mask] = baseline_median * rng.uniform(0.25, 1.6, size=spoof_flip_mask.sum()) * rng.lognormal(
    mean=-0.05, sigma=0.30, size=spoof_flip_mask.sum()
)

order_size = np.clip(order_size, 0.5, None)

# Price derived from BBO and touch distance
price = np.where(
    side == "buy",
    best_bid - distance_ticks * TICK_SIZE,
    best_offer + distance_ticks * TICK_SIZE,
)
price = np.round(price / TICK_SIZE) * TICK_SIZE
price = np.clip(price, 0.01, None)

distance_from_touch = np.where(
    side == "buy",
    (best_bid - price) / TICK_SIZE,
    (price - best_offer) / TICK_SIZE,
)
distance_from_touch = np.round(np.clip(distance_from_touch, 0.0, None), 3)


# =====================================
# Order lifetime and outcome logic
# =====================================
order_lifetime = np.zeros(N_ROWS, dtype=float)

# Type A: long lifetimes (Gamma mean ~8-15s) + occasional short-lived noise
order_lifetime[mask_a] = rng.gamma(shape=3.2, scale=3.5, size=mask_a.sum())
short_noise_a = mask_a & (rng.random(N_ROWS) < 0.08)
order_lifetime[short_noise_a] = rng.lognormal(mean=-0.35, sigma=0.65, size=short_noise_a.sum())

# Type B: overlaps with spoofers, generally short
order_lifetime[mask_b] = np.clip(rng.lognormal(mean=-0.05, sigma=0.70, size=mask_b.sum()), 0.15, 8.0)

# Type C: mixed behaviour with overlap, not all ultra-short
order_lifetime[mask_c] = np.clip(rng.lognormal(mean=0.05, sigma=0.80, size=mask_c.sum()), 0.12, 10.0)

# Spoof layer/flip lifetime adjustments with overlap retained
order_lifetime[spoof_layer_mask] = np.clip(
    rng.lognormal(mean=-1.0, sigma=0.55, size=spoof_layer_mask.sum()), 0.08, 2.4
)
order_lifetime[spoof_flip_mask] = np.clip(
    rng.lognormal(mean=-0.30, sigma=0.65, size=spoof_flip_mask.sum()), 0.12, 3.0
)

# Base probabilities with overlap and dependence on market context
cancel_prob = np.zeros(N_ROWS, dtype=float)
exec_prob = np.zeros(N_ROWS, dtype=float)
open_prob = rng.uniform(0.02, 0.05, size=N_ROWS)  # explicit 2%-5% open bucket

cancel_prob[mask_a] = 0.22 + 0.015 * np.clip(distance_from_touch[mask_a], 0, 8) + 0.06 * (regime[mask_a] == 1)
exec_prob[mask_a] = 0.58 - 0.028 * np.clip(distance_from_touch[mask_a], 0, 8) - 0.05 * (regime[mask_a] == 1)

cancel_prob[mask_b] = 0.70 + 0.020 * np.clip(distance_from_touch[mask_b], 0, 6) + 0.04 * (regime[mask_b] == 1)
exec_prob[mask_b] = 0.24 - 0.015 * np.clip(distance_from_touch[mask_b], 0, 6) + 0.02 * (regime[mask_b] == 1)

cancel_prob[mask_c] = 0.56 + 0.020 * np.clip(distance_from_touch[mask_c], 0, 7) + 0.02 * (regime[mask_c] == 1)
exec_prob[mask_c] = 0.30 - 0.018 * np.clip(distance_from_touch[mask_c], 0, 7) + 0.01 * (regime[mask_c] == 1)

# Inject stochastic variability so no deterministic boundaries emerge
cancel_prob += rng.normal(0.0, 0.035, size=N_ROWS)
exec_prob += rng.normal(0.0, 0.030, size=N_ROWS)

# Spoofing behaviour specifics (still imperfectly separable)
cancel_prob[spoof_layer_mask] = np.clip(
    0.86 + rng.normal(0.0, 0.035, size=spoof_layer_mask.sum()), 0.78, 0.97
)
exec_prob[spoof_layer_mask] = np.clip(
    0.08 + rng.normal(0.0, 0.025, size=spoof_layer_mask.sum()), 0.02, 0.18
)

cancel_prob[spoof_flip_mask] = np.clip(
    0.15 + rng.normal(0.0, 0.06, size=spoof_flip_mask.sum()), 0.04, 0.35
)
exec_prob[spoof_flip_mask] = np.clip(
    0.64 + rng.normal(0.0, 0.08, size=spoof_flip_mask.sum()), 0.35, 0.90
)

# Keep probabilities coherent with explicit open bucket
cancel_prob = np.clip(cancel_prob, 0.05, 0.97)
exec_prob = np.clip(exec_prob, 0.02, 0.94)

# Normalize cancel/exec to exactly consume (1-open) with preserved ratio
budget = 1.0 - open_prob
sum_ce = cancel_prob + exec_prob
ratio_cancel = np.divide(cancel_prob, np.maximum(sum_ce, 1e-8))
cancel_prob = ratio_cancel * budget
exec_prob = (1.0 - ratio_cancel) * budget

u = rng.random(N_ROWS)
cancel_flag = u < cancel_prob
execution_flag = (u >= cancel_prob) & (u < (cancel_prob + exec_prob))

# Logical consistency checks (hard constraints)
assert np.all(best_bid < best_offer)
assert np.all(~(cancel_flag & execution_flag))


# =====================================
# Order identifiers and final frame
# =====================================
order_id = np.arange(1, N_ROWS + 1, dtype=np.int64)

# Price at which order is resting; for convenience keep derived mid and spread columns
df = pd.DataFrame(
    {
        "timestamp": timestamp,
        "session_id": session_id,
        "order_id": order_id,
        "trader_id": trader_id.astype(int),
        "trader_type": trader_type,
        "side": side,
        "order_size": np.round(order_size, 6),
        "price": np.round(price, 6),
        "best_bid": np.round(best_bid, 6),
        "best_offer": np.round(best_offer, 6),
        "mid_price": np.round(mid_price, 6),
        "spread": np.round(spread, 6),
        "cancel_flag": cancel_flag.astype(bool),
        "execution_flag": execution_flag.astype(bool),
        "order_lifetime": np.round(order_lifetime, 6),
        "distance_from_touch": np.round(distance_from_touch, 6),
        "volatility_regime": volatility_regime,
        "instrument": INSTRUMENT,
    }
)

# Add hidden ground-truth marker for evaluation workflows
# (retained in dataset to enable calibration backtesting and FN/FP analysis)
df["is_spoof_event"] = spoof_event_flag

# Enforce exact row count
assert len(df) == N_ROWS

# Save dataset
df.to_csv(OUTPUT_PATH, index=False)


# =====================================
# Required summaries
# =====================================
cancel_by_type = df.groupby("trader_type")["cancel_flag"].mean().sort_index()
size_summary = df["order_size"].describe(percentiles=[0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99])
spoof_share = df["is_spoof_event"].mean()
lifetime_by_type = df.groupby("trader_type")["order_lifetime"].describe(percentiles=[0.25, 0.5, 0.75])

open_share = (~df["cancel_flag"] & ~df["execution_flag"]).mean()

print("Spoofing traders:", sorted(type_c_traders.tolist()))
print("DataFrame shape:", df.shape)
print("CSV exported to:", OUTPUT_PATH)
print("\nCancel ratio per trader_type:")
print(cancel_by_type)

print("\nOrder size distribution summary:")
print(size_summary)

print("\nSpoofing share of total activity: {:.2%}".format(spoof_share))
print("Open-order share (neither cancel nor execute): {:.2%}".format(open_share))

print("\nLifetime distribution comparison by trader_type:")
print(lifetime_by_type)
