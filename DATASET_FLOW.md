# 📊 Dataset Loading & Combination Flow

> **File:** `wsn_cnn_dddqn/data_loader.py`  
> **Purpose:** Load 5 raw Excel datasets → clean → label → align → merge into a single unified NumPy matrix `(X, y)` for 1-D CNN training.

---

## 1. The 5 Raw Datasets

| # | File | Rows | Unique Nodes | Has Label? | Label Column |
|---|------|------|--------------|------------|--------------|
| 1 | `WSN_CrossLayer_Selected_Features.xlsx` | 5,000 | 99 | ❌ No | — |
| 2 | `WSN_DS_Selected_Features.xlsx` | 374,661 | Many | ✅ Yes | `Attack type` |
| 3 | `WSN_Latency_Selected_Features.xlsx` | 1,000 | 1,000 | ✅ Yes | `Latency_Category` |
| 4 | `WSN_Project_Important_Features(1).xlsx` | 1,000 | 1,000 | ❌ No | — |
| 5 | `WSN_Selected_Features(3).xlsx` | 10,000 | 10,000 | ❌ No | — |

---

## 2. End-to-End Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         build_cnn_dataset()                                 │
│  (main entry point — returns dict with X, y, scaler, feature_names)        │
└──────────────────────────────┬──────────────────────────────────────────────┘
                               │
          ┌────────────────────┼─────────────────────┐
          │                    │                      │
   ┌──────▼──────┐      ┌──────▼──────┐      ┌───────▼─────┐
   │ CrossLayer  │      │     DS      │      │   Latency   │
   │  Loader     │      │  Loader     │      │   Loader    │
   └──────┬──────┘      └──────┬──────┘      └──────┬──────┘
          │                    │                     │
   ┌──────▼──────┐      ┌──────▼──────┐      ┌──────▼──────┐
   │  Project    │      │  Selected   │      │             │
   │  Loader     │      │  Loader     │      │             │
   └──────┬──────┘      └──────┬──────┘      │             │
          │                    │             │             │
          └────────────────────┼─────────────┘             │
                               │                           │
                    ┌──────────▼──────────┐                │
                    │   Label Assignment  │◄───────────────┘
                    │  (Rule-based or Map)│
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │ Feature Selection   │
                    │  & Zero-Padding     │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │    np.vstack()      │
                    │  Vertical Concat    │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │  MinMaxScaler       │
                    │  Normalization      │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │   Final Dataset     │
                    │ X: (N, 7) float32   │
                    │ y: (N,)  int64      │
                    │ Labels: 0/1/2       │
                    └─────────────────────┘
```

---

## 3. Per-Dataset Processing Steps

### 3.1 CrossLayer Dataset
```
WSN_CrossLayer_Selected_Features.xlsx
  │
  ├── Read Excel (5,000 rows)
  ├── Rename columns → canonical names (e.g. Node_ID → node_id)
  ├── Drop: Initial_Energy
  ├── Select 7 CNN features:
  │     node_velocity, residual_energy, link_quality,
  │     packet_loss_rate, distance_to_nexthop, delay, throughput
  ├── Label: RULE-BASED  ──────────────────────────────────────────┐
  │     HEALTHY(0):    residual_energy > 0.6                       │
  │                    AND packet_loss_rate < 0.08                 │
  │                    AND link_quality > 0.7                      │
  │     CONGESTED(1):  packet_loss_rate > 0.15                     │
  │                    OR  link_quality < 0.55                     │
  │     UNHEALTHY(2):  all others (default)                        │
  └── Output: X=(5000, 7), y=(5000,)  ◄───────────────────────────┘
```

---

### 3.2 DS Dataset
```
WSN_DS_Selected_Features.xlsx
  │
  ├── Read Excel (374,661 rows)
  ├── Strip whitespace from column names
  ├── Rename columns → canonical names
  ├── Sample 15,000 rows (random_state=42) for tractability
  ├── Select 6 CNN features:
  │     expended_energy, dist_to_ch, dist_ch_to_bs,
  │     data_sent, data_received, data_sent_to_bs
  ├── Zero-Pad to 7 columns (pad 1 column of zeros on the right)
  ├── Label: MAP from existing "Attack type" column ─────────────────┐
  │     Normal    → 0 (Healthy)                                      │
  │     Flooding  → 1 (Congested – bandwidth attack)                 │
  │     TDMA      → 1 (Congested – timing disruption)               │
  │     Grayhole  → 2 (Unhealthy – selective drop)                  │
  │     Blackhole → 2 (Unhealthy – full drop)                       │
  └── Output: X=(15000, 7), y=(15000,) ◄───────────────────────────┘
```

---

### 3.3 Latency Dataset
```
WSN_Latency_Selected_Features.xlsx
  │
  ├── Read Excel (1,000 rows)
  ├── Rename columns → canonical names
  ├── Select 7 CNN features:
  │     hop_count, transmission_delay, buffer_occupancy,
  │     channel_utilization, energy_level, link_quality, pdr
  ├── Zero-Pad to 7 columns (already 7 — no padding needed)
  ├── Label: MAP from existing "Latency_Category" column ────────────┐
  │     Low Latency    → 0 (Healthy)                                 │
  │     Medium Latency → 2 (Unhealthy)                              │
  │     High Latency   → 1 (Congested)                              │
  └── Output: X=(1000, 7), y=(1000,) ◄─────────────────────────────┘
```

---

### 3.4 Project Dataset
```
WSN_Project_Important_Features(1).xlsx
  │
  ├── Read Excel (1,000 rows)
  ├── Rename columns → canonical names
  ├── Select 6 CNN features:
  │     residual_energy, proximity_to_ch, total_energy_used,
  │     network_lifetime, data_packets_sent, data_packets_received
  ├── Zero-Pad to 7 columns (pad 1 column of zeros on the right)
  ├── Label: RULE-BASED ──────────────────────────────────────────────┐
  │     proxy_pdr = data_packets_received / data_packets_sent        │
  │     HEALTHY(0):    pdr > 0.8  AND  e_norm > 0.6                 │
  │     CONGESTED(1):  pdr < 0.4  OR   e_norm < 0.25               │
  │     UNHEALTHY(2):  all others (default)                          │
  └── Output: X=(1000, 7), y=(1000,) ◄──────────────────────────────┘
```

---

### 3.5 Selected Dataset
```
WSN_Selected_Features(3).xlsx
  │
  ├── Read Excel (10,000 rows)
  ├── Rename columns → canonical names
  ├── Drop: Timestamp
  ├── Select 9 CNN features:
  │     residual_energy, transmission_power, signal_strength,
  │     noise_level, energy_consumption, packet_loss_rate,
  │     network_lifetime, temperature, humidity
  ├── Truncate to first 7 columns (align with CrossLayer width)
  ├── Label: RULE-BASED ──────────────────────────────────────────────┐
  │     e_norm  = residual_energy / max(residual_energy)             │
  │     plr_norm = packet_loss_rate / max(packet_loss_rate)          │
  │     HEALTHY(0):    e_norm > 0.5  AND  plr_norm < 0.15           │
  │     CONGESTED(1):  plr_norm > 0.6  OR  e_norm < 0.1            │
  │     UNHEALTHY(2):  all others (default)                          │
  └── Output: X=(10000, 7), y=(10000,) ◄────────────────────────────┘
```

---

## 4. Merging & Final Normalization

```
 CrossLayer  │  DS        │  Latency   │  Project   │  Selected
 (5000, 7)   │  (15000,7) │  (1000, 7) │  (1000, 7) │  (10000, 7)
─────────────┴────────────┴────────────┴────────────┴────────────
                         np.vstack()
                              │
                    ┌─────────▼─────────┐
                    │  X = (32000, 7)   │
                    │  y = (32000,)     │
                    └─────────┬─────────┘
                              │
               MinMaxScaler().fit_transform(X)
               (scales each of the 7 columns to [0, 1])
                              │
                    ┌─────────▼─────────┐
                    │  X: float32       │
                    │  y: int64         │
                    │  Classes: 0/1/2   │
                    └───────────────────┘
```

> **Note:** The DS dataset is sampled down from 374,661 → 15,000 rows for tractability.  
> The total row count may vary slightly based on NaN drops per dataset.

---

## 5. Label Encoding Summary

| Label | Class Name | Meaning |
|-------|------------|---------|
| `0` | **Healthy** | High energy, low loss, good link quality |
| `1` | **Congested** | High packet loss / bandwidth attack / high latency |
| `2` | **Unhealthy** | Low energy, poor PDR, severe network degradation |

---

## 6. Feature Width Alignment Strategy

All datasets must share the **same number of columns (7)** before stacking:

| Dataset | Native Features | Width | Action |
|---------|----------------|-------|--------|
| CrossLayer | 7 | 7 | ✅ Reference — no change |
| DS | 6 | → 7 | ➕ Pad 1 zero column on right |
| Latency | 7 | 7 | ✅ Already aligned |
| Project | 6 | → 7 | ➕ Pad 1 zero column on right |
| Selected | 9 | → 7 | ✂️ Truncate to first 7 columns |

---

## 7. Output Dictionary

The `build_cnn_dataset()` function returns:

```python
{
    "X":             np.ndarray  # shape (N, 7), dtype float32 — normalized features
    "y":             np.ndarray  # shape (N,),  dtype int64   — class labels 0/1/2
    "scaler":        MinMaxScaler  # fitted scaler (for inference-time normalization)
    "feature_names": list[str]    # 7 feature names from CrossLayer
    "n_features":    int          # 7
}
```

---

## 8. Data Flow Summary (One-Liner View)

```
5 Excel files
  → Load + Rename columns
    → Rule-based OR existing label mapping → y
      → Select/Pad/Truncate features to width=7 → X
        → np.vstack all X,  np.concatenate all y
          → MinMaxScaler([0,1])
            → Final (X, y) ready for 1-D CNN
```
