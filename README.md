# Autonomous Wireless Sensor Network (WSN) Routing via Machine Learning Telemetry Encoders & Deep Reinforcement Learning

A comprehensive research-grade system that models, classifies, and routes packet traffic across an IEEE 802.15.4 Wireless Sensor Network. The system uses **Machine Learning classifiers** to dynamically monitor node condition (Healthy, Congested, Unhealthy) and **Deep Reinforcement Learning (Dueling Double DQN)** to discover optimal, congestion-resilient paths to the Base Station.

---

## 1. System Architecture & Project Flow

The project is structured into a modular 4-phase pipeline:

```
[ Phase 1: WSN Simulation & Data Generation ]
  * IEEE 802.15.4 Radio & Heinzelman Energy Model
  * Drop-tail FIFO Queue Dynamics & Bursty Traffic
  * 12,000 Telemetry Samples with Zero-Leakage Split
                        │
                        ▼
[ Phase 2: Predictive Telemetry Classification ]
  * 9 Machine Learning Models (Random Forest, XGBoost, MLP, CNN, etc.)
  * Multi-class Prediction: Healthy (0), Congested (1), Unhealthy (2)
  * Real-Time Node Health & Congestion Risk Estimation
                        │
                        ▼
[ Phase 3: Autonomous Path Routing (Deep RL) ]
  * Markov Decision Process (MDP) with 44-Dim State Space
  * Dueling Double Deep Q-Network (Dueling DDQN) with Action Masking
  * Proactive Congestion & Depletion Avoidance toward Base Station
                        │
                        ▼
[ Phase 4: Multi-Modal Result Presentation ]
  * Interactive Real-Time Web Visualizer (Canvas GUI on http://localhost:8080)
  * 300 DPI High-Resolution Network Topology Diagram (live_network_topology.png)
  * Real-Time Terminal Demonstration Script (demo.py)
```

---

## 2. Dataset Specification

### 2.1 Simulation Environment
Data is gathered from an event-driven physics simulation of sensor motes operating under IEEE 802.15.4:
* **Topology:** 35–50 sensor nodes distributed randomly across a $200\text{ m} \times 200\text{ m}$ field.
* **Base Station (Sink):** Fixed destination node at the center or edge.
* **Transmission Range:** $R_{\text{tx}} = 50.0\text{ m}$ (log-distance path loss with log-normal shadowing).
* **Battery Model:** Heinzelman first-order radio model ($E_{\text{elec}} = 50\text{ nJ/bit}$, $\epsilon_{\text{fs}} = 10\text{ pJ/bit/m}^2$, $\epsilon_{\text{mp}} = 0.0013\text{ pJ/bit/m}^4$).
* **Queue Dynamics:** Drop-tail FIFO queues with capacity $Q_{\text{cap}} = 30\text{ packets}$.
* **Traffic Regimes:** Constant Bit Rate (CBR), Poisson arrival, and Bursty Pareto traffic.

### 2.2 Telemetry Feature Vector
For every sensor node, 7 continuous telemetry metrics are captured:
1. **Residual Energy Ratio ($E_{\text{res}}$):** $\frac{E_{\text{current}}}{E_{\text{initial}}} \in [0, 1]$
2. **Queue Occupancy Ratio ($Q_{\text{occ}}$):** $\frac{Q_{\text{len}}}{Q_{\text{cap}}} \in [0, 1]$
3. **Traffic Intensity:** Ratio of arrival rate to service capacity $\frac{\lambda_{\text{arr}}}{\mu_{\text{service}}}$
4. **Link ETX:** Expected Transmission Count derived from receiver SINR and packet loss rate.
5. **Packet Arrival Rate:** Instantaneous arrival frequency ($\text{pkts/sec}$).
6. **Packet Transmission Rate:** Forwarding throughput ($\text{pkts/sec}$).
7. **Distance to Sink Ratio:** Normalized Euclidean distance to Base Station $\frac{d_{\text{BS}}}{d_{\text{max}}}$.

### 2.3 Ground-Truth Labeling & Partitioning
Labels are generated following IETF CODA (Congestion Detection and Avoidance) and PCCP criteria:
* **Class 0 — Healthy (Green):** Normal buffer load ($Q_{\text{occ}} < 0.70$), ample battery ($E_{\text{res}} > 0.15$), and low loss ($\text{PLR} < 0.40$).
* **Class 1 — Congested (Amber):** High buffer occupancy ($Q_{\text{occ}} \ge 0.70$) or excessive traffic intensity ($\ge 0.90$), indicating impending queue drops.
* **Class 2 — Unhealthy (Red):** Critical battery depletion ($E_{\text{res}} \le 0.15$) or severe wireless channel degradation ($\text{PLR} \ge 0.40$).

**Dataset Split (Zero Data Leakage across independent random network topologies):**
* **Total Samples:** 12,000 samples
* **Training Set:** 8,400 samples (70%)
* **Validation Set:** 1,800 samples (15%)
* **Test Set:** 1,800 samples (15%)
* **Saved in:** `V2/datasets/train.csv`, `val.csv`, `test.csv`

---

## 3. Machine Learning Models & Selection Decision

### 3.1 Evaluated Classifiers (9 Models)
Nine machine learning algorithms were trained and benchmarked across accuracy, Macro-F1, inference latency, and embedded viability:
1. **Logistic Regression:** Linear baseline.
2. **Decision Tree:** Interpretable rule-based baseline.
3. **Random Forest:** Ensemble of bagging decision trees.
4. **XGBoost:** Extreme Gradient Boosted trees.
5. **Support Vector Machine (SVM-RBF):** Non-linear kernel classification.
6. **Multi-Layer Perceptron (MLP):** Deep feedforward neural network.
7. **1D-CNN (`TemporalCNN1D` — Selected Champion Classifier):** Feature-space 1D temporal convolution.
8. **CNN-LSTM:** Spatio-temporal recurrent sequential model.
9. **Graph Convolutional Network (GCN):** Topology graph neural network.

### 3.2 Model Selection Rationale
* **Active Classifier Selected: 1D-CNN (`TemporalCNN1D`)**
* **Technical Rationale:** While tree models (Random Forest, XGBoost) offer lower computational latency on microcontrollers, 1D-CNN provides superior representation learning over consecutive multi-channel sensor telemetry. Its 1D convolutional kernels extract local temporal momentum (rate of queue accumulation and battery discharge velocity) directly from normalized telemetry sequences.
* **Outputs Generated by 1D-CNN:**
  1. **Discrete Class Label:** $y \in \{\text{Healthy (0)}, \text{Congested (1)}, \text{Unhealthy (2)}\}$.
  2. **Class Probabilities:** $[P(\text{Healthy}), P(\text{Congested}), P(\text{Unhealthy})]$.
  3. **Continuous Health/Congestion Risk Score:**
     $$\text{Risk}(u) = \min\left(1.0, \frac{0.8 \cdot P(\text{Congested}) + 2.0 \cdot P(\text{Unhealthy})}{2.0}\right)$$
     This continuous risk score is fed directly into the Dueling Double DQN routing engine to penalize degrading nodes before packet drops occur.

---

## 4. Routing Techniques & Champion Selection (No Dijkstra or A*)

Standard textbook shortest-path algorithms (Dijkstra, A*) were intentionally avoided because:
1. **Centralized Global Knowledge Assumption:** Broadcasting every node's live queue and battery state creates a **broadcast storm**, rapidly draining sensor batteries.
2. **Bottleneck Congestion:** Shortest-path routing funnels all traffic through identical central nodes, leading to buffer overflow and packet drops.

Instead, the system implements **Autonomous, Distributed Reinforcement Learning**, benchmarks all algorithms, and explicitly selects the champion router:

### 4.1 Champion Router: Autonomous Deep Reinforcement Learning (Dueling Double DQN)
* **Selected as the Active Production Router.**
* **Core Advantages:**
  * **Dual-Stream Value Decomposition:** Decouples state value $V(s)$ from neighbor advantage $A(s, a)$, eliminating Q-value overestimation.
  * **Invalid Action Masking:** Completely eliminates routing loops and ping-pong bounces without requiring hop-count broadcast.
  * **1D-CNN Risk Integration:** Proactively detours around nodes flagged as congested or low battery by 1D-CNN before buffer drops occur.
* **State Space $\mathcal{S}$ (44 Dimensions):**
  * Local telemetry: $[E_{\text{res}}, Q_{\text{occ}}, d_{\text{BS}}, \text{Degree}]$ (4 dims)
  * Candidate 1-hop neighbor states for up to 8 neighbors: $[\text{Valid}, \Delta d_{\text{BS}}, Q_{\text{occ}}, E_{\text{res}}, \text{Risk}_{\text{1D-CNN}}]$ ($8 \times 5 = 40$ dims)
* **Multi-Objective Reward Function:**
  $$R(s, a, s') = R_{\text{delivery}} + R_{\text{progress}} - R_{\text{queue}} - R_{\text{energy}} - R_{\text{risk}} - R_{\text{step}}$$
  * Terminal delivery to Base Station: $+10.0$
  * Progress toward sink: $+1.5 \cdot \frac{d(u) - d(v)}{R_{\text{tx}}}$
  * Buffer bloat penalty: $-2.5 \cdot Q_{\text{occ}}(v)$
  * Battery depletion penalty: $-2.0 \cdot (1 - E_{\text{res}}(v))$
  * 1D-CNN health risk penalty: $-3.5 \cdot \text{Risk}(v)$
  * Hop step cost: $-0.2$
  * Routing loop penalty: $-8.0$

### 4.2 Classical Tabular Q-Routing Baseline (Boyan & Littman, 1994)
* Each node $u$ maintains local Q-values $Q(u, v)$ estimating expected delivery time via neighbor $v$.
* Fully decentralized learning with asynchronous updates.

### 4.3 Localized Greedy Geographic Routing (GPSR Baseline)
* Forwards packets to the neighbor providing maximum distance reduction toward the sink. Serves as a localized baseline.

---

## 5. How Results are Displayed

The system provides three synchronized presentation modes:

### Mode 1: Interactive Real-Time Web Visualizer (Canvas GUI)
* **How to run:**
  ```bash
  python V2/visualizer.py
  ```
* **URL:** `http://localhost:8080`
* **Features:**
  * **2D Field Canvas:** Displays the $200\text{ m} \times 200\text{ m}$ topology with 35 sensor nodes, Base Station, and active radio links.
  * **Color-Coded Status Rings:** 🟢 Healthy, 🟡 Congested, 🔴 Unhealthy.
  * **Animated Packet Forwarding:** Watch a glowing packet bullet travel hop-by-hop along the Dueling DDQN route.
  * **Live Control Buttons:**
    * `[Run ML Classification]`: Re-runs Random Forest across all nodes.
    * `[Inject Congestion]`: Simulates a traffic burst on central nodes (watch them turn amber).
    * `[Find RL Shortest Path]`: Discovers and animates the optimal path from any selected source node to the Base Station.
  * **Telemetry Inspector Sidebar:** Click any node on the canvas to inspect its battery, queue length, arrival rate, and ML prediction probabilities.

### Mode 2: High-Resolution Publication Diagram
* **How to run:**
  ```bash
  python V2/plot_topology.py
  ```
* **Output:** `V2/results/plots/live_network_topology.png` (300 DPI)
* **Contents:**
  * Complete 2D topology with IEEE 802.15.4 link mesh.
  * Color-coded classification of all nodes.
  * Glowing cyan arrows tracing the multi-hop RL route.
  * Inset table detailing source, hop count, and bypassed congested nodes.

### Mode 3: Real-Time Terminal Demonstration
* **How to run:**
  ```bash
  python V2/demo.py
  ```
* **Output:**
  * **Phase 1:** Topology configuration (35 nodes, 112 links, radio range 50m).
  * **Phase 2:** ML Classification table showing per-node battery, queue, and predicted state with confidence %.
  * **Phase 3:** Multi-hop path traces from peripheral edge nodes (e.g., Node 11 $\rightarrow$ Node 1 $\rightarrow$ Node 27 $\rightarrow$ Node 31 $\rightarrow$ Node 28 $\rightarrow$ Base Station), reporting exact decision latency and bypassed congested bottlenecks.
  * **Phase 4:** Automatic generation of the topology image.

---

## 6. Quickstart Commands

```bash
# 1. Activate the Python virtual environment
source V2/venv/bin/activate

# 2. Run the live terminal demonstration (under 2 seconds)
python V2/demo.py

# 3. Generate the high-resolution topology diagram
python V2/plot_topology.py

# 4. Launch the interactive browser visualizer
python V2/visualizer.py
# Open browser at: http://localhost:8080
```

---

## 7. Directory Structure

```
WSN/
├── README.md                      # Master documentation (this file)
└── V2/
    ├── configs/
    │   └── default_config.json    # WSN network, energy, channel & traffic parameters
    ├── datasets/
    │   ├── train.csv              # 8,400 training samples
    │   ├── val.csv                # 1,800 validation samples
    │   ├── test.csv               # 1,800 test samples
    │   └── labeling.py            # CODA/PCCP ground-truth labeling logic
    ├── models/                    # Neural architectures (MLP, CNN, LSTM, GCN)
    ├── preprocessing/
    │   ├── preprocessor.py        # Robust standard scaling
    │   └── scaler.joblib          # Serialized fitted scaler
    ├── simulator/
    │   ├── channel.py             # IEEE 802.15.4 path loss & log-normal shadowing
    │   ├── energy.py              # Heinzelman first-order radio dissipation model
    │   ├── node.py                # Sensor node state, FIFO queue & telemetry
    │   ├── network.py             # Event-driven discrete simulation engine
    │   └── packet.py              # Packet headers, hop history & drop metrics
    ├── routing/
    │   ├── rl_agent.py            # Dueling DDQN Neural Net & Tabular Q-Router
    │   ├── rl_environment.py     # 44-Dim MDP Environment with Action Masking
    │   ├── conventional.py        # Localized greedy geographic baseline (GPSR)
    │   ├── cost_functions.py      # Multi-metric edge cost functions
    │   └── hysteresis.py          # Route flapping suppression filter
    ├── experiments/
    │   ├── train_ml_classifiers.py# 9-model ML training & benchmarking pipeline
    │   ├── train_rl_agent.py      # Dueling DDQN reinforcement learning loop
    │   └── run_routing_suite.py   # Factorial routing performance suite
    ├── results/
    │   ├── checkpoints/           # Trained models (.pt and .joblib)
    │   │   ├── random_forest.joblib
    │   │   ├── xgboost.joblib
    │   │   └── dueling_ddqn_router.pt
    │   ├── plots/                 # High-resolution evaluation charts & topology diagram
    │   │   ├── live_network_topology.png
    │   │   ├── fig1_ml_macro_f1_vs_latency.png
    │   │   ├── fig2_pdr_comparison_by_traffic.png
    │   │   ├── fig3_latency_and_energy_tradeoff.png
    │   │   └── fig8_rl_training_curves.png
    │   └── tables/                # Benchmark performance metrics (CSV)
    ├── demo.py                    # Terminal live demonstration script
    ├── plot_topology.py           # High-resolution topology diagram generator
    ├── visualizer.py              # Interactive web dashboard server
    └── visualizer/
        └── index.html             # Dark-theme 2D canvas visualizer application
```
