# Data Center Cooling Benchmarks: Evaluation and Comparison of Intelligent Control Architectures

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/imamdoula004/Data-Center-Cooling-Benchmarks/blob/main/Data_Center_Cooling_Benchmarks.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Official benchmarking code and digital twin simulation environment for the IEEE research manuscript:

**"AI-Enhanced Hybrid EMPC for Secure, Cost and Carbon Optimal Data Center Cooling."**

---

## 📖 Project Overview

Modern data centers consume massive amounts of electricity, with cooling systems alone accounting for up to **40% of total facility energy usage**. Traditional controllers (e.g., classical PID loops) operate reactively using short-term local feedback. They suffer from **operational myopia**, meaning they cannot anticipate upcoming thermal workloads, dynamic electricity grid tariffs, or fluctuating carbon intensity signals. Furthermore, under cyber-physical **False Data Injection (FDI)** attacks, reactive controllers are prone to closed-loop instability, leading to server damage or extreme cooling overhead.

This repository provides a comprehensive **digital twin benchmarking suite** designed to evaluate, validate, and compare advanced control strategies for data center cooling under normal and adversarial conditions. The environment models a coupled multi-zone facility subjected to dynamic workloads, grid volatility, hardware failures, and cyber-physical FDI attacks.

```mermaid
graph TD
    subgraph Input Data Pipeline
        K[Kaggle Telemetry Datasets] -->|Load & Clean| D[Numeric Matrix]
        D -->|Sequence Engine| S[Sequence Window: 24h x 38 Features]
    end

    subgraph Forecasting & Control Optimization
        S -->|Predict Horizon| L[Bi-LSTM Model]
        L -->|6h Forecast| F[Temporal Mean & Zone Map]
        F -->|Disturbance d_t| E[EMPC Optimization Core]
        T[Current Room Temps T_t] --> E
        P[Electricity Price p_t] --> E
        C[Carbon Intensity c_t] --> E
        E -->|Solve CVXPY| Ue[Economic Control u_empc]
    end

    subgraph Supervisory & Blending Layer
        T -->|Error: T_ref - T_t| PID[Supervisory PID Loop]
        PID -->|Safe Control| Up[PID Control u_pid]
        T -->|Calculate Deviation| V[Lyapunov Stability: V_t]
        V -->|Compute Blending| A[Blending Factor: alpha = tanh 0.12V_t]
        Ue -->|Weight = alpha| B[Control Blending Block]
        Up -->|Weight = 1 - alpha| B
        B -->|u_blended| DAMP[Lyapunov Damping: u * 1 - 0.01V_t]
        DAMP -->|Clip to 0.2, 1.0| U[Final Command u_t]
    end

    subgraph Cyber-Physical Digital Twin
        U -->|Effective Cooling| PHY[Heterogeneous Multi-Zone Server Building]
        W[Workload Profile w_t] --> PHY
        FDI[FDI Attack Vector a_t] --> PHY
        PHY -->|Thermal Evolution| T_new[Next Temperature T_t+1]
        T_new -->|Feedback Loop| T
    end

    classDef input fill:#e1f5fe,stroke:#03a9f4,stroke-width:2px,color:#000;
    classDef model fill:#fff3e0,stroke:#ff9800,stroke-width:2px,color:#000;
    classDef blend fill:#e8f5e9,stroke:#4caf50,stroke-width:2px,color:#000;
    classDef physics fill:#fce4ec,stroke:#e91e63,stroke-width:2px,color:#000;
    
    class K,D,S input;
    class L,F,E,Ue,PID,Up model;
    class V,A,B,DAMP,U blend;
    class PHY,T_new,T,W,FDI physics;
```

---

## 🎛 Control Architectures Evaluated

The suite compares five control strategies representing different paradigms in industrial automation:

### 1. Proportional-Integral-Derivative (PID) Control
A classical feedback controller that reactively adjusts cooling commands based on the proportional, integral, and derivative errors of room temperature deviation from the reference zone.

### 2. Model Predictive Control (MPC)
A tracking controller that optimizes cooling actuator commands over a receding horizon to minimize room temperature deviations from the target setpoint.

### 3. Economic Model Predictive Control (EMPC)
An optimization-based controller that minimizes operational electricity costs and carbon emissions over a forecasting horizon while maintaining zone temperatures within safety limits.

### 4. Robust Economic MPC (ROBUST)
A security-hardened EMPC formulation that dynamically tightens actuator bounds and penalizes control efforts under periods of high estimated cyber-physical risk.

### 5. Hybrid EMPC-PID with Bi-LSTM (HYBRID)
Our proposed architecture that combines economic MPC commands with a Lyapunov-supervised PID fallback loop based on real-time temperature stability margins.

---

## 🧮 Digital Twin Physics & Cyber-Physical Attack Model

### 1. Heterogeneous Multi-Zone Thermal Dynamics
A state-space model that simulates heat transfer, thermal inertia, cooling actuation, workload heat dissipation, and sensor noise across five coupled data center server zones.

### 2. Cyber-Physical False Data Injection (FDI) Attack Model
An adversarial simulation that hijacks temperature telemetry using a mixture of persistent offsets, random data bursts, and feedback-amplified instability injections.

---

## 🏆 Benchmarking & Evaluation Metrics

The framework evaluates controllers over a continuous multi-hour simulation window under FDI attacks and records the following metrics:
* **Energy Consumption (kWh)**: Total electricity consumed by the chilled air fan and cooling system.
* **Operational Cost ($)**: Real-time monetary expenditure computed using fluctuating hourly grid electricity prices.
* **Scope 2 Carbon (kg CO₂)**: Environmental footprint tracking carbon emissions based on dynamic grid carbon intensity.
* **Lyapunov Stability**: Average Lyapunov energy to evaluate temperature convergence and feedback loop stability under stress.
* **Average Temperature & Std. Dev. (°C)**: Thermal performance indices to measure temperature deviation and control variance in each zone.
* **Computation Time (s)**: Processing overhead for solving optimization objectives at each simulation time step.

---

## 🖲 Data Preprocessing & Sequence Engineering

### 1. Robust Dataset Loading & Cleaning
Real-world telemetry is sourced from four separate Kaggle datasets. The data loader handles misaligned time indices, missing columns, and string values using clean numeric conversion, forward-fill/back-fill pipelines, and dataset-specific normalization.

### 2. Receding Horizon Sequence Engine
Generates 24-hour historical windows with 6-hour prediction horizons, applying a linear ramp weight to prioritize recent history and avoid flat forecasts.

---

## 📂 Repository Structure

```
├── Data_Center_Cooling_Benchmarks.ipynb        # Clean benchmarking notebook (run to generate results)
├── LICENSE                                     # MIT License
├── README.md                                   # Repository documentation
└── .gitignore                                  # Git ignore list
```

> [!NOTE]
> Running the Jupyter notebook local or in Colab executes the entire benchmarking pipeline and generates the performance tables and figures (saving them locally in the workspace).

---

## 🚀 Getting Started

### 1. Prerequisites
Install the required scientific computing and machine learning dependencies:
```bash
pip install numpy pandas scipy tensorflow cvxpy matplotlib seaborn
```

### 2. Clone the Repository
```bash
git clone https://github.com/imamdoula004/Data-Center-Cooling-Benchmarks.git
cd Data-Center-Cooling-Benchmarks
```

### 3. Kaggle Dataset Configuration
The simulation runs on real-world energy, weather, and grid telemetry. The notebook automatically handles dataset download using the Kaggle API.

1. Generate your Kaggle API token from your Kaggle Account page (`Account -> Create New API Token`). This downloads a `kaggle.json` credentials file.
2. Place `kaggle.json` in the root folder of the project.
3. The notebook will automatically configure the environment and download the required datasets:
   - **Energy and Environmental Data:** [mertkont/energy-and-environment-data](https://www.kaggle.com/datasets/mertkont/energy-and-environment-data)
   - **Electricity Consumption:** [littlebaldturtle/electricity-consumption](https://www.kaggle.com/datasets/littlebaldturtle/electricity-consumption)
   - **Electricity Load Forecasting:** [saurabhshahane/electricity-load-forecasting](https://www.kaggle.com/datasets/saurabhshahane/electricity-load-forecasting)
   - **Data Center Cold Source Control:** [programmer3/data-center-cold-source-control-dataset](https://www.kaggle.com/datasets/programmer3/data-center-cold-source-control-dataset)

### 4. Running the Benchmarks
Open the Jupyter notebook locally:
```bash
jupyter notebook Data_Center_Cooling_Benchmarks.ipynb
```
Or execute it in Google Colab (by clicking the badge at the top). Run all cells sequentially to:
1. Initialize the dataset paths and download Kaggle CSVs.
2. Clean, normalize, and construct time series sequence tensors.
3. Train the Bidirectional LSTM forecaster model.
4. Execute the 7-day multi-zone digital twin simulation under cyber-physical attacks.
5. Compile and export the raw raw and normalized KPI metrics.
6. Generate and save all publication plots.

---

## 🔬 Reproducibility

To ensure complete deterministic replication of all benchmarking results, the simulation environment fixes random seeds across all packages:
```python
import os
import random
import numpy as np
import tensorflow as tf

SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
```
This guarantees identical trajectories, KPI metrics, and figures on every run.

---

## 📝 Citation

If you use this benchmarking suite, codebase, or results in your research, please cite the following manuscript:

```bibtex
@article{doula2026datacoolingbenchmarks,
  title={AI-Enhanced Hybrid EMPC for Secure, Cost and Carbon Optimal Data Center Cooling},
  author={Doula, Imam Ud and Rahman, Adrita and Fatima, Samiha},
  journal={IEEE Transactions on Sustainable Computing (Under Review)},
  year={2026}
}
```

---

## 🛡 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.
