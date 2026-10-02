# 🚦 SIH26137 — Quantum-Inspired Traffic Route Optimization (Delhi NCR)

A quantum-inspired route optimization system (QPSO-based dynamic VRP) evaluated on real Delhi NCR 2024 traffic probe data (Aug 11–30, including Rakshabandhan festival traffic).

---

## 🚀 Quick Start

### 1. Launch Interactive Dashboard (Localhost)

```bash
# Clone the repository
git clone https://github.com/Gutsmaster/Quantum-Inspired-Intelligent-Traffic-Route-Optimization.git
cd Quantum-Inspired-Intelligent-Traffic-Route-Optimization

# Option A: Using Streamlit directly (Windows / Linux / macOS)
pip install streamlit==1.51.0 pandas==3.0.2
python -m streamlit run dashboard/app.py

# Option B: Using the bash launcher script
./run_dashboard.sh
```

Once launched, open your browser at `http://localhost:8501`.

---

## 🛠️ Full Pipeline Execution

To re-generate all benchmark results and map outputs (requires ~11 minutes runtime):

```bash
./run_full_pipeline.sh    # requires data/new_delhi_traffic_dataset/
```

*If script permissions are needed on Linux/macOS:*
```bash
chmod +x run_dashboard.sh run_full_pipeline.sh
```

---

## 📊 Headline Benchmark Results

Held-out evaluation: 8 instances × 4 seeds = 32 paired runs (never used for parameter tuning):

| Algorithm vs. | Mean Cost Improvement | QPSO Wins | Wilcoxon p-value |
|---|---|---|---|
| First-pass QPSO | **+54.5%** | 100% | < 0.0001 |
| Tuned Local Search | **+28.5%** | 91% | < 0.0001 |
| Ant Colony Optimization (ACO) | **+39.6%** | 97% | < 0.0001 |
| Plain Genetic Algorithm (GA) | **+12.2%** | 88% | < 0.0001 |
| Classical PSO (identical infra) | **+3.9%** | 62% | 0.104 (directional) |
| **Memetic GA (same local search)** | **-3.4%** | 19% | **0.0024 (GA superior)** |
| OR-Tools CVRPTW (5s runtime) | -8.9% mean gap (Delhi instances: -4.9%) | - | - |

---

## 📍 Real Data Findings (Delhi Probe Data)

- **Festival Anomalies (Rakshabandhan Aug 19 vs Normal Aug 12):** A route plan optimized for a normal weekday incurs an **8.1% cost penalty at 9:00 AM** and **2.7% at 6:00 PM** when executed on Rakshabandhan without re-optimization.
- **Departure Time Regret:** Operating on a static 9:00 AM plan introduces a **5.7% mean regret** across departure hours (6:00 AM - 9:00 PM).
- **Road Closure Resilience:** After a simulated road closure, retrieving an alternative plan from the diverse-plan archive produces a path costing only **3.6% above full re-optimization instantly**, eliminating re-computation delays.

---

## 📁 Repository Structure

```
├── dashboard/
│   └── app.py              # Streamlit interactive visualizer
├── outputs/
│   ├── results.json        # Compiled benchmark & experiment stats
│   ├── map.html            # Delhi NCR road route visualization
│   ├── benchmark_table.csv # Detailed benchmark metrics per instance
│   └── *.png               # Plots (Pareto, convergence, scalability, recovery)
├── src/                    # Core algorithm implementation & pipeline scripts
├── build_map.py            # Route map generator
├── run_everything.py       # Full benchmark execution pipeline
├── run_dashboard.sh        # Bash launcher for local dashboard
├── run_full_pipeline.sh    # Bash script to rerun complete experiment suite
├── requirements.txt        # Python dependency manifest
└── README.md               # Project documentation
```

---

## ⚠️ Notes & Limitations

- **Real inputs:** Road geometry, hourly probe counts, festival/monsoon observation windows.
- **Derived inputs:** BPR congestion function fitted to dataset rush-hour statistics.
- **Synthetic/Approximations:** Customer demand & time windows; Solomon-style test instances; COPERT-inspired emissions curves.
- **Negative results noted:** The quantum update rule shows non-statistically significant gain over classical PSO ($p \approx 0.10$); reverse-annealing recovery yielded no measurable benefit; Memetic GA outperforms QPSO.

---

## 📑 Dataset Citation

- **Dataset:** Ryan Madhuwala (RAW), Garudex Labs & Parv Mittal — *New Delhi Traffic Probe Count & Analytics Dataset (2024)*.
