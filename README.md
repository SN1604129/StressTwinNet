# StressTwinNet
**A Digital Twin of the Human Stress System**

StressTwinNet is a research-oriented framework for modeling, predicting,
and simulating human stress dynamics using physiological signals.

## Key Ideas
- Learn latent stress states from multi-modal wearable data
- Model temporal stress dynamics (baseline → stress → recovery)
- Simulate intervention-aware futures
- Generate human-readable explanations using GenAI

## Dataset
We use the **WESAD (Wearable Stress and Affect Detection)** dataset:
- ECG / PPG (BVP)
- EDA
- Respiration
- Labels: baseline, stress, recovery

Raw data should be placed in:
data/raw/WESAD/

## Project Structure
- `models/` → Encoder & dynamics models
- `training/` → Training scripts
- `evaluation/` → Evaluation scripts
- `utils/` → Data loading, preprocessing, windowing
- `genai/` → Explanation layer
- `notebooks/` → Visualisation & experiments

## Research Goal
This project aims to build a **stress digital twin** capable of:
1. Encoding physiological states
2. Predicting future stress trajectories
3. Simulating interventions
4. Explaining predictions in natural language
EOF
