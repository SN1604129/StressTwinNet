# StressTwinNet

StressTwinNet is a research-grade digital twin of the human stress system built using multimodal physiological data and deep learning.

The project models latent stress states, temporal stress dynamics, and intervention-aware behavior using signals such as ECG, EDA, RESP, and PPG from the WESAD dataset.

The long-term goal is to develop a publishable, PhD-level system capable of:

- Learning latent physiological stress representations
- Modeling temporal stress transitions
- Simulating intervention effects
- Generating human-readable explanations (GenAI integration)

---

## 📂 Project Structure


StressTwinNet/
│
├── data/ # Raw and processed datasets (not tracked in git)
├── models/ # Model architectures
├── training/ # Training pipelines
├── evaluation/ # Metrics and evaluation scripts
├── genai/ # Explanation generation modules (later phase)
├── notebooks/ # Experiments and analysis
├── utils/ # Data loading and preprocessing
├── checkpoints/ # Saved model weights
├── main.py # Entry point
├── requirements.txt
└── README.md


---

## ⚙️ Environment Setup


python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt


---

## 📊 Dataset

Primary dataset: **WESAD (Wearable Stress and Affect Detection)**

Raw data should be placed in:


data/raw/WESAD/


Dataset files are intentionally excluded from version control.

---

## 🚀 Development Roadmap

- [x] Project initialization
- [ ] Data loader implementation
- [ ] Signal preprocessing pipeline
- [ ] Windowing & dataset builder
- [ ] Latent state encoder
- [ ] Temporal dynamics model
- [ ] Intervention-aware simulation
- [ ] Explainable stress reports (GenAI)

---

## 🎯 Research Direction

StressTwinNet is designed for high-quality academic research and potential Q1 publication, focusing on interpretable and dynamic modeling of physiological stress systems.
