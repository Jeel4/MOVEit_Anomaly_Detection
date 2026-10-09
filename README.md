# AI-Driven Anomaly Detection Framework for MOVEit Transfer

**Group 30 — Cyber Security Case Study**

## 1. Project Overview

This project implements an AI-driven anomaly detection framework inspired by the MOVEit Transfer cyberattack of May 2023 (CVE-2023-34362).

The framework identifies unusual session behaviour by learning patterns from benign traffic and assigning anomaly scores to sessions. It combines multiple machine learning approaches to investigate whether behavioural anomalies can help detect attack-like activity without relying exclusively on predefined attack signatures.

**Important:** This is a synthetic proof-of-concept implementation. It has not been validated against real MOVEit victim logs or a live MOVEit server.

## 2. Objectives

- Study the MOVEit Transfer zero-day exploitation incident.
- Generate synthetic benign and attack-like session data.
- Extract and analyse 15 behavioural features.
- Implement and compare multiple anomaly detection algorithms.
- Evaluate the models using ROC-AUC, PR-AUC, precision, recall, F1-score and false-positive rate.
- Investigate a hybrid anomaly detection approach based on score fusion.

## 3. Technologies Used

- Python 3
- NumPy
- pandas
- scikit-learn
- SciPy
- Matplotlib

## 4. Models Implemented

1. Isolation Forest
2. Autoencoder using an MLP reconstruction model
3. Local Outlier Factor (LOF)
4. One-Class Support Vector Machine (baseline)
5. Hybrid model combining Isolation Forest, Autoencoder and LOF

## 5. Dataset

The project uses reproducibly generated synthetic session data rather than real attack logs.

| Dataset split | Number of sessions |
|---|---:|
| Benign training | 20,000 |
| Benign validation | 4,000 |
| Benign testing | 8,000 |
| Attack testing | 400 |
| Behavioural features | 15 |

The attack-like sessions simulate behavioural patterns associated with the MOVEit incident. Some benign sessions also simulate high-volume clients and noisy scanners to make detection more challenging.

## 6. Installation

### Prerequisites

- Python 3.10 or a compatible Python 3 version
- pip
- Windows PowerShell, Command Prompt, or a compatible terminal

### Step 1: Clone the repository

Replace the URL below with the actual GitHub repository URL after the repository has been created.

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd G30_MOVEit_Anomaly_Detection
```

### Step 2: Create a virtual environment

```bash
python -m venv .venv
```

### Step 3: Activate the environment

For Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, run the following in the current terminal:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
```

### Step 4: Install dependencies

```bash
python -m pip install -r requirements.txt
```

## 7. Running the Project

### Quick demonstration

```bash
python main.py --quick
```

This runs the quick experiment and generates model evaluation results and plots.

### Full experiment

```bash
python main.py
```

The full experiment uses the dataset configuration described above and may take longer to execute.

## 8. Generated Outputs

The program saves its results in the `outputs/` directory.

| File | Description |
|---|---|
| `metrics.json` | Dataset configuration, threshold and model metrics |
| `model_comparison.csv` | Comparison of the evaluated models |
| `scored_test_sessions.csv` | Test sessions with generated scores and predictions |
| `roc_curves.png` | ROC curves for the evaluated models |
| `hybrid_score_distribution.png` | Distribution of benign and attack-like hybrid scores |
| `feature_shift.png` | Differences in mean feature values between attack-like and benign sessions |

## 9. Experimental Results

Results from the full experiment:

| Model | ROC-AUC | PR-AUC | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|---:|
| Isolation Forest | 0.9906 | 0.9288 | 0.8443 | 0.8675 | 0.8557 |
| Autoencoder | 0.9611 | 0.8373 | 0.8234 | 0.7225 | 0.7696 |
| LOF | 0.9898 | 0.9384 | 0.8152 | 0.8825 | 0.8475 |
| One-Class SVM | 0.9879 | 0.9352 | 0.8794 | 0.8750 | 0.8772 |
| Hybrid | 0.9870 | 0.9248 | 0.8325 | 0.8700 | 0.8509 |

Isolation Forest achieved the highest ROC-AUC, while One-Class SVM achieved the highest F1-score in this run. The hybrid approach showed competitive performance but did not outperform every individual model.

The hybrid alert threshold for this run was approximately **0.9726**.

These metrics describe one seeded experiment on synthetic data. They should not be interpreted as validated real-world detection performance.

## 10. Limitations

- The evaluation uses synthetic rather than real MOVEit traffic.
- The experiment represents a proof of concept, not a production-ready security product.
- Results are based on a single seeded run.
- Real-world traffic variation, concept drift and adversarial evasion require further investigation.
- Deployment would require integration with actual telemetry, security monitoring and incident response systems.

## 11. Future Scope

- Evaluate the framework on representative real or red-team-generated managed file-transfer logs.
- Conduct repeated experiments with different random seeds.
- Investigate sequence-based anomaly detection.
- Improve explainability and alert prioritisation.
- Explore integration with security orchestration and automated response systems.

## 12. Group Members

- Harsh Mate
- Ishan Sinha
- Jeel Patel
- Jiya Tyagi

## 13. Repository

GitHub: 

## 14. Disclaimer

This project is an academic cyber security case study. It simulates attack-like behaviour for anomaly detection research and does not perform exploitation of MOVEit Transfer or validate detection against real victim systems.