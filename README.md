# Bridge Health AI

An AI system that estimates the condition of a bridge from two kinds of input: **photos of the bridge surface** (analysed by a CNN) and **vibration signals** (analysed by an LSTM). The two damage scores are combined into a single **Health Index from 0 to 100**, shown on a Streamlit dashboard.

**Author:** [Akash Verma](https://github.com/Aakasshhhh) · **Repository:** [github.com/Aakasshhhh/Bridge_Health_AI](https://github.com/Aakasshhhh/Bridge_Health_AI)

---

## Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [How It Works](#how-it-works)
4. [Tech Stack](#tech-stack)
5. [Project Structure](#project-structure)
6. [Dataset](#dataset)
7. [Getting Started](#getting-started)
8. [Training the Models](#training-the-models)
9. [Running the Application](#running-the-application)
10. [API Reference](#api-reference)
11. [Results](#results)
12. [Known Limitations and Roadmap](#known-limitations-and-roadmap)
13. [Author](#author)
14. [License](#license)

---

## Overview

Checking the health of a bridge usually means looking at two things: what the surface looks like (cracks) and how the structure behaves (vibration). This project builds a small end-to-end prototype of that idea:

- A **Convolutional Neural Network (CNN)** looks at a bridge image and estimates the probability that it shows a crack.
- A **Long Short-Term Memory (LSTM)** network looks at a vibration time series and estimates the probability that it shows damage.
- A **fusion step** combines both probabilities into one Health Index and a status label.
- A **Flask API** serves the vibration model, and a **Streamlit dashboard** lets you upload an image and a signal and see the combined result.

## Features

- Binary crack classification on bridge-surface images (150×150 RGB input).
- Binary damage classification on vibration signals using a stacked LSTM.
- Weighted score fusion into a 0–100 Health Index with three status levels.
- REST endpoint (`/predict`) for the vibration model.
- Interactive dashboard with model status indicators, image preview, signal plot and a final assessment.
- Helper scripts for splitting an image dataset and generating simulated vibration data.

## How It Works

```
 Bridge image ──► CNN (crack probability) ───────────────┐
                                                         ├─► Weighted fusion ─► Health Index (0–100) + status
 Vibration CSV ─► Flask API ─► LSTM (damage probability) ┘
```

**Fusion rule** (`src/fusion.py`):

```
combined_damage = 0.6 × image_damage + 0.4 × vibration_damage
health_index    = (1 − combined_damage) × 100
```

| Health Index | Status             |
| ------------ | ------------------ |
| 80 and above | Healthy            |
| 50 to 79     | Needs Monitoring   |
| Below 50     | Critical Condition |

### CNN architecture (`src/train_cnn.py`)

| Layer               | Details                                             |
| ------------------- | --------------------------------------------------- |
| Input               | 150 × 150 × 3, pixels rescaled by 1/255             |
| Conv2D + MaxPooling | 32 filters, 3×3, ReLU, then 2×2 pooling             |
| Conv2D + MaxPooling | 64 filters, 3×3, ReLU, then 2×2 pooling             |
| Flatten, Dense      | 128 units, ReLU                                     |
| Dropout             | 0.5                                                 |
| Output              | 1 unit, sigmoid (probability of "Positive" / crack) |

Training: Adam optimizer, binary cross-entropy loss, batch size 32, 10 epochs.

### LSTM architecture (`src/train_lstm.py`)

| Layer   | Details                                                        |
| ------- | -------------------------------------------------------------- |
| Input   | 100 time steps × 1 feature, standardised with `StandardScaler` |
| LSTM    | 64 units, returns sequences                                    |
| Dropout | 0.3                                                            |
| LSTM    | 32 units                                                       |
| Dropout | 0.3                                                            |
| Dense   | 16 units, ReLU                                                 |
| Output  | 1 unit, sigmoid (probability of damage)                        |

Training: Adam optimizer, binary cross-entropy loss, batch size 32, up to 30 epochs, 80/20 stratified train/test split, 20% of the training data used for validation, early stopping on validation loss (patience 5, best weights restored).

## Tech Stack

| Area           | Tools                              |
| -------------- | ---------------------------------- |
| Language       | Python                             |
| Deep learning  | TensorFlow / Keras                 |
| Image handling | OpenCV, Keras `ImageDataGenerator` |
| Data           | NumPy, Pandas, scikit-learn        |
| API            | Flask                              |
| Dashboard      | Streamlit, Matplotlib              |

## Project Structure

```
Bridge_Health_AI/
├── app/
│   ├── app.py                    # Flask API serving the LSTM model
│   └── app_dashboard.py          # Streamlit dashboard (image + signal → Health Index)
├── src/
│   ├── fusion.py                 # Combines CNN and LSTM scores into the Health Index
│   ├── generate_vibration_data.py# Creates the simulated vibration dataset
│   ├── train_cnn.py              # Trains and saves the image CNN
│   └── train_lstm.py             # Trains and saves the vibration LSTM
├── Split_data.py                 # Moves a random subset of images into train/val folders
├── run_dashboard.bat             # Windows shortcut that starts the API and the dashboard
├── requirements.txt
└── README.md
```

The folders `data/`, `models/` and `outputs/`, and trained model files (`*.h5`), are listed in `.gitignore`, so they are not in the repository. You create them locally by following the steps below.

## Dataset

**Images.** The CNN needs a binary crack dataset with one folder per class, named `Positive` (cracked) and `Negative` (no crack). This project was trained on a public concrete-crack image dataset with 20,000 images per class, split into:

- 16,000 images per class for training (32,000 in total)
- 4,000 images per class for validation (8,000 in total)

Expected layout:

```
data/images/
├── train/
│   ├── Negative/
│   └── Positive/
└── val/
    ├── Negative/
    └── Positive/
```

Keras assigns labels alphabetically, so `Negative` = 0 and `Positive` = 1. The CNN's output is therefore the probability of damage.

**Vibration signals.** There is no public vibration data in this project. `src/generate_vibration_data.py` creates **simulated** accelerometer-style signals:

- 1,000 "healthy" samples: Gaussian noise with a standard deviation of 0.2.
- 1,000 "damaged" samples: Gaussian noise with a standard deviation of 0.8, plus three random spikes.
- Each sample has 100 time steps and a label (0 = healthy, 1 = damaged).

## Getting Started

### Prerequisites

- Python 3.10 or newer (developed on Python 3.13, Windows)
- `pip` and `git`
- About 2 GB of free disk space for the dataset and models

### Installation

**1. Clone the repository**

```bash
git clone https://github.com/Aakasshhhh/Bridge_Health_AI.git
cd Bridge_Health_AI
```

**2. Create and activate a virtual environment**

```bash
python -m venv venv

# Windows (PowerShell)
venv\Scripts\Activate.ps1

# Windows (Command Prompt)
venv\Scripts\activate.bat

# macOS / Linux
source venv/bin/activate
```

**3. Install the dependencies**

```bash
pip install -r requirements.txt
```

**4. Prepare the image dataset**

Download a binary crack image dataset with `Positive` and `Negative` folders, then use `Split_data.py` to create the training and validation folders.

Open `Split_data.py` and edit the three settings at the top:

| Setting                 | Meaning                                                                     |
| ----------------------- | --------------------------------------------------------------------------- |
| `SOURCE_DIR`            | Folder with the downloaded images of one class (for example `...\Positive`) |
| `TRAIN_DESTINATION_DIR` | Target folder, for example `data\images\train\Positive`                     |
| `TRAIN_COUNT`           | Number of images to move (16000 for train, 4000 for validation)             |

Run the script **four times**, changing the settings each time:

| Run | Source class | Destination                  | Count |
| --- | ------------ | ---------------------------- | ----- |
| 1   | Positive     | `data/images/train/Positive` | 16000 |
| 2   | Negative     | `data/images/train/Negative` | 16000 |
| 3   | Positive     | `data/images/val/Positive`   | 4000  |
| 4   | Negative     | `data/images/val/Negative`   | 4000  |

```bash
python Split_data.py
```

The script **moves** files (it does not copy them), so keep a backup of the original download if you want to reuse it.

## Training the Models

### Train the image CNN

Run from the project root:

```bash
python src/train_cnn.py
```

This trains for 10 epochs and saves `models/cnn_model.h5`. On a CPU it takes a few minutes per epoch.

### Generate the vibration data and train the LSTM

These two scripts use relative paths, so run them from inside `src/`:

```bash
cd src
python generate_vibration_data.py
python train_lstm.py
cd ..
```

This creates `data/vibrations/vibration_data.csv`, prints the test accuracy, and saves `models/lstm_bridge_health.h5` and the fitted scaler `models/scaler.pkl`. The API needs both files.

## Running the Application

The dashboard needs the Flask API to be running, so start both.

**Option A: Windows shortcut**

```bash
run_dashboard.bat
```

The script expects the virtual environment folder to be named `venv` in the project root.

**Option B: two terminals (any operating system)**

Terminal 1, the API:

```bash
python app/app.py
```

The API runs at `http://127.0.0.1:5000`.

Terminal 2, the dashboard:

```bash
streamlit run app/app_dashboard.py
```

Streamlit prints a local URL (usually `http://localhost:8501`). Open it in your browser.

**Using the dashboard**

1. Check the sidebar: both the image model and the Flask API should show as online.
2. Upload a **vibration CSV**. The first row is used as one signal of 100 time steps (a `label` column is ignored). To make a test file from the generated data, run this from the project root: `python -c "import pandas as pd; df = pd.read_csv('data/vibrations/vibration_data.csv'); df[df.label == 1].head(1).to_csv('sample_damaged.csv', index=False)"` (use `df.label == 0` for a healthy sample).
3. Upload a **bridge image** (JPG or PNG).
4. The page shows the CNN confidence, the LSTM confidence, the Health Index and the final status.

## API Reference

Base URL: `http://127.0.0.1:5000`

### `GET /`

Health check. Returns a message saying whether the API is running and whether the model loaded.

### `POST /predict`

Request body (JSON):

```json
{ "vibration_data": [0.01, -0.12, 0.08, "... 100 values in total"] }
```

Successful response:

```json
{
  "status": "Success",
  "bridge_condition": "Healthy ✅",
  "health_index": 87.4,
  "damage_confidence": 0.126
}
```

| Status code | Meaning                                            |
| ----------- | -------------------------------------------------- |
| 200         | Prediction returned                                |
| 400         | `vibration_data` missing, or fewer than 100 values |
| 500         | Internal error while predicting                    |
| 503         | The model file could not be loaded                 |

## Results

### Image CNN

Final epoch (epoch 10 of 10), trained on 32,000 images and validated on 8,000:

| Metric   | Training | Validation |
| -------- | -------- | ---------- |
| Accuracy | 99.37%   | 99.24%     |
| Loss     | 0.0247   | 0.0289     |

Validation accuracy stayed between about 97.8% and 99.3% across the 10 epochs. These figures are for a random train/validation split of one public dataset. They show the model learns this dataset well, but they do not prove it works on bridges from other sources.

### Vibration LSTM

`train_lstm.py` prints the test accuracy when it finishes. Because the training data is simulated and the two classes are easy to tell apart, that score reflects the synthetic task and should not be read as real-world performance.

## Known Limitations and Roadmap

- **Simulated vibration data.** The LSTM is trained on generated signals, not measurements from real bridges. Replacing them with real sensor data is the most important next step.
- **Score-level fusion only.** The CNN and LSTM are trained separately and combined with fixed weights (0.6 and 0.4). Learning the weights, or training a joint model, is future work.
- **Evaluation.** Add held-out testing on a second image dataset, plus a confusion matrix, precision and recall.
- **Model format.** Save models in the native `.keras` format instead of legacy `.h5`.
- **Packaging.** Pin dependency versions and add a short demo GIF.

## Author

**Akash Verma**
B.Tech (Artificial Intelligence & Machine Learning), LNCT Group of Colleges, Bhopal

- GitHub: [github.com/Aakasshhhh](https://github.com/Aakasshhhh)
- LinkedIn: [linkedin.com/in/akash-verma-4250022b1](https://www.linkedin.com/in/akash-verma-4250022b1/)
- Project repository: [github.com/Aakasshhhh/Bridge_Health_AI](https://github.com/Aakasshhhh/Bridge_Health_AI)

## License

Copyright (c) 2026 Akash Verma. All rights reserved.

The source code is published for viewing and evaluation. Please do not copy, modify or redistribute it without written permission from the author.
