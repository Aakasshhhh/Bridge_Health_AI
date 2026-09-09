import numpy as np
import pandas as pd
import os

DATA_PATH = "../data/vibrations/"
os.makedirs(DATA_PATH, exist_ok=True)

def generate_vibration_data(num_samples=1000, label="Healthy"):
    """
    Simulates vibration sensor readings for bridges.
    Each sample is a time series of accelerometer-like data.
    """
    time_steps = 100  

    if label == "Healthy":
        data = np.random.normal(loc=0, scale=0.2, size=(num_samples, time_steps))
    else:
        data = np.random.normal(loc=0, scale=0.8, size=(num_samples, time_steps))
        for i in range(num_samples):
            spikes = np.random.choice(time_steps, size=3, replace=False)
            data[i, spikes] += np.random.uniform(2, 4)

    return data

def save_data():
    healthy_data = generate_vibration_data(1000, "Healthy")
    damaged_data = generate_vibration_data(1000, "Damaged")

    df_healthy = pd.DataFrame(healthy_data)
    df_healthy["label"] = 0  

    df_damaged = pd.DataFrame(damaged_data)
    df_damaged["label"] = 1  

    df = pd.concat([df_healthy, df_damaged], axis=0).sample(frac=1).reset_index(drop=True)

    file_path = os.path.join(DATA_PATH, "vibration_data.csv")
    df.to_csv(file_path, index=False)
    print(f"✅ Data saved at: {file_path}")
    print(df.head())

if __name__ == "__main__":
    save_data()
