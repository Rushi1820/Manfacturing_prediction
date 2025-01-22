import pandas as pd
import random
import numpy as np

def generate_synthetic_data(num_records=100000):
    data = {
        "Machine_ID": [f"M-{random.randint(100, 999)}" for _ in range(num_records)],
        "Temperature": [round(random.uniform(50.0, 120.0), 2) for _ in range(num_records)],
        "Run_Time": [random.randint(100, 10000) for _ in range(num_records)],
        "Downtime_Flag": [random.choice([0, 1]) for _ in range(num_records)]
    }
    return pd.DataFrame(data)

# Generate 1000 rows of synthetic data
df = generate_synthetic_data()

# Save to CSV (optional)
df.to_csv("synthetic_machine_data.csv", index=False)

print(df.head())
