import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Number of data points
n = 500

# Start time
start_time = datetime(2025, 12, 26, 0, 0)

timestamps = [start_time + timedelta(minutes=5*i) for i in range(n)]

# Simulate temperature around 22-26°C with small fluctuations
temperature = np.round(np.random.normal(loc=23.5, scale=1.0, size=n), 1)

# Humidity around 40-55%
humidity = np.round(np.random.normal(loc=45, scale=3, size=n), 0)

# CO2 levels around 400-700 ppm
co2 = np.round(np.random.normal(loc=500, scale=50, size=n), 0)

# Energy consumption roughly correlated with HVAC and occupancy
hvac_on = np.random.choice([0, 1], size=n, p=[0.3, 0.7])
occupancy = np.random.choice([0, 1], size=n, p=[0.5, 0.5])
energy = np.round(4 + hvac_on*1.5 + occupancy*0.5 + np.random.normal(0, 0.2, n), 2)

# Create DataFrame
df = pd.DataFrame({
    "timestamp": timestamps,
    "temperature": temperature,
    "humidity": humidity,
    "co2": co2,
    "energy": energy,
    "hvac_on": hvac_on,
    "occupancy": occupancy
})

# Save to CSV
df.to_csv("bms_data.csv", index=False)
print("bms_data.csv generated with 500 rows")
