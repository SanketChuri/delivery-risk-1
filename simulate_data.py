import pandas as pd
import random

drivers = [f"D{i:02d}" for i in range(1, 41)]
jobs = [f"J{i:03d}" for i in range(1, 81)]

# UK city clusters (so map looks real, not like random dots in the ocean)
cities = [
    (51.5074, -0.1278),   # London
    (53.4808, -2.2426),   # Manchester
    (52.4862, -1.8904),   # Birmingham
    (54.9783, -1.6178),   # Newcastle
    (55.9533, -3.1883),   # Edinburgh
]

data = []

for job in jobs:
    driver = random.choice(drivers)
    scheduled = random.randint(20, 60)
    actual = scheduled + random.randint(-5, 30)

    # pick a base city
    base_lat, base_lon = random.choice(cities)

    # create pickup & drop nearby
    pickup_lat = base_lat + random.uniform(-0.05, 0.05)
    pickup_lon = base_lon + random.uniform(-0.05, 0.05)

    drop_lat = pickup_lat + random.uniform(-0.1, 0.1)
    drop_lon = pickup_lon + random.uniform(-0.1, 0.1)

    data.append({
        "job_id": job,
        "driver_id": driver,
        "scheduled_time": scheduled,
        "actual_time": actual,
        "priority": random.choice(["high", "medium", "low"]),
        "traffic_level": random.choice(["low", "medium", "heavy"]),
        "status": random.choice(["on_route", "delayed", "delivered"]),
        "pickup_lat": pickup_lat,
        "pickup_lon": pickup_lon,
        "drop_lat": drop_lat,
        "drop_lon": drop_lon,
    })

df = pd.DataFrame(data)
df.to_csv("data/orders_with_locations.csv", index=False)

print("Data generated with locations.")