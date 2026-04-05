import pandas as pd
import random

drivers = [f"D{i:02d}" for i in range(1, 41)]
jobs = [f"J{i:03d}" for i in range(1, 81)]

data = []

for job in jobs:
    driver = random.choice(drivers)
    scheduled = random.randint(20, 60)
    actual = scheduled + random.randint(-5, 30)

    data.append({
        "job_id": job,
        "driver_id": driver,
        "scheduled_time": scheduled,
        "actual_time": actual,
        "priority": random.choice(["high", "medium", "low"]),
        "traffic_level": random.choice(["low", "medium", "heavy"]),
        "status": random.choice(["on_route", "delivered", "delayed"])
    })

df = pd.DataFrame(data)
df.to_csv("data/orders_with_locations.csv", index=False)

print("Data generated.")