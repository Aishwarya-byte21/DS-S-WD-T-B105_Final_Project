import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# ---------------------------
# CONFIG
# ---------------------------
NUM_ROWS = 10000
START_DATE = datetime(2026, 2, 1)

# Domain → Category mapping
domain_map = {
    "instagram.com": "social",
    "facebook.com": "social",
    "x.com": "social",
    "reddit.com": "social",
    "youtube.com": "video",
    "netflix.com": "video",
    "hotstar.com": "video",
    "primevideo.com": "video",
    "github.com": "learning",
    "kaggle.com": "learning",
    "stackoverflow.com": "learning",
    "coursera.org": "learning",
    "medium.com": "learning",
    "amazon.com": "shopping",
    "flipkart.com": "shopping",
    "wikipedia.org": "other"
}

domains = list(domain_map.keys())

# RAM usage base per category
ram_profile = {
    "social": (500, 900),
    "video": (1200, 2000),
    "learning": (700, 1200),
    "shopping": (600, 1000),
    "other": (400, 800)
}

# ---------------------------
# GENERATE BROWSING DATA
# ---------------------------
rows = []

current_time = START_DATE

for i in range(NUM_ROWS):

    # Random session break (simulate inactivity)
    if random.random() < 0.05:
        current_time += timedelta(minutes=random.randint(20, 60))

    # Normal browsing gap
    else:
        current_time += timedelta(seconds=random.randint(10, 120))

    domain = random.choice(domains)
    category = domain_map[domain]

    rows.append({
        "timestamp": current_time,
        "url": f"https://{domain}/page{random.randint(1,100)}",
        "domain": domain,
        "category": category,
        "browser": random.choice(["chrome", "edge"])
    })

browser_df = pd.DataFrame(rows)

# ---------------------------
# GENERATE RAM DATA
# ---------------------------
ram_rows = []

ram_time = START_DATE

while ram_time <= browser_df['timestamp'].max():

    # Simulate RAM baseline
    base_ram = np.random.uniform(5000, 8000)

    # Pick random category influence
    cat = random.choice(list(ram_profile.keys()))
    browser_ram = np.random.uniform(*ram_profile[cat])

    ram_rows.append({
        "timestamp": ram_time,
        "ram_used_mb": base_ram,
        "ram_available_mb": 16000 - base_ram,
        "browser_ram_mb": browser_ram
    })

    ram_time += timedelta(seconds=5)

ram_df = pd.DataFrame(ram_rows)

# ---------------------------
# SAVE FILES
# ---------------------------
browser_df.to_csv("browsing_history.csv", index=False)
ram_df.to_csv("ram_log.csv", index=False)

# domain map
domain_df = pd.DataFrame(list(domain_map.items()), columns=["domain", "category"])
domain_df.to_csv("domain_category_map.csv", index=False)

print("✅ Synthetic data generated successfully!")
print("Browsing rows:", len(browser_df))
print("RAM rows:", len(ram_df))