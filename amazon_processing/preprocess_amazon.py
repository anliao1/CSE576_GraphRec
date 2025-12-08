import re
import numpy as np
import pandas as pd
import pickle

from scipy.io.matlab import savemat
from datetime import datetime

INPUT_FILE = "amazon-meta.txt"

rows = []  # will store tuples of (user_raw, asin_raw, category, rating, helpful, ts)
current_asin = None
current_categories = []
inside_reviews = False

# Regex patterns
asin_re = re.compile(r"^ASIN:\s+(\S+)")
category_re = re.compile(r"^\s+\|(.*)")  # category path
review_re = re.compile(
    r"(\d{4}-\d{1,2}-\d{1,2})\s+cutomer:\s+(\S+)\s+rating:\s+(\d+)\s+votes:\s+(\d+)\s+helpful:\s+(\d+)"
)

def date_to_timestamp(date_str):
    # Convert "YYYY-MM-DD" → UNIX seconds
    try:
        return int(pd.Timestamp(date_str).timestamp())
    except:
        return 0

with open(INPUT_FILE, "r", encoding="utf-8", errors="ignore") as f:
    for line in f:
        line = line.strip()

        # 1. Detect ASIN
        m = asin_re.match(line)
        if m:
            current_asin = m.group(1)
            current_categories = []
            inside_reviews = False
            continue

        # 2. Detect categories
        if line.startswith("categories:"):
            continue

        cat_match = category_re.match(line)
        if cat_match and not inside_reviews:
            current_categories.append(cat_match.group(1))
            continue

        # 3. Detect start of reviews
        if line.startswith("reviews:"):
            inside_reviews = True
            continue

        # 4. Parse review line
        if inside_reviews:
            rm = review_re.search(line)
            if rm:
                date_str = rm.group(1)
                user = rm.group(2)
                rating = int(rm.group(3))
                votes = int(rm.group(4))
                helpful = int(rm.group(5))

                ts = date_to_timestamp(date_str)

                # Use the first category path, or "0" if missing
                category_id = current_categories[0] if current_categories else "0"

                rows.append((user, current_asin, category_id, rating, helpful, ts))

# -------------------------------------------------------------------
# Convert to DataFrame
# -------------------------------------------------------------------

df = pd.DataFrame(rows, columns=[
    "user_raw", "item_raw", "category_raw", "rating", "helpfulness", "timestamp"
])

print("Total review rows extracted:", len(df))

# Remap users, items, categories → contiguous integers
df["user_id"] = df["user_raw"].astype("category").cat.codes
df["item_id"] = df["item_raw"].astype("category").cat.codes
df["category_id"] = df["category_raw"].astype("category").cat.codes

# Build matrix N×6
matrix = df[[
    "user_id", "item_id", "category_id", "rating", "helpfulness", "timestamp"
]].to_numpy(dtype=np.int64)

print("Final matrix shape:", matrix.shape)

np.save("amazon.npy", matrix)


# Build mapping dictionaries
mapping = {
    "rating": matrix,
    "user_mapping": dict(zip(df["user_id"], df["user_raw"])),
    "item_mapping": dict(zip(df["item_id"], df["item_raw"])),
    "category_mapping": dict(zip(df["category_id"], df["category_raw"]))
}


