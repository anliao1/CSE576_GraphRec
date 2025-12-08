import numpy as np
import pickle
from scipy.io import loadmat
from sklearn.model_selection import train_test_split

# ============================================================
# Load Amazon N×6 matrix
# ============================================================
rating_mat = np.load("amazon.npy")
# Columns: [user_id, item_id, category, rating, helpfulness, timestamp]

users = rating_mat[:,0].astype(int)
items = rating_mat[:,1].astype(int)
ratings = rating_mat[:,3].astype(float)

num_users = users.max() + 1
num_items = items.max() + 1

print("Users:", num_users)
print("Items:", num_items)
print("Interactions:", rating_mat.shape[0])

# ============================================================
# Build history_u_lists and history_ur_lists
# ============================================================
history_u_lists = [[] for _ in range(num_users)]
history_ur_lists = [[] for _ in range(num_users)]

for u, i, r in zip(users, items, ratings):
    history_u_lists[u].append(i)
    history_ur_lists[u].append(r)

# ============================================================
# Build history_v_lists and history_vr_lists
# ============================================================
history_v_lists = [[] for _ in range(num_items)]
history_vr_lists = [[] for _ in range(num_items)]

for u, i, r in zip(users, items, ratings):
    history_v_lists[i].append(u)
    history_vr_lists[i].append(r)

# ============================================================
# Build ratings_list = [u, v, r] (GraphRec internal format)
# ============================================================
ratings_list = np.vstack([users, items, ratings]).T

# ============================================================
# Train / Val / Test split (80/10/10)
# ============================================================
train_data, test_data = train_test_split(ratings_list, test_size=0.2, random_state=42)
val_data, test_data = train_test_split(test_data, test_size=0.5, random_state=42)

train_u = train_data[:,0].astype(int)
train_v = train_data[:,1].astype(int)
train_r = train_data[:,2].astype(float)

val_u = val_data[:,0].astype(int)
val_v = val_data[:,1].astype(int)
val_r = val_data[:,2].astype(float)

test_u = test_data[:,0].astype(int)
test_v = test_data[:,1].astype(int)
test_r = test_data[:,2].astype(float)

# ============================================================
# Load trust graph for Amazon (previously extracted)
# ============================================================
trust_mat = loadmat("amazon_trust.mat")["trust"]

# Convert trust edges into adjacency list representation
social_adj_lists = {u: set() for u in range(num_users)}
for u, v in trust_mat:
    social_adj_lists[int(u)].add(int(v))
    social_adj_lists[int(v)].add(int(u))

# Convert sets to lists
for u in social_adj_lists:
    social_adj_lists[u] = list(social_adj_lists[u])

# ============================================================
# PACK EVERYTHING INTO GRAPHREC FORMAT (tuple of 15 elements)
# ============================================================
graphrec_tuple = (
    history_u_lists,
    history_ur_lists,
    history_v_lists,
    history_vr_lists,
    train_u,
    train_v,
    train_r,
    val_u,
    val_v,
    val_r,
    test_u,
    test_v,
    test_r,
    social_adj_lists,
    ratings_list
)

# ============================================================
# Save as pickle in GraphRec-compatible format
# ============================================================
with open("amazon.pickle", "wb") as f:
    pickle.dump(graphrec_tuple, f)

print("Saved: amazon.pickle")
