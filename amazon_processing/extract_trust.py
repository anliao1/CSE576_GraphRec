import scipy.io as sio
import numpy as np
N = 2342   


# ============================================================
# 2. Load full Ciao trust (user_u, user_v)
# ============================================================
ciao = sio.loadmat("ciao_trust.mat")["trust"]

print("Loaded Ciao trust:", ciao.shape)


# ============================================================
# 3. Keep edges where both endpoints < N
# ============================================================
mask = (ciao[:,0] < N) & (ciao[:,1] < N)
trust_sub = ciao[mask]

print("Filtered trust edges:", trust_sub.shape)


# ============================================================
# 4. Save as Amazon’s trust graph
# ============================================================
sio.savemat("amazon_trust.mat", {"trust": trust_sub})

print("Saved amazon_trust.mat")
