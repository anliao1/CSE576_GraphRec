import pandas as pd
import pickle
import numpy as np
from collections import defaultdict
import os

def preprocess_movielens(ratings_csv, trust_txt, output_file='movielens.pickle', num_target_users=2342):
    print(f"--- Step 1: Processing MovieLens Ratings ({ratings_csv}) ---")
    
    if not os.path.exists(ratings_csv):
        raise FileNotFoundError(f"Could not find {ratings_csv}")
        
    df = pd.read_csv(ratings_csv)
    
    all_ml_users = sorted(df['userId'].unique())
    if len(all_ml_users) < num_target_users:
        print(f"Warning: MovieLens only has {len(all_ml_users)} users. Using all available.")
        target_ml_users = all_ml_users
    else:
        target_ml_users = all_ml_users[:num_target_users]
        
    print(f"Selected {len(target_ml_users)} MovieLens users.")
    
    df = df[df['userId'].isin(target_ml_users)]
    
    ml_user_map = {uid: i for i, uid in enumerate(target_ml_users)}
    num_users = len(target_ml_users)
    
    unique_items = df['movieId'].unique()
    ml_item_map = {iid: i for i, iid in enumerate(unique_items)}
    num_items = len(unique_items)
    
    print(f"Total Unique Items: {num_items}")
    
    unique_ratings = sorted(df['rating'].unique())
    ratings_map = {r: i for i, r in enumerate(unique_ratings)}
    
    print("Building interaction history...")
    
    history_u_lists = {i: [] for i in range(num_users)}
    history_ur_lists = {i: [] for i in range(num_users)}
    history_v_lists = {i: [] for i in range(num_items)}
    history_vr_lists = {i: [] for i in range(num_items)}
    
    train_u, train_v, train_r = [], [], []
    val_u, val_v, val_r = [], [], []      
    test_u, test_v, test_r = [], [], []
    
    # Random split: 80% Train, 10% Validation, 10% Test
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    total_len = len(df)
    train_end = int(total_len * 0.8)
    val_end = int(total_len * 0.9)
    
    for idx, row in df.iterrows():
        uid = ml_user_map[row['userId']]
        iid = ml_item_map[row['movieId']]
        r_val = float(row['rating'])
        r_idx = ratings_map[r_val]
        
        if idx < train_end:
            # Training set (80%)
            train_u.append(uid)
            train_v.append(iid)
            train_r.append(r_val)
            
            history_u_lists[uid].append(iid)
            history_ur_lists[uid].append(r_idx)
            history_v_lists[iid].append(uid)
            history_vr_lists[iid].append(r_idx)
            
        elif idx < val_end:
            # Validation set (10%)
            val_u.append(uid)
            val_v.append(iid)
            val_r.append(r_val)
            
        else:
            # Testing set (10%)
            test_u.append(uid)
            test_v.append(iid)
            test_r.append(r_val)

    print(f"--- Step 2: Processing Trust Network ({trust_txt}) ---")
    
    if not os.path.exists(trust_txt):
        raise FileNotFoundError(f"Could not find {trust_txt}")

    ciao_users = set()
    ciao_connections = [] 
    
    with open(trust_txt, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if '::::' not in line:
                continue
            parts = line.strip().split('::::')
            try:
                u, v = parts[0], parts[1]
                ciao_users.add(u)
                ciao_users.add(v)
                ciao_connections.append((u, v))
            except IndexError:
                continue

    sorted_ciao_users = sorted(list(ciao_users))
    ciao_map = {uid: i for i, uid in enumerate(sorted_ciao_users)}
    
    print(f"Found {len(sorted_ciao_users)} users in trust network.")
    
    social_adj_lists = defaultdict(set)
    
    mapped_edges_count = 0
    for u_raw, v_raw in ciao_connections:
        u_idx = ciao_map[u_raw]
        v_idx = ciao_map[v_raw]
        
        if u_idx < len(target_ml_users) and v_idx < len(target_ml_users):
            social_adj_lists[u_idx].add(v_idx)
            social_adj_lists[v_idx].add(u_idx) 
            mapped_edges_count += 1
            
    print(f"Mapped {mapped_edges_count} social connections.")

    data_to_save = (
        history_u_lists, 
        history_ur_lists, 
        history_v_lists, 
        history_vr_lists, 
        train_u, train_v, train_r, 
        val_u, val_v, val_r,       
        test_u, test_v, test_r, 
        social_adj_lists, 
        ratings_map
    )
    
    print(f"--- Step 3: Saving to {output_file} ---")
    with open(output_file, 'wb') as f:
        pickle.dump(data_to_save, f)
        
    print("Preprocessing Complete!")

if __name__ == "__main__":
    preprocess_movielens(
        ratings_csv='ml-32m/ratings.csv', 
        trust_txt='Ciao/dataset/trustnetwork.txt',  
        output_file='ml-32m/movielens.pickle'
    )
