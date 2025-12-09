import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable
import pickle
import numpy as np
import time
import random
from collections import defaultdict
from UV_Encoders import UV_Encoder
from UV_Aggregators import UV_Aggregator
from Social_Encoders import Social_Encoder
from Social_Aggregators import Social_Aggregator
import torch.nn.functional as F
import torch.utils.data
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from math import sqrt
import datetime
import argparse
import os

"""
GraphRec: Graph Neural Networks for Social Recommendation. 
Wenqi Fan, Yao Ma, Qing Li, Yuan He, Eric Zhao, Jiliang Tang, and Dawei Yin. 
In Proceedings of the 28th International Conference on World Wide Web (WWW), 2019. Preprint[https://arxiv.org/abs/1902.07243]

If you use this code, please cite our paper:
```
@inproceedings{fan2019graph,
  title={Graph Neural Networks for Social Recommendation},
  author={Fan, Wenqi and Ma, Yao and Li, Qing and He, Yuan and Zhao, Eric and Tang, Jiliang and Yin, Dawei},
  booktitle={WWW},
  year={2019}
}
```

"""


class GraphRec(nn.Module):

    def __init__(self, enc_u, enc_v_history, r2e,
                 user_hist_bins=None, attr_dim=0, fuse_type='gate'):
        super(GraphRec, self).__init__()
        self.enc_u = enc_u
        self.enc_v_history = enc_v_history
        self.embed_dim = enc_u.embed_dim

        self.w_ur1 = nn.Linear(self.embed_dim, self.embed_dim)
        self.w_ur2 = nn.Linear(self.embed_dim, self.embed_dim)
        self.w_vr1 = nn.Linear(self.embed_dim, self.embed_dim)
        self.w_vr2 = nn.Linear(self.embed_dim, self.embed_dim)
        self.w_uv1 = nn.Linear(self.embed_dim * 2, self.embed_dim)
        self.w_uv2 = nn.Linear(self.embed_dim, 16)
        self.w_uv3 = nn.Linear(16, 1)
        self.r2e = r2e
        self.bn1 = nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        self.bn2 = nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        self.bn3 = nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        self.bn4 = nn.BatchNorm1d(16, momentum=0.5)
        self.criterion = nn.MSELoss()

        # ----- user attribute embeddings (optional) -----
        self.user_hist_bins = user_hist_bins  # tensor [num_users] or None
        self.attr_dim = attr_dim
        self.fuse_type = fuse_type

        if (user_hist_bins is not None) and (attr_dim > 0):
            self.use_attr = True
            # 6 bins (0..5)
            self.E_hist = nn.Embedding(6, attr_dim)
            if fuse_type == 'concat':
                # [u_emb || a_emb] -> d
                self.fuse_fc = nn.Linear(self.embed_dim + attr_dim, self.embed_dim)
            else:
                # gated fusion: u + g * W_a a
                self.gate_fc = nn.Linear(self.embed_dim + attr_dim, self.embed_dim)
                self.attr_fc = nn.Linear(attr_dim, self.embed_dim)
        else:
            self.use_attr = False

    def forward(self, nodes_u, nodes_v):
        # base GraphRec encodings
        embeds_u = self.enc_u(nodes_u)          # [B, d]
        embeds_v = self.enc_v_history(nodes_v)  # [B, d]

        # ----- fuse user attributes if enabled -----
        if self.use_attr:
            # user_hist_bins: [num_users]; nodes_u: [B]
            bins = self.user_hist_bins[nodes_u]     # [B]
            a = self.E_hist(bins)                   # [B, attr_dim]

            if self.fuse_type == 'concat':
                concat = torch.cat([embeds_u, a], dim=-1)
                embeds_u = torch.tanh(self.fuse_fc(concat))
            else:
                concat = torch.cat([embeds_u, a], dim=-1)
                g = torch.sigmoid(self.gate_fc(concat))
                a_proj = self.attr_fc(a)
                embeds_u = embeds_u + g * a_proj

        # ----- original GraphRec MLP -----
        x_u = F.relu(self.bn1(self.w_ur1(embeds_u)))
        x_u = F.dropout(x_u, training=self.training)
        x_u = self.w_ur2(x_u)

        x_v = F.relu(self.bn2(self.w_vr1(embeds_v)))
        x_v = F.dropout(x_v, training=self.training)
        x_v = self.w_vr2(x_v)

        x_uv = torch.cat((x_u, x_v), 1)
        x = F.relu(self.bn3(self.w_uv1(x_uv)))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.bn4(self.w_uv2(x)))
        x = F.dropout(x, training=self.training)
        scores = self.w_uv3(x)
        return scores.squeeze()

    def loss(self, nodes_u, nodes_v, labels_list):
        scores = self.forward(nodes_u, nodes_v)
        return self.criterion(scores, labels_list)



def train(model, device, train_loader, optimizer, epoch, best_rmse, best_mae):
    model.train()
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        batch_nodes_u, batch_nodes_v, labels_list = data
        optimizer.zero_grad()
        loss = model.loss(batch_nodes_u.to(device), batch_nodes_v.to(device), labels_list.to(device))
        loss.backward(retain_graph=True)
        optimizer.step()
        running_loss += loss.item()
        if i % 100 == 0:
            print('[%d, %5d] loss: %.3f, The best rmse/mae: %.6f / %.6f' % (
                epoch, i, running_loss / 100, best_rmse, best_mae))
            running_loss = 0.0
    return 0


def test(model, device, test_loader):
    model.eval()
    tmp_pred = []
    target = []
    with torch.no_grad():
        for test_u, test_v, tmp_target in test_loader:
            test_u, test_v, tmp_target = test_u.to(device), test_v.to(device), tmp_target.to(device)
            val_output = model.forward(test_u, test_v)
            tmp_pred.append(list(val_output.data.cpu().numpy()))
            target.append(list(tmp_target.data.cpu().numpy()))
    tmp_pred = np.array(sum(tmp_pred, []))
    target = np.array(sum(target, []))
    expected_rmse = sqrt(mean_squared_error(tmp_pred, target))
    mae = mean_absolute_error(tmp_pred, target)
    return expected_rmse, mae


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Social Recommendation: GraphRec model')
    parser.add_argument('--batch_size', type=int, default=128, metavar='N', help='input batch size for training')
    parser.add_argument('--embed_dim', type=int, default=64, metavar='N', help='embedding size')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate')
    parser.add_argument('--test_batch_size', type=int, default=1000, metavar='N', help='input batch size for testing')
    parser.add_argument('--epochs', type=int, default=100, metavar='N', help='number of epochs to train')

    parser.add_argument('--dataset',
                        choices=['toy', 'ciao', 'epinion'],
                        default='epinion',
                        help='which dataset to use')

    parser.add_argument('--data_pickle',
                        default='',
                        help='explicit path to .pickle file (overrides --dataset)')

    parser.add_argument('--use_attr',
                        action='store_true',
                        help='enable user attribute embeddings (GraphRec-Attr)')

    parser.add_argument('--fuse',
                        choices=['concat', 'gate'],
                        default='gate',
                        help='how to fuse user attributes with user embedding')

    parser.add_argument('--attr_dim',
                        type=int,
                        default=8,
                        help='dimension of user attribute embedding')

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    use_cuda = False
    if torch.cuda.is_available():
        use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Used CUDA: {use_cuda}")

    embed_dim = args.embed_dim

    # ---- pick which pickle to load ----
    if args.data_pickle:
        path_data = args.data_pickle
    else:
        if args.dataset == 'toy':
            path_data = './data/toy_dataset.pickle'       # if you have the toy pickle
        elif args.dataset == 'ciao':
            path_data = './data/ciao_full.pickle'
        else:  # epinion
            path_data = './data/epinion_full.pickle'

    print(f"Loading data from: {path_data}")
    with open(path_data, 'rb') as data_file:
        (history_u_lists, history_ur_lists,
         history_v_lists, history_vr_lists,
         train_u, train_v, train_r,
         val_u,   val_v,   val_r,
         test_u,  test_v,  test_r,
         social_adj_lists,
         ratings_list) = pickle.load(data_file)

    # for i in range(20):
    #     print(train_u[i], train_v[i], train_r[i])
    """
    ## toy dataset 
    history_u_lists, history_ur_lists:  user's purchased history (item set in training set), and his/her rating score (dict)
    history_v_lists, history_vr_lists:  user set (in training set) who have interacted with the item, and rating score (dict)
    
    train_u, train_v, train_r: training_set (user, item, rating)
    test_u, test_v, test_r: testing set (user, item, rating)
    
    # please add the validation set
    
    social_adj_lists: user's connected neighborhoods
    ratings_list: rating value from 0.5 to 4.0 (8 opinion embeddings)
    """

    trainset = torch.utils.data.TensorDataset(torch.LongTensor(train_u), torch.LongTensor(train_v),
                                              torch.FloatTensor(train_r))
    valset = torch.utils.data.TensorDataset(torch.LongTensor(val_u), torch.LongTensor(val_v),
                                            torch.FloatTensor(val_r))
    testset = torch.utils.data.TensorDataset(torch.LongTensor(test_u), torch.LongTensor(test_v),
                                             torch.FloatTensor(test_r))
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(valset, batch_size=args.test_batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=True)
    num_users = history_u_lists.__len__()
    num_items = history_v_lists.__len__()
    num_ratings = ratings_list.__len__()

    u2e = nn.Embedding(num_users, embed_dim).to(device)
    v2e = nn.Embedding(num_items, embed_dim).to(device)
    r2e = nn.Embedding(num_ratings, embed_dim).to(device)
        
        # ---- derived user attributes: history length bins ----
    # history_u_lists is a list/dict indexed by user_id -> list of item_ids
    hist_lengths = []
    for u in range(num_users):
        # works for list or dict with 0..num_users-1 keys
        hist_lengths.append(len(history_u_lists[u]))

    def bin_len(L):
        if L == 0:
            return 0
        elif L <= 5:
            return 1
        elif L <= 10:
            return 2
        elif L <= 20:
            return 3
        elif L <= 50:
            return 4
        else:
            return 5

    hist_bins = torch.tensor([bin_len(L) for L in hist_lengths],
                             dtype=torch.long,
                             device=device)


    # user feature
    # features: item * rating
    agg_u_history = UV_Aggregator(v2e, r2e, u2e, embed_dim, cuda=device, uv=True)
    enc_u_history = UV_Encoder(u2e, embed_dim, history_u_lists, history_ur_lists, agg_u_history, cuda=device, uv=True)
    # neighobrs
    agg_u_social = Social_Aggregator(lambda nodes: enc_u_history(nodes).t(), u2e, embed_dim, cuda=device)
    enc_u = Social_Encoder(lambda nodes: enc_u_history(nodes).t(), embed_dim, social_adj_lists, agg_u_social,
                           base_model=enc_u_history, cuda=device)

    # item feature: user * rating
    agg_v_history = UV_Aggregator(v2e, r2e, u2e, embed_dim, cuda=device, uv=False)
    enc_v_history = UV_Encoder(v2e, embed_dim, history_v_lists, history_vr_lists, agg_v_history, cuda=device, uv=False)

    # model
    if args.use_attr:
        graphrec = GraphRec(enc_u, enc_v_history, r2e,
                            user_hist_bins=hist_bins,
                            attr_dim=args.attr_dim,
                            fuse_type=args.fuse).to(device)
    else:
        graphrec = GraphRec(enc_u, enc_v_history, r2e).to(device)

    optimizer = torch.optim.RMSprop(graphrec.parameters(), lr=args.lr, alpha=0.9)

    best_rmse = 9999.0
    best_mae = 9999.0
    endure_count = 0

    for epoch in range(1, args.epochs + 1):

        train(graphrec, device, train_loader, optimizer, epoch, best_rmse, best_mae)
        val_rmse, val_mae = test(graphrec, device, val_loader)
        # Use validation set to tune the hyper-parameters and for early stopping
        if best_rmse > val_rmse:
            best_rmse = val_rmse
            best_mae = val_mae
            endure_count = 0
        else:
            endure_count += 1
        print("Validation - rmse: %.4f, mae:%.4f " % (val_rmse, val_mae))

        if endure_count > 5:
            break

    # Final evaluation on test set
    test_rmse, test_mae = test(graphrec, device, test_loader)
    print("\nFinal Test Results - rmse: %.4f, mae:%.4f " % (test_rmse, test_mae))


if __name__ == "__main__":
    main()
