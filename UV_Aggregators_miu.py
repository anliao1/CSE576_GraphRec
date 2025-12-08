import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import random
from Attention import Attention


class UV_Aggregator(nn.Module):
    """
    item and user aggregator: for aggregating embeddings of neighbors (item/user aggreagator).
    """

    def __init__(self, v2e, r2e, u2e, embed_dim, cuda="cpu", uv=True):
        super(UV_Aggregator, self).__init__()
        self.uv = uv
        self.v2e = v2e
        self.r2e = r2e
        self.u2e = u2e
        self.device = cuda
        self.embed_dim = embed_dim
        self.w_r1 = nn.Linear(self.embed_dim * 2, self.embed_dim)
        self.w_r2 = nn.Linear(self.embed_dim, self.embed_dim)
        self.att = Attention(self.embed_dim)
        self.alpha_gate = nn.Linear(self.embed_dim, self.embed_dim)#update
        self.mu_gate = nn.Linear(self.embed_dim * 2, self.embed_dim)#update
    def forward(self, nodes, history_uv, history_r):

        embed_matrix = torch.empty(len(history_uv), self.embed_dim, dtype=torch.float).to(self.device)

        for i in range(len(history_uv)):
            history = history_uv[i]
            num_histroy_item = len(history)
            tmp_label = history_r[i]

            if self.uv == True:
                # user component
                e_uv = self.v2e.weight[history]
                uv_rep = self.u2e.weight[nodes[i]]
            else:
                # item component
                e_uv = self.u2e.weight[history]
                uv_rep = self.v2e.weight[nodes[i]]

            e_r = self.r2e.weight[tmp_label]
            x = torch.cat((e_uv, e_r), 1)
            x = F.relu(self.w_r1(x))
            o_history = F.relu(self.w_r2(x))

#            context = torch.sigmoid(self.alpha_gate(uv_rep))      # uodate
#            dynamic_input = context * e_uv + (1 - context) * e_r  # update
#            x_dyn = F.relu(self.w_r1(torch.cat((dynamic_input, e_r), dim=1)))#update
#            o_history = F.relu(self.w_r2(x_dyn))#update

            if not self.uv:
                q_rep = uv_rep.unsqueeze(0).repeat(num_histroy_item, 1)
                gate_input = torch.cat([o_history, q_rep], dim=1)
                g_jt = torch.sigmoid(self.mu_gate(gate_input))
                o_history = g_jt * o_history + (1 - g_jt) * q_rep

            att_w = self.att(o_history, uv_rep, num_histroy_item)
            att_history = torch.mm(o_history.t(), att_w)
            att_history = att_history.t()

            embed_matrix[i] = att_history
        to_feats = embed_matrix
        return to_feats
