import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import random
from Attention import Attention


class Social_Aggregator(nn.Module):
    """
    Social Aggregator: for aggregating embeddings of social neighbors.
    """

    def __init__(self, features, u2e, embed_dim, cuda="cpu"):
        super(Social_Aggregator, self).__init__()

        self.features = features
        self.device = cuda
        self.u2e = u2e
        self.embed_dim = embed_dim
        self.att = Attention(self.embed_dim)
        self.gate_layer = nn.Linear(self.embed_dim * 2, self.embed_dim) # update
    def forward(self, nodes, to_neighs):
        embed_matrix = torch.empty(len(nodes), self.embed_dim, dtype=torch.float).to(self.device)
        for i in range(len(nodes)):
            tmp_adj = to_neighs[i]
            num_neighs = len(tmp_adj)
             
            e_u = self.u2e.weight[list(tmp_adj)]
            u_rep = self.u2e.weight[nodes[i]]

            u_rep_expanded = u_rep.unsqueeze(0).repeat(num_neighs, 1) #update
            gate_input = torch.cat([e_u, u_rep_expanded], dim=1) #update
            g_ij = torch.sigmoid(self.gate_layer(gate_input)) #update
            e_u_gated = g_ij * e_u + (1 - g_ij) * u_rep_expanded #update
            att_w = self.att(e_u_gated, u_rep, num_neighs) #update:e_u to e_u_gate
            att_history = torch.mm(e_u_gated.t(), att_w).t() #update to e_u_gate

            embed_matrix[i] = att_history
        to_feats = embed_matrix

        return to_feats
