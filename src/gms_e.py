import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import math
from mlp import MLP


# GMS-E: using Edge-splitting CVIG
class GMS(nn.Module):
    def __init__(self, args):
        super(GMS, self).__init__()
        self.args = args
        self.dim = args.dim

        self.init_ts = torch.ones(1)
        self.init_ts.requires_grad = False

        self.L_init = nn.Linear(1, args.dim)
        self.C_init = nn.Linear(1, args.dim)

        self.L_msg_pos = MLP(self.args.dim, self.args.dim, self.args.dim)
        self.L_msg_neg = MLP(self.args.dim, self.args.dim, self.args.dim)
        self.C_msg_pos = MLP(self.args.dim, self.args.dim, self.args.dim)
        self.C_msg_neg = MLP(self.args.dim, self.args.dim, self.args.dim)

        self.L_update = nn.LSTM(self.args.dim, self.args.dim)
        self.C_update = nn.LSTM(self.args.dim, self.args.dim)

        self.var_vote = MLP(self.args.dim, self.args.dim, 1)

    def forward(self, problem):
        n_vars = problem.n_vars
        n_clauses = problem.n_clauses
        n_probs = len(problem.objective)
        n_offsets = problem.iclauses_offset

        # split the positive and negative edges
        L_unpack_indices_pos = []
        L_unpack_indices_neg = []
        for lit, clu in problem.L_unpack_indices:
            if lit >= n_vars:
                L_unpack_indices_neg.append([lit - n_vars, clu])
            else:
                L_unpack_indices_pos.append([lit, clu])
        L_unpack_indices_pos = np.array(L_unpack_indices_pos, dtype=int)
        L_unpack_indices_neg = np.array(L_unpack_indices_neg, dtype=int)
        ts_L_unpack_indices_pos = torch.Tensor(L_unpack_indices_pos).t().long()
        ts_L_unpack_indices_neg = torch.Tensor(L_unpack_indices_neg).t().long()

        init_ts = self.init_ts.cuda()
        L_init = self.L_init(init_ts).view(1, 1, -1)
        L_init = L_init.repeat(1, n_vars, 1)
        C_init = self.C_init(init_ts).view(1, 1, -1)
        C_init = C_init.repeat(1, n_clauses, 1)

        L_state = (L_init, torch.zeros(1, n_vars, self.args.dim).cuda())
        C_state = (C_init, torch.zeros(1, n_clauses, self.args.dim).cuda())
        L_unpack_pos = torch.sparse.FloatTensor(ts_L_unpack_indices_pos, torch.ones(len(L_unpack_indices_pos)),
                                                torch.Size([n_vars, n_clauses])).cuda()
        L_unpack_neg = torch.sparse.FloatTensor(ts_L_unpack_indices_neg, torch.ones(len(L_unpack_indices_neg)),
                                                torch.Size([n_vars, n_clauses])).cuda()

        for _ in range(self.args.n_rounds):
            L_hidden = L_state[0].squeeze(0)
            L_pre_msg_pos = self.L_msg_pos(L_hidden)
            L_pre_msg_neg = self.L_msg_neg(L_hidden)
            LC_msg_pos = torch.sparse.mm(L_unpack_pos.t(), L_pre_msg_pos)
            LC_msg_neg = torch.sparse.mm(L_unpack_neg.t(), L_pre_msg_neg)
            _, C_state = self.C_update(
                LC_msg_pos.unsqueeze(0) + LC_msg_neg.unsqueeze(0), C_state)
            
            C_hidden = C_state[0].squeeze(0)
            C_pre_msg_pos = self.C_msg_pos(C_hidden)
            C_pre_msg_neg = self.C_msg_neg(C_hidden)
            CL_msg_pos = torch.sparse.mm(L_unpack_pos, C_pre_msg_pos)
            CL_msg_neg = torch.sparse.mm(L_unpack_neg, C_pre_msg_neg)
            _, L_state = self.L_update(
                CL_msg_pos.unsqueeze(0) + CL_msg_neg.unsqueeze(0), L_state)

        logits = L_state[0].squeeze(0)
        clauses = C_state[0].squeeze(0)

        x = logits
        var_vote = self.var_vote(x)

        return var_vote
