import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from mlp import MLP


# GMS-N: using Node-splitting CVIG
class GMS(nn.Module):
    def __init__(self, args):
        super(GMS, self).__init__()
        self.args = args
        self.dim = args.dim

        self.init_ts = torch.ones(1)
        self.init_ts.requires_grad = False

        self.L_init = nn.Linear(1, args.dim)
        self.C_init = nn.Linear(1, args.dim)

        self.L_msg = MLP(self.args.dim, self.args.dim, self.args.dim)
        self.C_msg = MLP(self.args.dim, self.args.dim, self.args.dim)

        self.L_update = nn.LSTM(self.args.dim * 2, self.args.dim)
        self.C_update = nn.LSTM(self.args.dim, self.args.dim)

        self.var_vote = MLP(self.args.dim * 2, self.args.dim, 1)

    def forward(self, problem):
        n_vars = problem.n_vars
        n_lits = problem.n_lits
        n_clauses = problem.n_clauses
        n_probs = len(problem.objective)
        n_offsets = problem.iclauses_offset

        ts_L_unpack_indices = torch.Tensor(problem.L_unpack_indices).t().long()

        init_ts = self.init_ts.cuda()
        L_init = self.L_init(init_ts).view(1, 1, -1)
        L_init = L_init.repeat(1, n_lits, 1)
        C_init = self.C_init(init_ts).view(1, 1, -1)
        C_init = C_init.repeat(1, n_clauses, 1)

        L_state = (L_init, torch.zeros(1, n_lits, self.args.dim).cuda())
        C_state = (C_init, torch.zeros(1, n_clauses, self.args.dim).cuda())
        L_unpack = torch.sparse.FloatTensor(ts_L_unpack_indices, torch.ones(problem.n_cells),
                                            torch.Size([n_lits, n_clauses])).cuda()

        for _ in range(self.args.n_rounds):
            L_hidden = L_state[0].squeeze(0)
            L_pre_msg = self.L_msg(L_hidden)
            LC_msg = torch.sparse.mm(L_unpack.t(), L_pre_msg)
            _, C_state = self.C_update(LC_msg.unsqueeze(0), C_state)
            
            C_hidden = C_state[0].squeeze(0)
            C_pre_msg = self.C_msg(C_hidden)
            CL_msg = torch.sparse.mm(L_unpack, C_pre_msg)
            _, L_state = self.L_update(torch.cat(
                [CL_msg, self.flip(L_state[0].squeeze(0), n_vars)], dim=1).unsqueeze(0), L_state)

        logits = L_state[0].squeeze(0)
        clauses = C_state[0].squeeze(0)

        x = torch.cat((logits[:n_vars, :], logits[n_vars:, :]), dim=1)
        var_vote = self.var_vote(x)

        return var_vote

    def flip(self, msg, n_vars):
        return torch.cat([msg[n_vars:2 * n_vars, :], msg[:n_vars, :]], dim=0)
