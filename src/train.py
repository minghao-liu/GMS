import argparse
import pickle
import os
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from config import parser

args = parser.parse_args()
if args.model == 'GMS_N':
    from gms_n import GMS
else:
    from gms_e import GMS


def check_maxsat(L_unpack_indices, clauses_offset, n_clauses, var_offset, n_vars, solution):
    clauses_answers = torch.zeros(n_clauses)
    for L_C_index in L_unpack_indices:
        L_index = L_C_index[0]
        sign = True
        if L_index >= var_offset:
            L_index = L_index - var_offset
            sign = False
        if L_index >= n_vars:
            L_index = L_index % n_vars
        clauses_answers[L_C_index[1] - clauses_offset] += solution[L_index] if sign else int(not solution[L_index])
    num_unsat_clauses = torch.sum(clauses_answers == 0).item()
    return num_unsat_clauses


def evaluate(probs, solutions):
    solutions = solutions.detach().cpu().numpy()
    n_vars = probs.n_vars // len(probs.objective)

    objective_diff = []
    prev_idx, sol_idx = 0, 0
    for idx in range(len(probs.objective)):
        n_cell = probs.n_cells_per_batch[idx] + prev_idx
        n_clauses = probs.iclauses_offset[idx+1] - probs.iclauses_offset[idx]
        find_objective = check_maxsat(probs.L_unpack_indices[prev_idx:n_cell], probs.iclauses_offset[idx],
                                   n_clauses, probs.n_vars, n_vars, solutions[sol_idx:sol_idx + n_vars])
        objective_diff.append(find_objective - probs.objective[idx])
        sol_idx += n_vars
        prev_idx = n_cell
    return objective_diff


##### START #####
net = GMS(args)
net = net.cuda()

task_name = args.task_name + '_n' + str(args.n_vars) + '_ep' + str(args.epochs) + '_nr' + 
            str(args.n_rounds) + '_d' + str(args.dim)
log_file = open(os.path.join(args.log_dir, task_name + '.log'), 'a+')
detail_log_file = open(os.path.join(args.log_dir, task_name + '_detail.log'), 'a+')

train, val = None, None
with open(os.path.join(args.data_dir, 'train', args.train_file), 'rb') as f:
    train = pickle.load(f)
with open(os.path.join(args.data_dir, 'val', args.val_file), 'rb') as f:
    val = pickle.load(f)

loss_fn = nn.BCELoss()
optim = optim.Adam(net.parameters(), lr=args.learning_rate, weight_decay=1e-10)
sigmoid = nn.Sigmoid()

best_var_acc = 0.0
best_obj_diff = 999999.0
start_epoch = 0

print('number of training batches:', len(train), flush=True)
print('number of training batches:', len(train), file=log_file, flush=True)
print('number of training batches:', len(train), file=detail_log_file, flush=True)
print('number of validation batches:', len(val), flush=True)
print('number of validation batches:', len(val), file=log_file, flush=True)
print('number of validation batches:', len(val), file=detail_log_file, flush=True)

if args.restore is not None:
    print('restore from', args.restore, flush=True)
    print('restore from', args.restore, file=log_file, flush=True)
    print('restore from', args.restore, file=detail_log_file, flush=True)
    model = torch.load(args.restore)
    start_epoch = model['epoch']
    best_var_acc = model['var_acc']
    best_obj_diff = model['obj_diff']
    net.load_state_dict(model['state_dict'])

for epoch in range(start_epoch, args.epochs):
    #
    print('[%d/%d epoch] previous best: var acc %.3f; obj diff %.3f' % (
        epoch + 1, args.epochs, best_var_acc, best_obj_diff))
    print('[%d/%d epoch] previous best: var acc %.3f; obj diff %.3f' % (
        epoch + 1, args.epochs, best_var_acc, best_obj_diff), file=log_file, flush=True)
    print('[%d/%d epoch] previous best: var acc %.3f; obj diff %.3f' % (
        epoch + 1, args.epochs, best_var_acc, best_obj_diff), file=detail_log_file, flush=True)

    train_bar = tqdm(train)
    VTP, VTN, VFN, VFP = torch.zeros(1).long(), torch.zeros(1).long(), torch.zeros(1).long(), torch.zeros(1).long()
    obj_diff = 0.0
    total_obj_diff = []
    
    net.train()
    for _, prob in enumerate(train_bar):
        optim.zero_grad()
        var_outputs = net(prob)
        var_outputs = sigmoid(var_outputs).view(-1)
        var_target = torch.Tensor(sum(prob.solution, [])).cuda().float()  # 2d-list -> 1d-list
        
        valid_var_outputs = []
        for index in range(len(prob.objective)):
            valid_var_outputs.append(var_outputs[int(index * var_outputs.shape[0] / len(prob.objective)):int(
                (index + 1) * var_outputs.shape[0] / len(prob.objective))])
        valid_var_outputs = torch.cat(valid_var_outputs, dim=0)
        
        var_loss = loss_fn(valid_var_outputs, var_target)
        tot_loss = var_loss
        desc = 'tot loss: %.4f, var loss: %.4f ' % (tot_loss.item(), var_loss.item())
        
        tot_loss.backward()
        optim.step()

        var_preds = torch.where(valid_var_outputs > 0.5, torch.ones(valid_var_outputs.shape).cuda(),
                                torch.zeros(valid_var_outputs.shape).cuda())
        cur_obj_diff = evaluate(prob, var_preds)
        total_obj_diff.extend(cur_obj_diff)
        obj_diff = np.mean(total_obj_diff)

        VTP += (var_preds.eq(1) & var_target.eq(1)).cpu().sum()
        VTN += (var_preds.eq(0) & var_target.eq(0)).cpu().sum()
        VFN += (var_preds.eq(0) & var_target.eq(1)).cpu().sum()
        VFP += (var_preds.eq(1) & var_target.eq(0)).cpu().sum()
        VTOT = VTP + VTN + VFN + VFP

        desc += 'obj_diff: %.3f ' % obj_diff
        desc += 'var acc: %.3f, TP: %.3f, TN: %.3f, FN: %.3f, FP: %.3f; ' % (
        (VTP.item() + VTN.item()) * 1.0 / VTOT.item(), VTP.item() * 1.0 / VTOT.item(),
        VTN.item() * 1.0 / VTOT.item(), VFN.item() * 1.0 / VTOT.item(), VFP.item() * 1.0 / VTOT.item())

        if (_ + 1) % 100 == 0:
            print(desc, file=detail_log_file, flush=True)
    print(desc, file=log_file, flush=True)

    val_bar = tqdm(val)
    VTP, VTN, VFN, VFP = torch.zeros(1).long(), torch.zeros(1).long(), torch.zeros(1).long(), torch.zeros(1).long()
    obj_diff = 0.0
    total_obj_diff = []
    
    net.eval()
    for _, prob in enumerate(val_bar):
        optim.zero_grad()
        var_outputs = net(prob)
        var_outputs = sigmoid(var_outputs).view(-1)
        var_target = torch.Tensor(sum(prob.solution, [])).cuda().float()  # 2d-list -> 1d-list

        valid_var_outputs = []
        for index in range(len(prob.objective)):
            valid_var_outputs.append(var_outputs[int(index * var_outputs.shape[0] / len(prob.objective)):int(
                (index + 1) * var_outputs.shape[0] / len(prob.objective))])
        valid_var_outputs = torch.cat(valid_var_outputs, dim=0)

        var_loss = loss_fn(valid_var_outputs, var_target)
        tot_loss = var_loss
        desc = 'tot loss: %.4f, var loss: %.4f ' % (tot_loss.item(), var_loss.item())
        
        var_preds = torch.where(valid_var_outputs > 0.5, torch.ones(valid_var_outputs.shape).cuda(),
                                torch.zeros(valid_var_outputs.shape).cuda())
        cur_obj_diff = evaluate(prob, var_preds)
        total_obj_diff.extend(cur_obj_diff)
        obj_diff = np.mean(total_obj_diff)

        VTP += (var_preds.eq(1) & var_target.eq(1)).cpu().sum()
        VTN += (var_preds.eq(0) & var_target.eq(0)).cpu().sum()
        VFN += (var_preds.eq(0) & var_target.eq(1)).cpu().sum()
        VFP += (var_preds.eq(1) & var_target.eq(0)).cpu().sum()
        VTOT = VTP + VTN + VFN + VFP

        desc += 'obj_diff: %.3f ' % obj_diff
        desc += 'var acc: %.3f, TP: %.3f, TN: %.3f, FN: %.3f, FP: %.3f; ' % (
         (VTP.item() + VTN.item()) * 1.0 / VTOT.item(),
         VTP.item() * 1.0 / VTOT.item(), VTN.item() * 1.0 / VTOT.item(),
         VFN.item() * 1.0 / VTOT.item(), VFP.item() * 1.0 / VTOT.item())

        if (_ + 1) % 100 == 0:
            print(desc, file=detail_log_file, flush=True)
    print(desc, file=log_file, flush=True)

    var_acc = (VTP.item() + VTN.item()) * 1.0 / VTOT.item()
    torch.save(
        {'epoch': epoch + 1, 'var_acc': var_acc, 'obj_diff': obj_diff, 'state_dict': net.state_dict()},
        os.path.join(args.model_dir, task_name + '.pth.tar'))
    # if var_acc >= best_var_acc:
    #     best_var_acc = var_acc
    #     torch.save(
    #         {'epoch': epoch + 1, 'var_acc': var_acc, 'obj_diff': obj_diff, 'state_dict': net.state_dict()},
    #         os.path.join(args.model_dir, task_name + '_best_var.pth.tar'))
    if obj_diff <= best_obj_diff:
        best_obj_diff = obj_diff
        torch.save(
            {'epoch': epoch + 1, 'var_acc': var_acc, 'obj_diff': obj_diff, 'state_dict': net.state_dict()},
            os.path.join(args.model_dir, task_name + 'best_obj.pth.tar'))
