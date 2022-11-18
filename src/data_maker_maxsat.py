import copy
import datetime
import gc
import math
import os
import random
import sys
import time
import ast
import argparse
import pickle
import numpy as np
from mk_problem import mk_batch_problem


def parse_data_list(data_selector_str, is_inc):
    data_from_files = []
    data_selector = ast.literal_eval(data_selector_str)
    for data_item in data_selector:
        s, v, c, i, j = data_item
        for _ in range(i, j+1):
            dimacs_file = os.path.join("s{}v{}c{}".format(s, v, c), "s{}v{}c{}-{}.cnf".format(s, v, c, _))
            sol_file = os.path.join('sol', "s{}v{}c{}-{}.cnf.maxhs.sol".format(s, v, c, _))
            data_from_files.append((dimacs_file, sol_file))
    return data_from_files, False


def parse_dimacs(filename, is_wcnf=False):
    with open(filename, 'r') as f:
        xs = f.readlines()

    index = 0
    while xs[index][0] == 'c':
        index += 1

    header = xs[index].strip().split()
    assert (header[0] == 'p')

    clauses = [list(map(int, x.strip().split()[:-1])) for x in xs[index + 1:]]
    if is_wcnf:
        for clause in clauses:
            clause.pop(0)

    return header, clauses


def parse_solution(filename, test_mode):
    with open(filename, 'r') as f:
        xs = f.readlines()

    optimum, solution = None, None
    for x in xs:
        mark = x.strip().split()[0]
        if mark == 'o':
            optimum = int(x.strip().split()[1])
        elif mark == 'v':
            solution = list(x.strip().split()[1])
            solution = list(map(int, solution))
        else:
            continue

    assert (optimum is not None)
    if not test_mode:
        assert (solution is not None)

    return optimum, solution


def check_solution(n_vars, iclauses, objective, solution):
    unsat_count = 0
    for iclause in iclauses:
        sat = 0
        for var in iclause:
            if var > 0 and solution[var - 1] == 1:
                sat = 1
                break
            if var < 0 and solution[-var - 1] == 0:
                sat = 1
                break
        if sat == 0:
            unsat_count += 1
    if unsat_count != objective:
        return False
    return True


def generate(args):
    data_path = os.path.join(os.getcwd(), args.raw_data_path)
    
    log_file = None
    if args.log_path != '':
        log_file = open(args.log_path, 'a+')
    
    data_files, is_wcnf = parse_data_list(args.data_selector, args.test_mode)
    
    problems = []
    for i, (dimacs_name, sol_name) in enumerate(data_files):
        header, iclauses = parse_dimacs(os.path.join(data_path, dimacs_name), is_wcnf)
        n_vars, n_clauses = int(header[2]), int(header[3])
        objective, solution = parse_solution(os.path.join(data_path, sol_name), args.test_mode)
        if not args.test_mode:
            assert (check_solution(n_vars, iclauses, objective, solution))

        # because MaxHS may ignore the variables not appeared in the clauses,
        # here we assign them to 'False', so that len(solution) == n_vars
        if solution is not None and len(solution) < n_vars:
            solution += [0] * (n_vars - len(solution))

        problems.append([dimacs_name, n_vars, iclauses, objective, solution])

        if (i + 1) % 200 == 0:
            log_str = '\tgenerate {} instances'.format(i)
            if log_file is not None:
                print(log_str, flush=True, file=log_file)
            print(log_str, flush=True)
    
    return problems


def transform(args, problems):
    random.shuffle(problems)

    batches = []
    instances = []
    acc_n_nodes = 0

    log_file = None
    if args.log_path != '':
        log_file = open(args.log_path, 'a+')

    for p in problems:
        dimacs_name, n_vars, iclauses, objective, solution = p
        n_clauses = len(iclauses)
        n_cells = sum([len(iclause) for iclause in iclauses])

        cur_n_nodes = 2 * n_vars + n_clauses
        if cur_n_nodes > args.batch_size:
            log_str = '[WARNING] skip a too large problem: {}'.format(dimacs_name)
            if log_file is not None:
                print(log_str, flush=True, file=log_file)
            print(log_str, flush=True)
            continue

        batch_ready = False
        if args.one and len(instances) > 0:
            batch_ready = True
        elif not args.one and acc_n_nodes + cur_n_nodes > args.batch_size:
            batch_ready = True

        if batch_ready:
            batches.append(mk_batch_problem(instances))
            log_str = 'batch {} done ({} problems)'.format(len(batches), len(instances))
            if log_file is not None:
                print(log_str, flush=True, file=log_file)
            print(log_str, flush=True)

            del instances[:]
            acc_n_nodes = 0

        instances.append([dimacs_name, n_vars, iclauses, objective, solution])
        acc_n_nodes += cur_n_nodes

    if len(instances) > 0:
        batches.append(mk_batch_problem(instances))
        log_str = 'batch {} done ({} problems)'.format(len(batches), len(instances))
        if log_file is not None:
            print(log_str, flush=True, file=log_file)
        print(log_str, flush=True)
        del instances[:]

    return batches


def data_maker(args):
    problems = generate(args)
    batches = transform(args, problems)
    return batches
