import numpy as np
import math


def ilit_to_var_sign(x):
    assert(abs(x) > 0)
    var = abs(x) - 1
    sign = x < 0
    return var, sign


def ilit_to_vlit(x, n_vars):
    assert(x != 0)
    var, sign = ilit_to_var_sign(x)
    if sign: return var + n_vars
    else: return var


class Problem(object):
    def __init__(self, n_vars, iclauses, objective, solution, n_cells_per_batch, all_dimacs, iclauses_offset=None):
        self.n_vars = n_vars
        self.n_lits = 2 * n_vars
        self.n_clauses = len(iclauses)

        self.n_cells = sum(n_cells_per_batch)
        self.n_cells_per_batch = n_cells_per_batch

        self.objective = objective
        self.solution = solution
        self.compute_L_unpack(iclauses)

        # set offset
        self.iclauses_offset = iclauses_offset
        
        self.dimacs = all_dimacs

    def compute_L_unpack(self, iclauses):
        self.L_unpack_indices = np.zeros([self.n_cells, 2], dtype=np.int)
        cell = 0
        for clause_idx, iclause in enumerate(iclauses):
            vlits = [ilit_to_vlit(x, self.n_vars) for x in iclause]
            for vlit in vlits:
                self.L_unpack_indices[cell, :] = [vlit, clause_idx]
                cell += 1
        assert(cell == self.n_cells)


def shift_ilit(x, offset):
    assert(x != 0)
    if x > 0: return x + offset
    else:     return x - offset


def shift_iclauses(iclauses, offset):
    return [[shift_ilit(x, offset) for x in iclause] for iclause in iclauses]


def mk_batch_problem(problems):
    all_iclauses = []
    all_objective = []
    all_solution = []
    all_n_cells = []
    all_dimacs = []
    iclauses_offset = [0]
    offset = 0

    prev_n_vars = None
    for dimacs_name, n_vars, iclauses, objective, solution in problems:
        assert(prev_n_vars is None or n_vars == prev_n_vars)
        prev_n_vars = n_vars

        # multiple problems are merged into one Problem object
        # e.g., [[1, -2, 3], [1, 2], [-1, 3], [-4, -6], [4, 5]];
        # in which [[1, -2, 3], [1, 2], [-1, 3]] belongs to one problem,
        # and [[-4, -6], [4, 5]] belongs to another
        all_iclauses.extend(shift_iclauses(iclauses, offset))
        iclauses_offset.append(len(all_iclauses))
        all_objective.append(objective)
        all_solution.append(solution)
        all_n_cells.append(sum([len(iclause) for iclause in iclauses]))
        all_dimacs.append(dimacs_name)
        offset += n_vars

    return Problem(offset, all_iclauses, all_objective, all_solution, all_n_cells, all_dimacs, iclauses_offset)
