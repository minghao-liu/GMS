import os
import sys
import multiprocessing as mp

# if the generator and MaxHS are not installed from setup.sh, please modify these paths
generator_path = './utils/Power-Law-Random-SAT-Generator/CreateSAT'
maxhs_path = './utils/maxhs/bin/maxhs'


def run_maxhs(instance):
    datafile, solfile = instance
    cmd = "{} -no-printOptions -printSoln -verb=0 {} > {}".format(maxhs_path, datafile, solfile)
    os.system(cmd)
    print("[OK] " + datafile)


def gen(d, k, nv, nc, cnt, sol_flag):
    raw_data_path = 'raw_data_uf' if d == 'u' else 'raw_data_pl'
    raw_data_path2 = os.path.join(raw_data_path, 's{}v{}c{}'.format(k, nv, nc))
    if not os.path.exists(raw_data_path):
        os.mkdir(raw_data_path)
    if not os.path.exists(raw_data_path2):
        os.mkdir(raw_data_path2)
    for i in range(1, cnt+1):
        fname = os.path.join(raw_data_path2, 's{}v{}c{}-{}'.format(k, nv, nc, i))
        cmd = '{} -g {} -v {} -c {} -k {} -s {} -f {}'.format(
            generator_path, d, nv, nc, k, i, fname)
        os.system(cmd)
    if sol_flag:
        if not os.path.exists(os.path.join(raw_data_path, 'sol')):
            os.mkdir(os.path.join(raw_data_path, 'sol'))
        pool = mp.Pool(1)       # modify this according to the number of CPU cores
        instances = []
        for i in range(1, cnt+1):
            datafile = os.path.join(raw_data_path2, 's{}v{}c{}-{}.cnf'.format(k, nv, nc, i))
            solfile = os.path.join(raw_data_path, 'sol', 's{}v{}c{}-{}.cnf.sol'.format(k, nv, nc, i))
            instances.append((datafile, solfile))
        ret = pool.map(run_maxhs, instances)


if __name__ == "__main__":
    d = sys.argv[1]             # distribution
    k = int(sys.argv[2])        # clause size: k-CNF
    nv = int(sys.argv[3])       # number of variables
    nc = int(sys.argv[4])       # number of clauses
    cnt = int(sys.argv[5])      # number of instances
    sol = bool(sys.argv[6])     # compute and store the solutions by running MaxHS
                                # (for training and validation sets only)
    
    if d == "UF":
        gen('u', k, nv, nc, cnt, sol)
    elif d == "PL":
        gen('p', k, nv, nc, cnt, sol)
    else:
        print("distribution not found")
