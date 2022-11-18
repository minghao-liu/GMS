import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--task-name', type=str, default='gms', help='task name')
parser.add_argument('--model', type=str, default='GMS_N', choices=['GMS_N', 'GMS_E'])

parser.add_argument('--dim', type=int, default=128, help='dimension of embeddings')
parser.add_argument('--n_rounds', type=int, default=26, help='number of GNN layers')
parser.add_argument('--epochs', type=int, default=1)
parser.add_argument('--learning_rate', type=float, default=0.00001)

parser.add_argument('--model-dir', type=str, default='model/', help='model directory')
parser.add_argument('--data-dir', type=str, default='data/', help='data directory')
parser.add_argument('--log-dir', type=str, default='log/', help='log directory')
parser.add_argument('--restore', type=str, default=None, help='recover the training process from a model')

parser.add_argument('--train-file', type=str, default=None, help='training data file directory')
parser.add_argument('--val-file', type=str, default=None, help='validation data file directory')
parser.add_argument('--test-file', type=str, default=None, help='testing data file directory')

parser.add_argument('--n_vars', action='store', type=int, default=60)
