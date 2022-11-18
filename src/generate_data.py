import argparse
import datetime
import multiprocessing as mp
import os
import pickle
from data_maker_maxsat import data_maker


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_data_path', action='store', type=str, default='raw_data')
    parser.add_argument('--data_selector', action='store', type=str, default='')
    parser.add_argument('--data_type', action='store', type=str, choices=['train', 'val', 'test'], default='train')
    parser.add_argument('--batch_size', action='store', type=int, default=20000)
    parser.add_argument('--one', action='store_true', default=False, help='only one instance in a batch')
    parser.add_argument('--test_mode', action='store_true', default=False, help='generate test data (without solution)')
    parser.add_argument('--log_path', action='store', type=str, default='')
    parser.add_argument('--prefix', action='store', type=str, default='', help='prefix of the data file name')
    args = parser.parse_args()

    batches = data_maker(args)

    data_save_path = os.path.join(os.getcwd(), 'data', args.data_type)
    if not os.path.exists(data_save_path):
        os.system('mkdir -p {}'.format(data_save_path))
    dump_file = '{}_bs{}_nb{}.pkl'.format(args.prefix, args.batch_size, len(batches))

    print('writing {} batches to {}...'.format(len(batches), dump_file))
    with open(os.path.join(data_save_path, dump_file), 'wb') as f_dump:
        pickle.dump(batches, f_dump)
