import os
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import torch

from metrics.calculate_fid import calculate_fid_given_two_paths
from metrics.inception import InceptionV3

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('--exp_name', type=str,
                    help='Experiment name')
parser.add_argument('--batch-size', type=int, default=50,
                    help='Batch size to use')
parser.add_argument('--num-workers', type=int,
                    help=('Number of processes to use for data loading. '
                          'Defaults to `min(8, num_cpus)`'))
parser.add_argument('--device', type=str, default=None,
                    help='Device to use. Like cuda, cuda:0 or cpu')
parser.add_argument('--dims', type=int, default=2048,
                    choices=list(InceptionV3.BLOCK_INDEX_BY_DIM),
                    help=('Dimensionality of Inception features to use. '
                          'By default, uses pool3 features'))
parser.add_argument('--path-A', type=str,
                    help=('Paths to the original images or '
                          'to .npz statistic files. '
                          'Support multiple paths by using:'
                          'path_a1,path_a2,path_a3 ... seperated by ",". '))
parser.add_argument('--path-B', type=str,
                    help=('Paths to the generated images or '
                          'to .npz statistic files. '
                          'Support multiple paths by using:'
                          'path_a1,path_a2,path_a3 ... seperated by ",". '))
parser.add_argument('--blank_patches_list_A', type=str, default=None, required=False, 
                    help='Paths to the lsit of blank patches')
parser.add_argument('--blank_patches_list_B', type=str, default=None, required=False, 
                    help='Paths to the lsit of blank patches')

def main():
    args = parser.parse_args()
    #print('Exp.: ', args.exp_name)
    if args.device is None:
        device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')
    else:
        device = torch.device(args.device)

    if args.num_workers is None:
        num_avail_cpus = len(os.sched_getaffinity(0))
        num_workers = min(num_avail_cpus, 8)
    else:
        num_workers = args.num_workers

    path_As = args.path_A.split(",")
    path_Bs = args.path_B.split(",")

    fid_value = calculate_fid_given_two_paths(path_As, 
                                              path_Bs,
                                              args.batch_size,
                                              device,
                                              args.dims,
                                              num_workers,
                                              args.blank_patches_list_A,
                                              args.blank_patches_list_B)
    print(f'Exp::{args.exp_name}:: || FID: {fid_value:.4f}')

if __name__ == '__main__':
    main()
