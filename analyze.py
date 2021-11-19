import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import numpy as np

import math
import yaml
import pickle
import pprint 
import os
import logging
import sys

import data_loaders
import nets
import losses
import utils

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(787)
torch.cuda.manual_seed(787)

def get_parser():
    """Get parser object."""
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

    parser = ArgumentParser(
        description=__doc__, formatter_class=ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-f",
        "--file",
        dest="filename",
        help="experiment definition file",
        metavar="FILE",
        required=True,
    )
    return parser
            
if __name__ == '__main__':

    args = get_parser().parse_args()

    yaml_filepath = args.filename

    with open(yaml_filepath, 'r') as f:
        cfg = yaml.load(f, yaml.SafeLoader)
    N = cfg['dataset']['n_points']
    dt = (cfg['dataset']['tn'] - cfg['dataset']['t0'])/N
    D = cfg['ae']['net']['latent_dim']


    global savepath

    try:
        run = cfg['n_runs'] - 1
    except KeyError:
        run = 4

    savepath = 'results/{}_d={}w={}z={}det={}lat={}loss={}sigma={}/run{}'.format(
            cfg['head'],
            cfg['dataset']['name'],
            cfg['ae']['net']['width'],
            cfg['ae']['net']['latent_dim'],
            cfg['ae']['net']['add_det'],
            cfg['sde']['type'],
            cfg['ae']['net']['loss'],
            cfg['ae']['net']['sigma_type'],
            run)

    keys = ['sde_mse', 
            'sde_rel', 
            'sde_mse_valid', 
            'sde_rel_valid', 
            'mu_mse', 
            'mu_rel', 
            'mu_mse_valid', 
            'mu_rel_valid']
    gen_keys = ['step_mse','traj_mse']

    with open(os.path.join(savepath,'saved_stats.pkl'), 'rb') as f:
        all_stats = pickle.load(f)
    print(all_stats)

    with open(os.path.join(savepath,'test_stats.pkl'), 'rb') as f:
        all_test_stats = pickle.load(f)

    run_stats = all_stats['runs']
    gen_stats = all_test_stats['runs']

    #super hacky
    stats_array = {k: [dic[k] for dic in run_stats] for k in run_stats[0]}
    gen_stats_array = {k: [dic[k] for dic in gen_stats] for k in gen_stats[0]}


    '''
    for key in keys:
        stat = np.array(stats_array[key])
        print(key)
        print(stat.mean())
        print(stat.std())
    '''

    try:
        sigmas = np.stack(stats_array['sigma_hat'])
        #sigmas = ( sigmas - sigmas.min(1,keepdims=True) ) / ( sigmas.max(1,keepdims=True) - sigmas.min(1,keepdims=True) )
        sigma_m = sigmas.mean(0)
        sigma_s = sigmas.std(0)

        ind = np.arange(sigma_m.shape[0])

        plt.plot(ind, sigma_m)
        #plt.yscale('log')
        plt.ylabel('Eigenvalue')
        plt.xlabel('Rank')
        plt.fill_between(ind, sigma_m-sigma_s, sigma_m+sigma_s, alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(savepath, 'sigmastats.pdf'))
        plt.close('all')

    except KeyError:

        print('No sigma')


    for key in gen_keys:
        stat = torch.Tensor(gen_stats_array[key])
        print(key)
        print(stat.mean().item())
        print(stat.std().item())

    try:
        with open(os.path.join(savepath,'test_stats_crlb.pkl'), 'rb') as f:
            all_crlb_stats = pickle.load(f)

        crlb_stats = all_crlb_stats['runs']
        crlb_keys = ['crlb_mu_mse', 'crlb_sde_mse'] 
        crlb_stats_array = {k: [dic[k] for dic in crlb_stats] for k in crlb_stats[0]}

        for key in crlb_keys:
            stat = torch.Tensor(crlb_stats_array[key])
            print(key)
            print(stat.mean().item())
            print(stat.std().item())
            if key == crlb_keys[1]:
                xt = stat.mean().item()
                xt_s = stat.std().item()
        print('CRLB')
        lower = D/(xt-xt_s)/N/dt
        upper = D/(xt+xt_s)/N/dt
        mean = D/(xt)/N/dt
        print(lower)
        print(mean)
        print(upper)
        
    except:
        print('No crlb data')
