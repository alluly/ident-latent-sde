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
import setup

'''
#SECTION VAE Helpers
'''

device = torch.device('cpu')
torch.manual_seed(787)
torch.cuda.manual_seed(787)
torch.backends.cudnn.deterministic

def crlb(ae, 
        mu, 
        sigma, 
        dt, 
        train_data, 
        val_data,
        test_data, 
        optimizer, 
        scheduler, 
        n_epochs, 
        data_params,
        **kwargs):

    '''
    Returns the CRLB for mu and the model's performance
    '''

    global savepath

    train_dataset = train_data.dataset.dataset
    test_dataset  = test_data.dataset.dataset

    stats = {}

    ae.eval()
    ae = ae.to(device)
    mu = mu.to(device)
    sigma = sigma.to(device)

    gbm = False
    if 'gbm' in savepath:
        gbm = True

    # Compute the mapping over the training data since we want to see the fit within the whole time series
    if  train_dataset.im_type == 'dwass':
        frames = torch.Tensor(train_dataset.frames[:data_params['batch_size']]).float().to(device)
        ts     = torch.Tensor(train_dataset.ts[:data_params['batch_size']]).float().to(device)
    else:
        frames = torch.Tensor(train_dataset.frames).float().to(device)
        ts     = torch.Tensor(train_dataset.ts).float().to(device)

    # sample just to get the latent z 
    _, _, _, _, q_mu, _, _, _, z = ae.step(frames, 
                    ts, 
                    dt, 
                    mu, 
                    sigma)

    # compare the estimated mu to the true mu with the affine map Q
    d = z.shape[1]
    current_run = train_dataset.xt_orig[1:z.shape[0]]

    scale = (train_dataset.xt_orig[:].max() - train_dataset.xt_orig[:].min()) 
    if gbm:
        scale = (np.log(train_dataset.xt_orig[:]).max() - np.log(train_dataset.xt_orig[:]).min())

    q_max = q_mu.max()
    q_min = q_mu.min()

    global loss_type

    z = z[1:]
    q_mu = q_mu[1:]

    if data_params['affine']:

        q_scaled = ((z - q_min ) / (q_max - q_min) * (scale)).detach().cpu().numpy() 
        transformed_xt, Q, b, sde_mse, sde_rel = utils.calc_affine(
                current_run,
                q_scaled,
                savepath, 
                affine=data_params['affine'],
                gbm=gbm)

        if 'dna' in savepath:
            pass

        mu_mse, mu_rel, mu_crlb = utils.compare_mu2(mu, 
                q_mu, 
                ts, 
                Q, 
                b, 
                dt, 
                train_dataset, 
                os.path.join(savepath,'c2.pdf'),
                affine=data_params['affine'],
                loss_type=loss_type, 
                oracle=None)

        if 'dna' in savepath:
            test_frames = test_dataset.frames
            test_xt = test_dataset.xt_orig

            lat = (test_xt - b) @ np.linalg.inv(Q)

            frames_hat = ae.decode(torch.tensor(lat).float())
            skips = []
            offset = 1

            [skips.append(test_frames[row*offset].unsqueeze(0)) for row in range(8)]
            [skips.append(frames_hat[row*offset].unsqueeze(0)/frames_hat[row*8].detach().max()) for row in range(8)]
            [skips.append(torch.cat((test_frames[row*offset].mean(0,keepdims=True).unsqueeze(0), test_frames[row*offset].mean(0, keepdims=True).unsqueeze(0), frames_hat[row*offset].mean(0, keepdims=True).unsqueeze(0)),1)) for row in range(8)]
            comp_grid = torchvision.utils.make_grid(torch.cat(skips), pad_value=1, normalize=True, nrow=8)
            plots_list = [comp_grid.detach().cpu().numpy().transpose((1,2,0))]
            plot_titles = ['Comparison']
            utils.plot_subplots(plots_list, plot_titles, os.path.join(savepath, 'coord_skip.png'))

    else:

        if loss_type == 'exact': 
            q_scaled = ((z - q_min ) / (q_max - q_min) * (scale)).detach().cpu().numpy() 
        else:
            q_scaled = q_mu.detach().cpu().numpy() / (scale) 

        transformed_xt, Q, b, sde_mse, sde_rel = utils.calc_affine(
                current_run, 
                q_scaled, 
                savepath, 
                affine=data_params['affine'],
                gbm=gbm)

        q_scaled = ((z - q_min ) / (q_max - q_min) * (scale))

        mu_mse, mu_rel, mu_crlb = utils.compare_mu2(
                mu, 
                q_mu, 
                ts, 
                Q, 
                b, 
                dt, 
                train_dataset, 
                os.path.join(savepath,'c2.pdf'),
                affine=data_params['affine'],
                loss_type=loss_type, 
                oracle=None)

    stats['crlb_sde_mse']     = sde_mse.copy()
    stats['crlb_sde_mse_rel'] = sde_rel.copy()
    stats['crlb_mu_mse']      = mu_mse.copy()
    stats['crlb_mu_rel']      = mu_rel.copy()
    stats['crlb_mu']          = mu_crlb.copy()

    return stats

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

    global savepath
    all_stats = {'runs':[]}

    all_s = []

    try:
        n_runs = cfg['n_runs']
    except KeyError:
        n_runs = 5

    for run in range(n_runs):

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

        print(savepath)

        try:
            tries = cfg['n_tries']
        except KeyError:
            tries = 1

        if tries == 1:
            cfg['ae']['path']  = 'ae_best_val.pth'
            cfg['sde']['path'] = 'mu_best_val.pth'
        else:
            cfg['ae']['path']  = 'ae_best_val_bt.pth'
            cfg['sde']['path'] = 'mu_best_val_bt.pth'

        global loss_type
        loss_type = cfg['ae']['net']['loss']

        pp = pprint.PrettyPrinter(indent=4)

        pp.pprint(cfg)

        init = setup.setup(cfg, savepath)

        if 'small' in savepath:
            s,_ = torch.sort(init['sigma'], descending=True)
            s = s / s.sum()
            all_s.append(s.clone().detach().cpu())
            if run == n_runs-1:

                all_s = torch.stack(all_s)

                plt.plot(torch.arange(all_s.shape[1]), all_s.mean(0))
                plt.fill_between(torch.arange(all_s.shape[1]), all_s.mean(0) - all_s.std(0), all_s.mean(0)+all_s.std(0), alpha= 0.5)
                plt.savefig(os.path.join(savepath,'svd.pdf'))
                plt.close('all')
            stats = 0 
        else:
            stats = crlb(**init)

        all_stats['runs'].append(stats)

        print(stats)

    with open(os.path.join(savepath,'test_stats_crlb.pkl'), 'wb') as f:
        pickle.dump(all_stats, f)
