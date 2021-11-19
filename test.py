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
from setup import setup

import cv2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(787)
torch.cuda.manual_seed(787)

'''
#SECTION VAE Helpers
'''

def sample_n_frames(init_frames, ts, dt, ae, mu, sigma, n_generate=64):

    # function to generate the next n frames conditioned on the input
    with torch.no_grad():

        # Get the latent variables
        q_mu, q_sigma, det = ae.encode(init_frames)
        _, z, _ = ae.get_increments(q_mu, q_sigma)

        z_samples = torch.zeros((n_generate, z.shape[1]))
        z = z[-1,:].unsqueeze(0)

        # sample in z according to the learned SDE
        for i in range(n_generate):
            z_n = ae.get_next_z(z, ts[-1].unsqueeze(0) + i*dt, dt, mu, sigma)
            z_samples[i,:] = z_n.clone()
            z = z_n

        global savepath

        plots_list = [z_samples.detach().cpu().numpy()]
        plot_titles = ['Latent Traj']
        utils.plot_subplots(plots_list, plot_titles, os.path.join(savepath, 'latent_traj.png'), plot_type='plot', axis=True)

        conditional_frame = init_frames[0].unsqueeze(0).repeat(z_samples.size(0),1,1,1)

        if det is not None:
            in_z = torch.cat((z_samples.to(device), det[-1].repeat(z_samples.shape[0],1)), dim = 1)
        else:
            in_z = z_samples.to(device)

        frames = ae.decode(in_z, x=conditional_frame)

        return z_samples, frames#torch.cat((init_frames, frames), dim=0)

def sample_bridge(a, b, dt, n=100):

    '''
    Generates a brownian bridge between a and b.
    '''
    t = torch.linspace(0, 1, n)

    dW = torch.randn_like(t) * np.sqrt(dt)
    W = dW.cumsum(0)
    W[0] = 0
    W = W + a

    BB = (W - t * (W[-1] - b)).unsqueeze(1)

    return BB, t



def test(ae, 
        mu, 
        sigma, 
        dt, 
        test_data, 
        optimizer, 
        scheduler, 
        n_epochs, 
        data_params,
        **kwargs):

    global savepath

    test_dataset = test_data.dataset.dataset

    plot_train = True
    l2_small = True
    l2_small_valid = True

    losses_train = []
    losses_valid = []

    stats = {'traj_mse':0, 'step_mse':0}

    ae.eval()
    if type(mu) == nets.MLP or type(mu) == nets.Lin or type(mu) == nets.Well:
        mu.eval()

    for idx, (frames, ts) in enumerate(test_data):
        if len(frames.shape) > 2:
            utils.save_gif(frames[:64].detach().cpu(), os.path.join(savepath, 'orig_data_test.gif'))

        # transfer the data to the device
        # the rest is boilerplate
        frames = frames.float().to(device)
        ts     = ts.float().to(device)

        # plot latent trajectories
        # plot next frame reconstructions
        min_mse = 100000

        kl_loss, l2_loss,\
        frames_hat, mu_hat, q_mu, sigma_hat_full, q_sigma_full, inc, z = ae.step(frames, ts, dt, mu, sigma)

        if q_mu.shape[1] == 1:
            _, imin = torch.tensor(test_dataset.xt_orig).min(0)
            _, imax = torch.tensor(test_dataset.xt_orig).max(0)

            ranks = torch.tensor(test_dataset.xt_orig).argsort(0)
            sorted_frames = torch.tensor(test_dataset.frames[ranks[:,0]])

            fmin = torch.tensor(test_dataset.frames[imin])
            fmax = torch.tensor(test_dataset.frames[imax])

            b_frames_in = torch.stack((fmin,fmax)).float().cuda(0)
            z_  = (ae.encode(b_frames_in)[0]).detach().cpu()

            m = q_mu.mean()
            s = q_mu.std()

            nbridge = 16

            bb, _  = sample_bridge(z_[0], z_[1], dt, n=nbridge)

            sort_idx = np.round(np.linspace(0, sorted_frames.shape[0] - 1, nbridge)).astype(int)
            true_frames = sorted_frames[sort_idx]

            bridge_frames = ae.decode(bb.squeeze(0).to('cuda:0'))
            bridge_grid = torchvision.utils.make_grid(bridge_frames.detach().cpu(), pad_value=1, normalize=True)
            true_grid   = torchvision.utils.make_grid(torch.cat((true_frames.detach().cpu(),bridge_frames.cpu().detach())), pad_value=1, normalize=True)
            plots_list = [bridge_grid.cpu().numpy().transpose((1,2,0)), bb.squeeze().cpu()]
            plot_titles = ['Bridge', 'Points']
            utils.plot_subplots(plots_list, plot_titles, os.path.join(savepath, 'bridge-path.png'), plot_type=['imshow', 'plot'])
            plots_list = [bridge_grid.cpu().numpy().transpose((1,2,0))]
            plot_titles = ['']
            utils.plot_subplots(plots_list, plot_titles, os.path.join(savepath, 'bridge.png'), plot_type='imshow',figsize=(8,3))

            plots_list = [true_grid.cpu().numpy().transpose((1,2,0))]
            plot_titles = ['']
            utils.plot_subplots(plots_list, plot_titles, os.path.join(savepath, 'true_bridge.png'), plot_type='imshow',figsize=(8,4))

        mse_next = F.mse_loss(frames_hat, frames)
        stats['step_mse'] = mse_next

        for sample_iter in range(1):
            z_samples, sampled_frames = sample_n_frames(frames[:2], ts[:2], dt, ae.eval(), mu, sigma, n_generate=100) 
            mse = F.mse_loss(sampled_frames, frames)
            if mse < min_mse:
                min_mse = mse
                best_sample = sampled_frames
            _, sampled_frames2 = sample_n_frames(frames[:2], ts[:2], dt, ae.eval(), mu, sigma*2) 
        stats['traj_mse'] = min_mse

        with open(os.path.join(savepath, 'test_frames.pkl'),'wb') as f:
            import pickle
            pickle.dump(frames, f)

        with open(os.path.join(savepath, 'latent_test_frames.pkl'),'wb') as f:
            import pickle
            pickle.dump(best_sample, f)

        # create the image grids
        im_grid_hat_single = torchvision.utils.make_grid(frames_hat[:64].detach().cpu(), pad_value=1, normalize=True)
        im_grid_hat = torchvision.utils.make_grid(sampled_frames[:64].detach().cpu(), pad_value=1, normalize=True)
        im_grid     = torchvision.utils.make_grid(frames[:64].detach().cpu(), pad_value=1, normalize=True)

        plt.imshow(im_grid.permute(1,2,0))
        plt.savefig(os.path.join(savepath,'testing_images_only.png'))
        plt.close('all')

        odd_rows = []
        skips = []
        skip_step = []

        for row in range(4):
            odd_rows.append(frames[row*8:(row+1)*8])
            odd_rows.append(sampled_frames[row*8:(row+1)*8])

        offset = 8 
        if 'light' in savepath or 'heavy' in savepath:
            original = torch.tensor(test_data.dataset.dataset.orig_frames[-frames.shape[0]:]).cuda(0)
            mse_next = F.mse_loss(frames_hat, original)
            stats['step_mse'] = mse_next
            [skips.append(original[row*offset].unsqueeze(0)) for row in range(8)]
            [skip_step.append(original[row*offset].unsqueeze(0)) for row in range(8)]

        [skips.append(frames[row*8].unsqueeze(0)) for row in range(8)]
        [skips.append(best_sample[row*8].unsqueeze(0)/best_sample[row*8].detach().max()) for row in range(8)]

        [skip_step.append(frames[row*8].unsqueeze(0).detach()) for row in range(8)]
        [skip_step.append((frames_hat[row*8].unsqueeze(0).detach() - frames_hat[row*8].detach().min())/(frames_hat[row*8].detach().max() - frames_hat[row*8].detach().min())) for row in range(8)]
        #[skip_step.append(frames_hat[row*8].unsqueeze(0).detach()/frames_hat[row*8].detach().max()) for row in range(8)]

        if 'dna' in savepath and 'z16' not in savepath:
            from scipy.ndimage import maximum_filter
            import skimage.filters

            centers = np.array([cv2.minMaxLoc(maximum_filter(skimage.filters.gaussian(frames[dx].mean(0).cpu(), sigma=1), size=(10,10)))[-1] for dx in range(frames.shape[0])])
            r = torch.zeros_like(frames)


        with open(os.path.join(savepath, 'sde_samples.pkl'), 'wb') as f:
            import pickle
            pickle.dump(frames_hat,f)

        if len(frames.shape) > 2:

            comp_grid = torchvision.utils.make_grid(torch.cat(skip_step), pad_value=1, normalize=False, nrow=8)
            plots_list = [comp_grid.cpu().numpy().transpose((1,2,0))]
            plot_titles = ['']
            utils.plot_subplots(plots_list, plot_titles, os.path.join(savepath, 'test_skips_single.png'),figsize=(10,3))

            comp_grid = torchvision.utils.make_grid(torch.cat(skips), pad_value=1, normalize=True, nrow=8)
            plots_list = [comp_grid.cpu().numpy().transpose((1,2,0))]
            plot_titles = ['Comparison']
            utils.plot_subplots(plots_list, plot_titles, os.path.join(savepath, 'test_skips.png'))

            comp_grid = torchvision.utils.make_grid(torch.cat(odd_rows), pad_value=1, normalize=True)
            plots_list = [comp_grid.cpu().numpy().transpose((1,2,0))]
            plot_titles = ['Comparison']
            utils.plot_subplots(plots_list, plot_titles, os.path.join(savepath, 'test_comparison.png'))

            plots_list = [im_grid_hat.numpy().transpose((1,2,0))]
            plot_titles = ['Sampled (trajectory)']
            utils.plot_subplots(plots_list, plot_titles, os.path.join(savepath, 'test_sample_traj.png'))

            # save the images
            plots_list = [im_grid.numpy().transpose((1,2,0)),im_grid_hat_single.numpy().transpose((1,2,0))]
            plot_titles = ['Original','Sampled (single)']
            utils.plot_subplots(plots_list, plot_titles, os.path.join(savepath, 'test_sample.png'))

            # save the movies
            utils.save_gif(sampled_frames.detach().cpu(), os.path.join(savepath, 'movies/test_sample_traj.gif'))
            utils.save_gif(sampled_frames2.detach().cpu(), os.path.join(savepath, 'movies/test_sample_traj2.gif'))

        else:

            plots_list = [frames.detach().cpu().numpy(), frames_hat.detach().cpu().numpy()]
            plot_titles = ['Original','Sampled (single)']
            utils.plot_subplots(plots_list, plot_titles, os.path.join(savepath, 'test_sample.png'), plot_type='plot')

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


    try:
        n_runs = cfg['n_runs']
    except KeyError:
        n_runs = 5
    try:
        n_tries = cfg['n_tries']
    except KeyError:
        n_tries = 1

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

        if n_tries == 1:
            cfg['ae']['path']  = 'ae_best_val.pth'
            cfg['sde']['path'] = 'mu_best_val.pth'
        else:
            cfg['ae']['path']  = 'ae_best_val_bt.pth'
            cfg['sde']['path'] = 'mu_best_val_bt.pth'

        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(cfg)
        init  = setup(cfg, savepath)
        stats = test(**init)

        all_stats['runs'].append(stats)
        print(stats)

    with open(os.path.join(savepath,'test_stats.pkl'), 'wb') as f:
        pickle.dump(all_stats, f)
