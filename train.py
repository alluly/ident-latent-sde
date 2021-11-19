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

        return z_samples, frames


def plot_images(ae, mu, sigma, frames, frames_hat, dt, ts, l2_small):
    # plot latent trajectories
    # plot next frame reconstructions
    z_samples, sampled_frames = sample_n_frames(frames[:2], 
            ts[:2], 
            dt, 
            ae.eval(), 
            mu, 
            sigma) 
    _, sampled_frames2 = sample_n_frames(frames[:2], 
            ts[:2], 
            dt, 
            ae.eval(), 
            mu, 
            sigma*2) 

    # create the image grids
    im_grid_hat_single = torchvision.utils.make_grid(frames_hat[:64].detach().cpu(), pad_value=1, normalize=True)
    im_grid_hat = torchvision.utils.make_grid(sampled_frames[:64].detach().cpu(), pad_value=1, normalize=True)
    im_grid     = torchvision.utils.make_grid(frames[:64].detach().cpu(), pad_value=1, normalize=True)

    odd_rows = []

    for row in range(4):
        odd_rows.append(frames[row*8:(row+1)*8])
        odd_rows.append(sampled_frames[row*8:(row+1)*8])

    comp_grid = torchvision.utils.make_grid(torch.cat(odd_rows), pad_value=1, normalize=True)
    plots_list = [comp_grid.cpu().numpy().transpose((1,2,0))]
    plot_titles = ['Comparison']
    utils.plot_subplots(plots_list, plot_titles, os.path.join(savepath, 'train_comparison.png'))

    plots_list = [im_grid_hat.numpy().transpose((1,2,0))]
    plot_titles = ['Sampled (trajectory)']
    utils.plot_subplots(plots_list, plot_titles, os.path.join(savepath, 'train_sample_traj.png'))

    # save the images
    plots_list = [im_grid.numpy().transpose((1,2,0)),im_grid_hat_single.numpy().transpose((1,2,0))]
    plot_titles = ['Original','Sampled (single)']

    if l2_small: 
        utils.plot_subplots(plots_list, plot_titles, os.path.join(savepath, 'train_sample_best.png'))
    else:
        utils.plot_subplots(plots_list, plot_titles, os.path.join(savepath, 'train_sample.png'))

    # save the movies
    utils.save_gif(sampled_frames.detach().cpu(), os.path.join(savepath, 'movies/train_sample_traj.gif'))
    utils.save_gif(sampled_frames2.detach().cpu(), os.path.join(savepath, 'movies/train_sample_traj2.gif'))


def save_nets(ae, mu, sigma, suffix):
    '''
    Routine to save the current state of our network
    ae : autoencoder network 
    mu : latent drift network
    sigma : latent diffusion network
    suffix : str that defines how we want to save the network
    '''

    # save all the networks
    torch.save(ae.state_dict(), os.path.join(savepath,'saved_nets/ae_{}.pth'.format(suffix)))

    if type(mu) == nets.MLP or type(mu) == nets.Lin or type(mu) == nets.Well:
        torch.save(mu.state_dict(), os.path.join(savepath,'saved_nets/mu_{}.pth'.format(suffix)))
    else:
        with open(os.path.join(savepath,'mu_{}.pkl'.format(suffix)),'wb') as f:
            pickle.dump(mu, f)

    if type(sigma) == nets.MLP:
        torch.save(sigma.state_dict(), os.path.join(savepath,'saved_nets/sigma_{}.pth'.format(suffix)))
    else:
        with open(os.path.join(savepath,'sigma_{}.pkl'.format(suffix)),'wb') as f:
            pickle.dump(sigma, f)

def train(ae, 
        mu, 
        sigma, 
        dt, 
        train_data, 
        val_data, 
        optimizer, 
        scheduler, 
        n_epochs, 
        data_params,
        **kwargs):

    '''
    The main training routine:

    ae : neural network (torch.Module subclass) that represents our autoencoder
    mu : network or parameter that describes the latent drift
    sigma : network or parameter that describes the latent diffusion
    dt : time step
    train_data : dataloader with the training data
    val_data : dataloader with validation data
    optimizer : optimization algorithm torch.optim 
    scheduler : lr decay schedule
    n_epochs  : number of epochs to run
    data_params : parameters associated with the dataset

    returns statistics with respect to training
    '''

    global savepath
    global loss_type

    train_dataset = train_data.dataset.dataset
    val_dataset   = val_data.dataset.dataset
    try:
        inner_num = data_params['inner_iter']
    except:
        inner_num = 1


    if n_epochs > 1000:
        reserve_epoch = 499
    else:
        reserve_epoch = 49

    # plotting parameters
    l2_small   = True
    l2_small_valid = True

    losses_train = []
    losses_valid = []

    try:
        plot_freq = data_params['plot_freq']
    except KeyError:
        plot_freq = 50

    try:
        plot_train = data_params['plot_train']
    except KeyError:
        plot_train = True

    # setup the stats dict
    stats = {'kl': np.Inf, 
            'l2' : np.Inf, 
            'l2_valid': np.Inf, 
            'kl_valid': np.Inf, 
            'mu_mse': 0, 
            'mu_mse_valid': 0, 
            'mu_rel': 0,
            'mu_rel_valid': 0,
            'sde_mse': 0, 
            'sde_mse_valid': 0,
            'sde_rel': 0,
            'sde_rel_valid': 0,
            'val_cond_met': False}

    for epoch in range(n_epochs):

        ae.train()
        mu.train()
        #sigma.train()

        for idx, (frames, ts) in enumerate(train_data):

            # save a gif of the data
            if len(frames.shape) > 2:
                if idx == 0 and epoch == 0:
                    utils.save_gif(frames.detach().cpu(), os.path.join(savepath, 'orig_data.gif'))

            # transfer the data to the device
            # the rest is boilerplate
            frames = frames.float().to(device)
            ts     = ts.float().to(device)

            for _ in range(inner_num):

                optimizer.zero_grad()

                kl_loss, l2_loss,\
                        frames_hat, mu_hat, q_mu, sigma_hat_full, q_sigma_full, inc, z = ae.step(frames, ts, dt, mu, sigma)


                kl_loss1, l2_loss1,\
                        _, _, _, _, _, _, _ = ae.step(frames, ts, dt, mu, sigma, plus_one=True)

                sigma.data = sigma / sigma.norm(2) * torch.ones(z.shape[1]).norm(2)

                loss = kl_loss + kl_loss1 + l2_loss + l2_loss1 + 20*sigma.norm(1)

                losses_train.append((kl_loss.item(), l2_loss.item()))

                loss.backward()
                optimizer.step()

                # And that's the end of the train routine

            '''
            PLOT SECTION
            This is still quite messy and needs to be refactored,
            but this is all visualization calls
            '''

            if kl_loss < stats['kl']:

                stats['kl'] = kl_loss.item()
                stats['mu'] = mu_hat.mean().item()

            if plot_train and (epoch % plot_freq) == 0 and idx == 0: 

                if l2_loss < stats['l2']:
                    l2_small = True
                    stats['l2'] = l2_loss.item()
                else:
                    l2_small = False

                if len(frames.shape) > 2:
                    plot_images(ae, mu, sigma, frames, frames_hat, dt, ts, l2_small)

                # plot mu hat
                mu_hat_samples, hat_domain = utils.plot_mu_hat(mu, sigma, q_mu, ts, os.path.join(savepath, 'mu_hat_plot.png'))

                if len(frames.shape) < 3:
                    plots = [frames.cpu(), frames_hat.detach().cpu()]
                    names = ['Original', 'Sampled']
                    utils.plot_subplots(plots, names, os.path.join(savepath, 'train_recon.png'), plot_type='plot', axis=True)
                    _, sampled_frames = sample_n_frames(frames[:2], ts[:2], dt, ae, mu, sigma, n_generate=1000) 
                    plots = [frames.cpu(), sampled_frames.detach().cpu()]
                    names = ['Original', 'Sampled']
                    utils.plot_subplots(plots, names, os.path.join(savepath, 'train_sampled.png'), plot_type='plot', axis=True)

                    if frames.shape[1] == 1:

                        with torch.no_grad():

                            inx = torch.linspace(frames.min().item(), frames.max().item()).unsqueeze(1)
                            oned_enc = ae.encode(inx.cuda(0))[0].detach().data.clone().cpu()

                            enc_scale = ( inx.log() / oned_enc ).mean()
                            enc_shift = (inx.log() - enc_scale * oned_enc).mean()

                            plt.plot(inx.detach().cpu(), enc_scale * oned_enc.cpu(), label='encoder')
                            plt.plot(inx.detach().cpu(), inx.log().detach().cpu(),label='log')
                            plt.legend()

                            plt.savefig(os.path.join(savepath, 'encoder_plot.pdf'))

                            plt.close('all')

                '''
                AFFINE TRANSFORM SECTION
                '''

                # calculate the affine map between xt and z
                current_run = train_dataset.xt_orig[idx*z.shape[0]:(idx+1)*z.shape[0]]
                scale = (train_dataset.xt_orig.max() - train_dataset.xt_orig.min())

                q_mu = q_mu[:, :train_dataset.xt_orig.shape[1]]
                z    = z[:, :train_dataset.xt_orig.shape[1]]

                if not 'stocks' in savepath:
                    # if this is the stocks dataset, don't compute the scaling since there is none

                    if data_params['affine']:

                        transformed_xt, Q, b, sde_mse, sde_rel = utils.calc_affine(
                                current_run,
                                z.detach().cpu().numpy(), 
                                savepath, 
                                affine=data_params['affine'])

                        if z.shape[1] == mu_hat.shape[1]:
                    
                            mu_residuals, mu_rel, mu_crlb = utils.compare_mu2(
                                    mu, 
                                    q_mu, 
                                    ts, 
                                    Q, 
                                    b, 
                                    dt, 
                                    train_dataset, 
                                    os.path.join(savepath,'mu_comp_scaled.png'), 
                                    affine=data_params['affine'],
                                    loss_type=loss_type)
                        else: 
                            mu_residuals = torch.Tensor([np.NaN]).numpy()
                            mu_crlb = torch.Tensor([np.NaN]).numpy()
                            mu_rel  = torch.Tensor([np.NaN]).numpy()


                    else:

                        q_max = q_mu.max()
                        q_min = q_mu.min()

                        if loss_type == 'exact': 
                            q_scaled = ((q_mu - q_min ) / (q_max - q_min) * (scale) ).detach().cpu().numpy()
                            #q_scaled = q_mu.detach().cpu().numpy()  / np.sqrt(scale)
                        else:
                            q_scaled = q_mu.detach().cpu().numpy() / scale

                        transformed_xt, Q, b, sde_mse, sde_rel = utils.calc_affine(
                                current_run,
                                q_scaled,
                                #z.detach().cpu().numpy() / scale, 
                                savepath, 
                                affine=data_params['affine'])
                        if z.shape[1] == mu_hat.shape[1]:
                            mu_residuals, mu_rel, mu_crlb = utils.compare_mu2(
                                    mu, 
                                    q_mu, 
                                    ts, 
                                    Q, 
                                    b, 
                                    dt, 
                                    train_dataset, 
                                    os.path.join(savepath,'mu_comp_scaled.png'), 
                                    affine=data_params['affine'],
                                    loss_type=loss_type)
                        else: 
                            mu_residuals = torch.Tensor([np.NaN]).numpy()
                            mu_crlb = torch.Tensor([np.NaN]).numpy()
                            mu_rel  = torch.Tensor([np.NaN]).numpy()
                    stats['sde_mse'] = sde_mse.copy() 
                    stats['sde_rel'] = sde_rel.copy()

                    # compare the estimated mu to the true mu with the affine map q
                    stats['mu_mse']  = mu_residuals.copy()
                    stats['mu_rel']  = mu_rel.copy()
                    stats['mu_crlb'] = mu_crlb.copy()
                else:
                    mu_residuals = torch.Tensor([np.NaN]).numpy()
                    mu_crlb = torch.Tensor([np.NaN]).numpy()
                    mu_rel  = torch.Tensor([np.NaN]).numpy()

                    stats['sde_mse'] = torch.Tensor([np.NaN]).numpy()
                    stats['sde_rel'] = torch.Tensor([np.NaN]).numpy()

                    # compare the estimated mu to the true mu with the affine map Q
                    stats['mu_mse']  = torch.Tensor([np.NaN]).numpy()
                    stats['mu_rel']  = torch.Tensor([np.NaN]).numpy()
                    stats['mu_crlb'] = torch.Tensor([np.NaN]).numpy()


            # plot and print 
            print('Epoch {} iter {}'.format(epoch, idx))
            print('L2 loss {}'.format(l2_loss.item()))
            print('KL loss {}'.format(kl_loss.item()))

            plots_list = [(q_mu[1:]-q_mu[:-1]).detach().cpu().numpy(), mu_hat.detach().cpu().numpy()]
            plot_titles = ['q_mu', 'mu_hat'] 
            utils.plot_subplots(plots_list, plot_titles, os.path.join(savepath, 'mu_comp.png'), plot_type='plot', axis=True)

        if scheduler:
            if type(scheduler) == torch.optim.lr_scheduler.ReduceLROnPlateau:
                scheduler.step(l2_loss)
            else:
                scheduler.step()

        if (epoch % plot_freq) == 0:
            # save all the networks
            #if len(frames.shape) < 3:
            #    utils.plot_mu_hat(mu, None, z, ts, os.path.join(savepath, 'mu_hat_est.pdf'))
            save_nets(ae, mu, sigma, 'latest')
            with open(os.path.join(savepath, 'latent.pkl'), 'wb') as f:
                #lat_d = {'q_mu' : q_mu.detach().cpu().numpy(), 'ts' : ts, 'xt_orig' : dataset.xt_orig}
                lat_d = {'q_mu' : transformed_xt, 'ts' : ts, 'xt_orig' : train_dataset.xt_orig}
                pickle.dump(lat_d, f)

            if type(sigma) == nn.Parameter:
                print('Update sigma_hat')
                print(sigma)
                stats['sigma_hat'] = (sigma.sort(descending=True)[0]).detach().cpu().numpy()

        if (epoch % plot_freq) == 0 and plot_train:

            '''
            EVAL
            '''

            # with our validataion data, see how well we're predicting 
            with torch.no_grad():
                ae.eval()

                # first, compute how well we predict the next step on the validation data
                for idxt, (frames_test, ts_test) in enumerate(val_data):

                    frames_test = frames_test.float().to(device)
                    ts_test     = ts_test.float().to(device)
                    
                    kl_loss_test, l2_loss_test,\
                            frames_hat_test, mu_hat_test, q_mu_test, sigma_hat_full, q_sigma_full,  \
                            inc_test, z_test = ae.step(frames_test, 
                                    ts_test, 
                                    dt, 
                                    mu, 
                                    sigma)

                    losses_valid.append((kl_loss_test.item(), l2_loss_test.item()))

                q_mu_test = q_mu_test[:,:train_dataset.xt_orig.shape[1]]
                z_test    = z_test[:,:train_dataset.xt_orig.shape[1]]

                if len(frames_hat_test.shape) < 3 and l2_loss_test < stats['l2_valid']:
                    stats['l2_valid'] = l2_loss_test.item()
                    stats['kl_valid'] = kl_loss_test.item()
                    l2_small_valid = True
                    stats['val_cond_met'] = True
                    save_nets(ae, mu, sigma, 'best_val')
                    plots = [frames_test.cpu(), frames_hat_test.detach().cpu()]
                    names = ['Original', 'Sampled']
                    utils.plot_subplots(plots, names, os.path.join(savepath, 'valid_recon.png'), plot_type='plot')

                # if the l2 and kl are sufficiently small, save these as our current best networks
                if ((l2_loss_test < stats['l2_valid'] and epoch > reserve_epoch) or ('dna' in savepath)) and ('z={}'.format(train_dataset.xt_orig.shape[1]) in savepath):

                    stats['val_cond_met'] = True

                    #stats['l2_valid'] = kl_loss_test.item()*l2_loss_test.item()
                    stats['l2_valid'] = l2_loss_test.item()
                    stats['kl_valid'] = kl_loss_test.item()
                    l2_small_valid = True

                    save_nets(ae, mu, sigma, 'best_val')

                    # Compute the mapping over the training data since we want to see the fit within the whole time series
                    frames = torch.Tensor(train_dataset.frames).float().to(device)[:z.shape[0]]
                    ts     = torch.Tensor(train_dataset.ts).float().to(device)[:z.shape[0]]
                    
                    kl_loss, l2_loss,\
                            frames_hat, mu_hat, q_mu, sigma_hat_full, q_sigma_full, inc, z = ae.step(frames, 
                                    ts, 
                                    dt, 
                                    mu, 
                                    sigma)

                    if 'gbm' in savepath:
                        gbm = True
                    else: 
                        gbm = False

                    # compare the estimated mu to the true mu with the affine map Q
                    current_run = train_dataset.xt_orig[:z.shape[0]]
                    scale = train_dataset.xt_orig.max() - train_dataset.xt_orig.min()

                    if gbm:
                        scale = (np.log(train_dataset.xt_orig[:]).max() - np.log(train_dataset.xt_orig[:]).min()) 

                        
                    if len(frames.shape) < 3:
                        plots = [frames.cpu(), frames_hat.detach().cpu()]
                        names = ['Original', 'Sampled']
                        utils.plot_subplots(plots, names, os.path.join(savepath, 'valid_recon.png'), plot_type='plot')
                        continue

                    if data_params['affine']:
                        transformed_xt, Q, b, sde_mse, sde_rel = utils.calc_affine(current_run,
                                z.detach().cpu().numpy(), 
                                savepath, 
                                affine=data_params['affine'])

                        mu_mse, mu_rel, mu_crlb = utils.compare_mu2(mu, 
                                q_mu, 
                                ts, 
                                Q, 
                                b, 
                                dt, 
                                train_dataset, 
                                os.path.join(savepath,'mu_comp_best_val.png'),
                                affine=data_params['affine'],
                                loss_type=loss_type)
                    else:
                        q_max = q_mu.max()
                        q_min = q_mu.min()

                        if loss_type == 'exact': 
                            q_scaled = ((q_mu - q_min ) / (q_max - q_min) * (scale) ).detach().cpu().numpy()
                        else:
                            q_scaled = q_mu.detach().cpu().numpy() / scale

                        transformed_xt, Q, b, sde_mse, sde_rel = utils.calc_affine(
                                current_run,
                                q_scaled,
                                #z.detach().cpu().numpy() / scale, 
                                savepath, 
                                affine=data_params['affine'],
                                gbm=gbm)

                        mu_mse, mu_rel, mu_crlb = utils.compare_mu2(mu, 
                                q_mu, 
                                ts, 
                                Q, 
                                b, 
                                dt, 
                                train_dataset, 
                                os.path.join(savepath,'mu_comp_best_val.png'),
                                affine=data_params['affine'],
                                loss_type=loss_type)

                    stats['mu_mse_val']    = mu_mse.copy()
                    stats['mu_rel_val']    = mu_rel.copy()
                    stats['mu_crlb_val']   = mu_crlb.copy()
                    stats['sde_mse_valid'] = sde_mse.copy()
                    stats['sde_rel_valid'] = sde_rel.copy()

                else:

                    l2_small_valid = False

                    plt.plot(torch.arange(sigma.shape[0]).detach().cpu().numpy(), (sigma.sort(descending=True)[0]).detach().cpu().numpy())
                    plt.savefig(os.path.join(savepath, 'sigma_hat.pdf'))
                    plt.close('all')

                    if 'dna' in savepath or 'balls' in savepath:
                        stats['val_cond_met'] = True
                        save_nets(ae, mu, sigma, 'best_val')

                im_grid_test            = torchvision.utils.make_grid(frames_test[:64].detach().cpu(), pad_value=1, normalize=True)
                im_grid_hat_single_test = torchvision.utils.make_grid(frames_hat_test[:64].detach().cpu(), pad_value=1, normalize=True)

                # sample the frames for the next n images
                _, sampled_frames_test  = sample_n_frames(frames_test[:2], ts_test[:2], dt, ae.eval(), mu, sigma) 
                _, sampled_frames_test2 = sample_n_frames(frames_test[:2], ts_test[:2], dt, ae.eval(), mu, sigma*2) 
                im_grid_hat_test        = torchvision.utils.make_grid(sampled_frames_test[:64].detach().cpu(), pad_value=1, normalize=True)

                odd_rows = []

                for row in range(4):
                    odd_rows.append(frames_test[row*8:(row+1)*8])
                    odd_rows.append(sampled_frames_test[row*8:(row+1)*8])

                comp_grid = torchvision.utils.make_grid(torch.cat(odd_rows), pad_value=1, normalize=True)
                plots_list = [comp_grid.cpu().numpy().transpose((1,2,0))]
                plot_titles = ['Comparison']
                utils.plot_subplots(plots_list, 
                        plot_titles, 
                        os.path.join(savepath, 'valid_comparison.png'))

                if val_dataset.xt.shape[1] < 10:

                    utils.calc_affine(
                            val_dataset.xt[:z_test.shape[0]], 
                            np.sqrt(dt)*z_test.detach().cpu().numpy(), 
                            savepath, suffix='test')


            plots_list = [im_grid_test.numpy().transpose((1,2,0)), im_grid_hat_single_test.numpy().transpose((1,2,0))]
            plot_titles = ['Original','Sampled (single)']
            if l2_small_valid:
                utils.plot_subplots(plots_list, plot_titles, os.path.join(savepath, 'valid_sample_best.png'))
            else:
                utils.plot_subplots(plots_list, plot_titles, os.path.join(savepath, 'valid_sample.png'))

            plots_list = [im_grid_hat_test.numpy().transpose((1,2,0))]
            plot_titles = ['Sampled (trajectory)']
            if l2_small_valid:
                utils.plot_subplots(plots_list, plot_titles, os.path.join(savepath, 'valid_sample_traj_best.png'))
            else:
                utils.plot_subplots(plots_list, plot_titles, os.path.join(savepath, 'valid_sample_traj.png'))

            if len(sampled_frames_test.shape) > 2:
                utils.save_gif(sampled_frames_test.detach().cpu(), os.path.join(savepath, 'movies/valid_sample_traj.gif'))
                utils.save_gif(sampled_frames_test2.detach().cpu(), os.path.join(savepath, 'movies/valid_sample_traj_2.gif'))

            plt.figure(figsize=(10,5))
            plt.subplot(1,2,1)
            plt.title('NLL')
            plt.plot([kp[0] for kp in losses_train])
            plt.subplot(1,2,2)
            plt.title('l2')
            plt.yscale('log')
            plt.plot([kp[1] for kp in losses_train])
            plt.savefig(os.path.join(savepath, 'losses_train.png'))
            plt.close('all')

            plt.figure(figsize=(10,5))
            plt.subplot(1,2,1)
            plt.title('NLL')
            plt.plot([kp[0] for kp in losses_valid])
            plt.subplot(1,2,2)
            plt.title('l2')
            plt.yscale('log')
            plt.plot([kp[1] for kp in losses_valid])
            plt.savefig(os.path.join(savepath, 'losses_valid.png'))
            plt.close('all')

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
    import shutil

    args = get_parser().parse_args()

    yaml_filepath = args.filename

    with open(yaml_filepath, 'r') as f:
        cfg = yaml.load(f, yaml.SafeLoader)

    global savepath

    all_stats = {'config':cfg, 'runs':[]}

    try:
        n_runs = cfg['n_runs']
    except KeyError:
        n_runs = 5

    try:
        n_tries = cfg['n_tries']
    except KeyError:
        n_tries = 1
    print(n_tries)

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

        global loss_type
        loss_type = cfg['ae']['net']['loss']

        #if os.path.isfile(os.path.join(savepath, 'data.pkl')):
        #    os.remove(os.path.join(savepath, 'data.pkl'))

        if not os.path.exists(savepath):
            os.makedirs(savepath)

        if not os.path.exists(os.path.join(savepath,'movies')):
            os.makedirs(os.path.join(savepath,'movies'))

        if not os.path.exists(os.path.join(savepath,'saved_nets')):
            os.makedirs(os.path.join(savepath,'saved_nets'))

        log_format = "%(asctime)s %(message)s"
        logging.basicConfig(
            stream=sys.stdout,
            level=logging.INFO,
            format=log_format,
            datefmt="%m/%d %I:%M:%S %p",
        )
        fh = logging.FileHandler(os.path.join(savepath, "log.txt"))
        fh.setFormatter(logging.Formatter(log_format))
        logging.getLogger().addHandler(fh)
        logging.info("config = %s", cfg)

        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(cfg)

        best_loss = np.Inf

        for t_num in range(n_tries):

            while True: 
                initialized = setup.setup(cfg, savepath)
                stats = train(**initialized)
                val_cond_met = stats['val_cond_met']
                if val_cond_met or 'dna' in cfg['head'] or 'stocks' in cfg['head']:
                    break
            src_ae = os.path.join(savepath,'saved_nets/ae_best_val.pth')
            dst_ae = os.path.join(savepath,'saved_nets/ae_best_val_{}.pth'.format(t_num))
            src_mu = os.path.join(savepath,'saved_nets/mu_best_val.pth')
            dst_mu = os.path.join(savepath,'saved_nets/mu_best_val_{}.pth'.format(t_num))
            shutil.copyfile(src_ae, dst_ae)
            shutil.copyfile(src_mu, dst_mu)
            print('=========== End of Training ===========')
            print('Printing results for try {}'.format(t_num))
            print('STAT: L2 on Train: {}'.format(stats['l2']))
            print('STAT: KL on Train: {}'.format(stats['kl']))
            print('STAT: L2 on Validation: {}'.format(stats['l2_valid']))
            print('STAT: KL on Validation: {}'.format(stats['kl_valid']))
            print('STAT: mu mse on Validation: {}'.format(stats['mu_mse']))
            print('STAT: SDE mse on Validation: {}'.format(stats['sde_mse']))
            print('========== End of Results ============')

            if stats['kl_valid'] + stats['l2_valid'] < best_loss:

                best_loss = stats['kl_valid'] + stats['l2_valid']
                src_ae = os.path.join(savepath,'saved_nets/ae_best_val.pth')
                dst_ae = os.path.join(savepath,'saved_nets/ae_best_val_bt.pth')
                src_mu = os.path.join(savepath,'saved_nets/mu_best_val.pth')
                dst_mu = os.path.join(savepath,'saved_nets/mu_best_val_bt.pth')
                shutil.copyfile(src_ae, dst_ae)
                shutil.copyfile(src_mu, dst_mu)

        all_stats['runs'].append(stats)
        print(stats)

    with open(os.path.join(savepath,'saved_stats.pkl'), 'wb') as f:
        pickle.dump(all_stats, f)

