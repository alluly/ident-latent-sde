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

def get_data(name : str, **params):

    '''
    Helper method to get the dataloaders given parameters for the dataset
    The parameters come from the yaml file for the experiment
    name : string that gives you the experiment type
    '''

    global savepath
    path = os.path.join(savepath, 'data.pkl')
    print("PATH")
    print(path)

    if name == 'bair':
        train_dataset = data_loaders.RobotPushDataset(train=True)
        test_dataset  = data_loaders.RobotPushDataset(train=False)

    elif name == 'balls':
        '''
        synthetic is the yellow dot experiment
        '''
        dataset = data_loaders.BallDataset(path=path, **params)

    elif name == 'Mball':
        '''
        synthetic is the yellow dot experiment
        '''
        dataset = data_loaders.MBallDataset(path=path, **params)


    elif name == 'smnist-d':
        '''
        smnist-d is the mnist digits dataset
        '''
        dataset = data_loaders.SMNISTDynamicDataset(path=path, **params)


    elif name == 'wass':
        '''
        wass is the wasserstein dataset between two images in the COIL20 dataset
        '''
        dataset = data_loaders.COILDataset(load_path=path, **params)

    elif name == 'ballwass':
        '''
        ballwass is the wasserstein dataset between two images in the COIL20 dataset 
        with two balls
        '''
        dataset = data_loaders.BallWassDataset(path=path, **params)

    elif name == 'stocks':
        '''
        stocks is the S&P 500 dataset 
        '''
        dataset = data_loaders.SP()

    elif name == 'vector':
        dataset = data_loaders.VectorDataset(**params)

    elif 'dna' in name:
        '''
        dna is the microscopy dataset 
        '''
        dataset = data_loaders.DNADataset(path=path, name=name, **params)
    else:
        raise NotImplementedError

    # plot the scaled data
    plt.plot(dataset.xt)
    plt.savefig(os.path.join(savepath, 'original_seq_scaled.pdf'))
    plt.close('all')

    # plot the original data
    if name != 'stocks':
        orig_plots = [dataset.xt_orig[:,i] for i in range(dataset.xt_orig.shape[1])]
        plot_titles = ['component {}'.format(i) for i in range(dataset.xt_orig.shape[1])]
        utils.plot_subplots(orig_plots, plot_titles, os.path.join(savepath, 'original_seq.pdf'), plot_type='plot', axis=True)

    # train val test split
    train_inds = range(len(dataset) - params['n_test'] * 2)
    val_inds   = range(len(dataset) - params['n_test'] * 2, len(dataset) - params['n_test']) 
    test_inds  = range(len(dataset) - params['n_test'], len(dataset)) 

    train_dataset = torch.utils.data.Subset(dataset, train_inds)
    val_dataset   = torch.utils.data.Subset(dataset, val_inds)
    test_dataset  = torch.utils.data.Subset(dataset, test_inds)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, \
            num_workers = 0,  batch_size = params['batch_size'], drop_last=True)

    val_dataloader = torch.utils.data.DataLoader(val_dataset, \
            num_workers = 0,  batch_size = params['batch_size'])

    test_dataloader  = torch.utils.data.DataLoader(test_dataset, \
            num_workers = 0, batch_size = params['batch_size'])

    return train_dataloader, val_dataloader, test_dataloader

def setup(cfg, sp):

    global savepath
    savepath = sp

    optim_params   = cfg['optimizer']
    dataset_params = cfg['dataset']
    ae_params      = cfg['ae']
    sde_params     = cfg['sde']

    dataset = dataset_params['name']
    dt  = ( dataset_params['tn'] - dataset_params['t0'] ) / dataset_params['n_points']

    if dataset == 'balls':

        fcn_sigma = '[1, 0, 0, 1]'
        x_init = np.array([0, 0]) 

    elif dataset == 'Mball':

        fcn_sigma = list(np.eye(10).reshape(1,-1)[0]) 
        x_init = np.random.randn(10)
        #x_init = np.array([0,0,1,1,-1,-1,1,-1,-1,1])

    elif dataset == 'smnist' or dataset == 'smnist-d':

        fcn_sigma = list(np.eye(4).reshape(1,-1)[0]) 
        x_init = np.array([1, 1, -1,- 1]) 

    elif dataset == 'wass':

        fcn_sigma = '[1]'
        x_init = np.array([1])

    elif dataset == 'ballwass':

        fcn_sigma = np.array([1, 0, 0, 0, 1, 0, 0, 0, 1])
        x_init = np.array([1,0,-1])

    elif 'dna' in dataset:

        fcn_sigma = '[1, 0, 0, 1]'
        x_init = np.array([0, 0]) 
        #fcn_sigma = np.array([1, 0, 0, 0, 1, 0, 0, 0, 1])
        #x_init = np.array([1,0,-1])

    elif 'vector' in dataset:
        x_init = np.random.randn(3)
        fcn_sigma = np.array([1, 0, 0, 0, 1, 0, 0, 0, 1])

    else:

        fcn_mu    = None
        fcn_sigma = None
        dt = 0.5

    try:
        x_init    = dataset_params['x_init']
        fcn_sigma = dataset_params['fcn_sigma']

    except KeyError:
        pass

    finally:

        print(fcn_sigma)
        fcn_params = {'x_init': x_init, 'fcn_sigma': fcn_sigma, 'dt': dt}

    train_dataloader, val_dataloader, test_dataloader = get_data(**{**dataset_params, **fcn_params})

    # Setup the networks
    lr = optim_params['lr']
    z_dim = ae_params['net']['latent_dim']

    # Mu/sigma hyperparameters
    net_type = sde_params['type']
    width    = sde_params['width']
    depth    = sde_params['depth']
    act      = sde_params['act']

    # load from an existing path if necessary
    ae_path       = ae_params['path']
    ae_net_params = ae_params['net']

    ae    = nets.StochConvAE(**ae_net_params).to(device)

    lr_ae    = ae_params['lr']
    lr_mu    = sde_params['lr_mu']
    lr_sigma = sde_params['lr_sigma']
    

    if ae_path:
        try:
            ae.load_state_dict(torch.load(os.path.join(savepath,'saved_nets/{}'.format(ae_path))))
        except FileNotFoundError:
            print('trained AE not found, using random initialization')
            print('this is usually a bad error...')

    total_params = sum(p.numel() for p in ae.parameters())
    print('Total number of parameters in autoencoder: ' + str(total_params))

    # load the latent space functions
    if net_type == 'mlp':

        tri_inds = torch.tril_indices(z_dim,z_dim)
        upper_tri = torch.eye(z_dim)[tri_inds[0,:], tri_inds[1,:]]

        mu    = nets.MLP(z_dim + 1,width,depth,z_dim,activation=act).to(device)
        sigma = nets.MLP(z_dim + 1,width,depth,int((z_dim+1)*z_dim/2),activation=act).to(device)

        total_params = sum(p.numel() for p in mu.parameters())
        print('Total number of parameters in mu: ' + str(total_params))
        total_params = sum(p.numel() for p in sigma.parameters())
        print('Total number of parameters in sigma: ' + str(total_params))

        nn.init.zeros_(mu.out.weight.data)
        nn.init.ones_(mu.out.bias.data)
        nn.init.zeros_(sigma.out.weight.data)

        with torch.no_grad():
            sigma.out.bias = nn.Parameter(upper_tri.to(device), requires_grad=True)

        opt_params = [{'params': mu.parameters(), 'lr': lr_mu},
                  {'params': sigma.parameters(), 'lr': lr_sigma}, 
                  {'params': ae.parameters(), 'lr': lr_ae}]

    elif net_type == 'const':

        mu = nn.Parameter(torch.randn(z_dim).to(device),requires_grad=True)

        if ae_net_params['sigma_type'] == 'diag':
            sigma = nn.Parameter(torch.ones(z_dim).to(device),requires_grad=True)
        else:

            tri_inds = torch.tril_indices(z_dim,z_dim)
            upper_tri = torch.eye(z_dim)[tri_inds[0,:], tri_inds[1,:]]
            sigma = nn.Parameter(upper_tri.to(device),requires_grad=True)

        opt_params = [{'params': [mu], 'lr': lr_mu},
                  {'params': [sigma], 'lr': lr_sigma}, 
                  {'params': ae.parameters(), 'lr': lr_ae}]

    elif net_type == 'const-sig':

        tri_inds = torch.tril_indices(z_dim,z_dim)
        upper_tri = torch.eye(z_dim)[tri_inds[0,:], tri_inds[1,:]]

        mu = nets.MLP(z_dim + 1, width, depth, z_dim, activation=act).to(device)

        if sde_params['path']:
            mu.load_state_dict(torch.load(os.path.join(savepath, 'saved_nets/{}'.format(sde_params['path']))))

        if ae_net_params['sigma_type'] == 'diag':
            sigma = nn.Parameter(torch.ones(z_dim).to(device),requires_grad=True)
        else:
            sigma = nn.Parameter(upper_tri.to(device),requires_grad=True)

        opt_params = [{'params': mu.parameters(),'lr': lr_mu},
                  {'params': [sigma], 'lr': lr_sigma}, 
                  {'params': ae.parameters(), 'lr': lr_ae}]
    elif net_type == 'const-sig-nt':

        tri_inds = torch.tril_indices(z_dim,z_dim)
        upper_tri = torch.eye(z_dim)[tri_inds[0,:], tri_inds[1,:]]

        mu = nets.MLP(z_dim, width, depth, z_dim, activation=act).to(device)

        #nn.init.ones_(mu.out.bias.data)
        #nn.init.zeros_(mu.out.weight.data)
        #nn.init.ones_(mu.out.bias.data)

        if ae_net_params['sigma_type'] == 'diag':
            sigma = nn.Parameter(torch.ones(z_dim).to(device),requires_grad=True)
        else:
            sigma = nn.Parameter(upper_tri.to(device),requires_grad=True)

        if ae_net_params['loss'] == 'mc':

            opt_params = [{'params': mu.parameters(),'lr': lr_mu, 'eps':1e-7},#, 'betas':(0.5, 0.99), 'eps': 1e-5}, #-7
                      {'params': [sigma], 'lr': lr_sigma}, 
                      {'params': ae.parameters(), 'lr': lr_ae, 'betas':(0.5, 0.99), 'eps': 1e-5}] # -5

        elif ae_net_params['loss'] == 'exact':

            #nn.init.ones_(mu.out.bias.data)
            nn.init.zeros_(mu.out.weight.data)
            nn.init.ones_(mu.out.bias.data)
            opt_params = [{'params': mu.parameters(),'lr': lr_mu, 'eps':1e-3, 'weight_decay': 1e-5},#, 'betas':(0.5, 0.99), 'eps': 1e-5}, #-7
                      {'params': [sigma], 'lr': lr_sigma}, 
                      {'params': ae.parameters(), 'lr': lr_ae, 'betas':(0.5, 0.99), 'eps': 1e-5, 'weight_decay': 1e-5}] # -5
        else:
            
            opt_params = [{'params': mu.parameters(),'lr': lr_mu, 'eps':1e-3},#, 'betas':(0.5, 0.99), 'eps': 1e-5}, #-7
                      {'params': [sigma], 'lr': lr_sigma}, 
                      {'params': ae.parameters(), 'lr': lr_ae, 'betas':(0.5, 0.99), 'eps': 1e-5}] # -5

        if sde_params['path']:
            try:
                mu.load_state_dict(torch.load(os.path.join(savepath, 'saved_nets/{}'.format(sde_params['path']))))
            except FileNotFoundError:
                print('mu not found, using random initialization')

        try:
            suffix = 'latest'
            if sde_params['path']:
                with open(os.path.join(savepath,'sigma_{}.pkl'.format(suffix)),'rb') as f:
                    sigma = pickle.load(f)
        except:
            if ae_net_params['sigma_type'] == 'diag':
                sigma = nn.Parameter(torch.ones(z_dim).to(device),requires_grad=True)
            else:
                sigma = nn.Parameter(upper_tri.to(device),requires_grad=True)

    elif net_type == 'linear':
        tri_inds = torch.tril_indices(z_dim,z_dim)
        upper_tri = torch.eye(z_dim)[tri_inds[0,:], tri_inds[1,:]]

        mu    = nets.Lin(z_dim).to(device)

        if sde_params['path']:
            mu.load_state_dict(torch.load(os.path.join(savepath, 'saved_nets/{}'.format(sde_params['path']))))
        
        if ae_net_params['sigma_type'] == 'diag':

            sigma = nn.Parameter(torch.ones(z_dim).to(device),requires_grad=True)
        else:
            sigma = nn.Parameter(upper_tri.to(device),requires_grad=True)

        opt_params = [{'params': mu.parameters(),'lr': lr_mu},
                  {'params': [sigma], 'lr': lr_sigma}, 
                  {'params': ae.parameters(), 'lr': lr_ae}]
    
    elif net_type == 'well':
    
        tri_inds = torch.tril_indices(z_dim,z_dim)
        upper_tri = torch.eye(z_dim)[tri_inds[0,:], tri_inds[1,:]]

        mu    = nets.Well(z_dim).to(device)
        if sde_params['path']:
            mu.load_state_dict(torch.load(os.path.join(savepath, 'saved_nets/{}'.format(sde_params['path']))))

        if ae_net_params['sigma_type'] == 'diag':

            sigma = nn.Parameter(torch.ones(z_dim).to(device),requires_grad=True)
        else:
            sigma = nn.Parameter(upper_tri.to(device),requires_grad=True)

        opt_params = [{'params': mu.parameters(),'lr': lr_mu},
                  {'params': [sigma], 'lr': lr_sigma}, 
                  {'params': ae.parameters(), 'lr': lr_ae}] #'betas':(0.5, 0.99), 'eps': 1e-3}]
        
    optimizer = getattr(optim, optim_params['name'])(opt_params)
    if optim_params['sched']:
        scheduler = getattr(optim.lr_scheduler, optim_params['sched'])(optimizer, **optim_params['sched_param'])
    else:
        scheduler = None

    initialized = {'ae' : ae, 
            'mu'    : mu, 
            'sigma' : sigma, 
            'dt'    : dt,
            'train_data' : train_dataloader, 
            'val_data'   : val_dataloader,
            'test_data'  : test_dataloader,
            'optimizer'  : optimizer, 
            'scheduler'  : scheduler, 
            'n_epochs'   : optim_params['n_epochs'],
            'data_params': dataset_params
            }

    return initialized 
