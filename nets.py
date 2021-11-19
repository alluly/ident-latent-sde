####
## This file contains relevant network structures for learning the encoder/decoder and drift.
## The main functions are at the bottom for the encoder and decoder. 
###

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions.multivariate_normal import MultivariateNormal

import numpy as np

import math

import losses
import utils

torch.manual_seed(787)
torch.cuda.manual_seed(787)

def init_weights(net, init_dict, gain=0.5, input_class=None):
    def init_func(m):
        if input_class is None or type(m) == input_class:
            for key, value in init_dict.items():
                param = getattr(m, key, None)
                if param is not None:
                    if value == 'normal':
                        nn.init.normal_(param.data, 0.0, gain)
                    elif value == 'xavier':
                        nn.init.xavier_normal_(param.data, gain=gain)
                    elif value == 'kaiming':
                        nn.init.kaiming_normal_(param.data, a=0, mode='fan_in')
                    elif value == 'orthogonal':
                        nn.init.orthogonal_(param.data, gain=gain)
                    elif value == 'uniform':
                        nn.init.uniform_(param.data)
                    elif value == 'zeros':
                        nn.init.zeros_(param.data)
                    elif value == 'very_small':
                        nn.init.constant_(param.data, 1e-3*gain)
                    elif value == 'xavier1D':
                        nn.init.normal_(param.data, 0.0, gain/param.numel().sqrt())
                    elif value == 'identity':
                        nn.init.eye_(param.data)
                    else:
                        raise NotImplementedError('initialization method [%s] is not implemented' % value)

    net.apply(init_func)

##
# activation functions
##

class relu2(nn.Module):
    def __init__(self):
        super(relu2,self).__init__()

    def forward(self,x):
        return F.relu(x)**2

class swish(nn.Module):
    def __init__(self):
        super(swish,self).__init__()

    def forward(self,x):
        return x*torch.sigmoid(x)

## 
# Parameteric Drifts
##

class Lin(nn.Module):
    def __init__(self, z):
        super(Lin, self).__init__()
        self.l = nn.Parameter(torch.randn(z))

    def forward(self, *inputs):
        return self.l * torch.cat(inputs,dim=1)

class Well(nn.Module):
    def __init__(self, z):
        super(Well, self).__init__()
        self.a = nn.Parameter(torch.randn(z))
        self.b = nn.Parameter(torch.randn(z))

    def forward(self, *inputs):
        return self.a * torch.cat(inputs,dim=1) - self.a * torch.cat(inputs,dim=1) ** 3

##
# MLP Drift
##

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, out_size,
                 weight_init='xavier', bias_init='zeros', gain=0.5, **params):
        super(MLP, self).__init__()

        self.first_layer = ScaledLinear(input_size, hidden_size, bias=False, **params)
        self.hidden_layers = nn.ModuleList()
        for i in range(num_layers):
            self.hidden_layers.append(ScaledLinear(hidden_size, hidden_size, bias=False, **params))
        self.out= nn.Linear(hidden_size, out_size, bias=True)
        init_weights(self, {'weight':weight_init, 'bias':bias_init}, gain=gain)

    def forward(self,*inputs):
        inputs = torch.cat(inputs,dim=1)
        out = self.first_layer(inputs)
        for layer in self.hidden_layers:
            out = layer(out)
        out = self.out(out)
        return out

class ScaledLinear(nn.Module):
    def __init__(self, input_size, output_size, activation='nn.ReLU', activation_parameters = {}, bias=True):
        super(ScaledLinear, self).__init__()
        self.linear = nn.Linear(input_size, output_size, bias=bias)
        if activation_parameters.get('output_size', False) is None:
            activation_parameters['output_size'] = output_size
        self.activation = eval(activation)(**activation_parameters)

    def forward(self,x):
        out = self.activation(self.linear(x))
        return out

class L2Proj(nn.Module):
    def __init__ (self):
        super(L2Proj, self).__init__()
    def forward(self, x):
        if torch.norm(x) > 1:
            return x/torch.norm(x)
        else:
            return x

def triangle_vec_to_lower(vec,N):

    # helper method for converting vector to lower triangular matrix
    tri_inds = torch.tril_indices(N,N)

    lower = torch.zeros(vec.shape[0],N,N).to(vec.device)
    lower[:,tri_inds[0,:],tri_inds[1,:]] = vec

    return lower

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class Unflatten(nn.Module):
    def __init__(self, im_size=2):
        super(Unflatten, self).__init__()
        self.im_size = im_size

    def forward(self, x):
        return x.view(x.size(0), -1, self.im_size, self.im_size)

class Upsample2d(nn.Module):
    def forward(self, x):
        return F.interpolate(x, scale_factor=2)

class StochConvAE(nn.Module):
    def __init__(self, in_channels: int, 
            width:      int = 8, 
            n_layers:   int = 4, 
            latent_dim: int = 2, 
            latent_im_size: int = 8, 
            fs: int = 5, 
            act = nn.LeakyReLU(), 
            pooling = nn.MaxPool2d(2), 
            loss = 'mc', 
            sigmoid = True, 
            use_skip = False, 
            add_det = False, 
            flow = None, 
            sig_x = 0.1, 
            sig_det = 1e-3, 
            sigma_type='full',
            use_conv = True,
            pz_var=100):

        '''
        This method initializes an autoencoder with parameters

        in_channels: number of input channels
        width     : multiple to increase the width of each conv layer
        n_layers  : number of levels of convolutions
        latent_dim: size of the latent dimension
        latent_dim_im_size: size of the downscaled image
        fs        : filter size
        act       : activation function
        pooling   : pooling layer
        loss      : either 'mc' or 'exact' for computing the KL divergence
        sigmoid   : last layer include sigmoid or not
        use_skip  : add a skip connection between the input and output layers
        add_det   : add deterministic component
        flow      : add normalizing flow (not yet implemented)
        sig_x     : variance on the decoder (how to penalize reconstruction loss)
        sig_det   : variance on deterministic part
        sigma_type: if sigma should be diagonal or full matrix
        pz_var    : variance on the prior z
        use_conv  : encoder conv or not
        '''

        super(StochConvAE,self).__init__()

        self.loss     = loss
        self.use_skip = use_skip
        self.add_det  = add_det
        self.sigma_type = sigma_type

        self.sig_det = sig_det
        self.sig_x   = sig_x

        if pz_var:
            self.pz = MultivariateNormal(torch.zeros(latent_dim,device='cuda'), torch.eye(latent_dim,device='cuda') * pz_var)
        else:
            self.pz = None

        self.latent_dim = latent_dim

        if use_conv:

            padding = math.floor(fs/2)

            enc_modules = [nn.Conv2d(in_channels, width, fs, padding = padding), act, pooling]
            dec_modules = [nn.Linear(latent_dim + add_det, width *8*latent_im_size**2), Unflatten(latent_im_size)]

            for i in range(1, n_layers):

                if i == n_layers - 1:
                    enc_modules += [nn.Conv2d(width * 2 ** (i - 1), width * 2 ** i, fs, padding = padding),
                            nn.BatchNorm2d( width * 2 ** i ), 
                            act]
                else:
                    enc_modules += [nn.Conv2d(width * 2 ** (i - 1), width * 2 ** i, fs, padding = padding),
                            act,
                            pooling]

            for i in range(n_layers - 1, 0, -1):

                dec_modules += [Upsample2d(),
                    nn.Conv2d(width * 2 ** i, width * 2 ** (i - 1), fs, padding = padding),
                    nn.BatchNorm2d(width * 2 ** (i - 1)),
                    act]

            enc_modules.append(Flatten())

            last_layer_width = width + in_channels if use_skip else width

            dec_last_layer = [nn.Conv2d(last_layer_width, last_layer_width, fs, padding = padding),
                    nn.Conv2d(last_layer_width, in_channels, fs, padding = padding)]

            if sigmoid:
                dec_last_layer.append(nn.Sigmoid())

            self.mu_linear    = nn.Linear(width * 8 * latent_im_size ** 2, latent_dim, bias=True)

            if loss == 'vae' or self.sigma_type == 'diag':
                self.sigma_linear = nn.Linear(width * 8 * latent_im_size ** 2, latent_dim, bias=True)
            elif self.sigma_type == 'const':
                self.eps = nn.Parameter(torch.randn(1)) * torch.eye(latent_dim)
            else:
                self.sigma_linear = nn.Linear(width * 8 * latent_im_size ** 2, int((latent_dim+1)*latent_dim/2), bias=True)

        else:

            enc_modules = [nn.Linear(in_channels, width), act]
            dec_modules = [nn.Linear(latent_dim + add_det, width)]

            for i in range(1, n_layers):

                enc_modules += [nn.Linear(width, width),
                        nn.BatchNorm1d(width),
                        act]
                dec_modules += [nn.Linear(width, width),
                        nn.BatchNorm1d(width), 
                        act]

            dec_last_layer = [nn.Linear(width, in_channels)]

            if sigmoid:
                dec_last_layer.append(nn.Sigmoid())

            self.mu_linear = nn.Linear(width, latent_dim, bias=True)

            if loss == 'vae' or self.sigma_type == 'diag':
                self.sigma_linear = nn.Linear(width, latent_dim, bias=True)
            elif self.sigma_type == 'const':
                self.eps = nn.Parameter(torch.randn(1)) * torch.eye(latent_dim)
            else:
                self.sigma_linear = nn.Linear(width, int((latent_dim+1)*latent_dim/2), bias=True)

        self.encoder = nn.Sequential( * enc_modules)
        self.decoder = nn.Sequential( * dec_modules)
        self.decoder_last = nn.Sequential( * dec_last_layer)
        self.latent_dim = latent_dim

        if add_det:
            self.det = nn.Linear(width*8 * latent_im_size **2, add_det)

        self.flow = flow

        init_weights(self.encoder, {'weight':'xavier', 'bias':'zeros'}, input_class=nn.Conv2d)
        init_weights(self.decoder, {'weight':'xavier', 'bias':'zeros'}, input_class=nn.Conv2d)

    def get_increments(self, q_mu, q_sigma, dt=None):
        '''
        Gets increments given a mu and sigma coming from the encoder
        returns the increments, the sampled latent points z, and the full sigma q
        '''

        if q_mu.shape == q_sigma.shape: # if sigma is diagonal, things are easier

            if dt:
                z = q_mu * dt + q_sigma * torch.randn_like(q_sigma).normal_(0, np.sqrt(dt))
            else:
                z = q_mu + q_sigma * torch.randn_like(q_sigma)

            inc = z[1:] - z[:-1]
            q_sigma_full = torch.diag_embed(q_sigma)

        else:

            # transform sigma
            lower_triangular_q = triangle_vec_to_lower(q_sigma, self.latent_dim)
            q_sigma_full       = torch.bmm(lower_triangular_q, lower_triangular_q.permute(0,2,1))

            if dt:
                epsilon  = torch.randn((q_mu.shape[0], q_mu.shape[1], 1)).normal_(0, np.sqrt(dt)).to(q_mu.device) 
            else:
                epsilon  = torch.randn((q_mu.shape[0], q_mu.shape[1], 1)).normal_(0, 1).to(q_mu.device) 

            z = q_mu + torch.bmm(q_sigma_full, epsilon).squeeze(2)

            # calculate the increments
            # NOTE: this assumes that data are sequential (do not shuffle)
            inc = z[1:] - z[:-1]

        return inc, z, q_sigma_full


    def get_next_z(self, z_init, ts, dt, mu, sigma):
        '''
        Method to compute z given mu and sigma functions, not encoder mu and sigma
        z_init : initial z to integrate from
        ts : time stamp
        dt : change in time
        mu : mu function
        sigma : sigma function
        returns z_n, one step integrated using the learned mu and sigma
        '''

        net_inputs = torch.cat( (ts.unsqueeze(1), z_init), dim=1)

        mu_hat, sigma_hat = utils.sample_mu_sigma(mu, sigma, net_inputs)

        if self.sigma_type == 'diag':
            z_n  = z_init + mu_hat * dt + sigma_hat * torch.randn_like(sigma_hat).normal_(0,np.sqrt(dt)) 

        elif self.sigma_type == 'const':
            z_n  = z_init + mu_hat * dt + self.eps * torch.randn_like(sigma_hat).normal_(0,np.sqrt(dt)) 

        else:
            lower_triangular = triangle_vec_to_lower(sigma_hat, self.latent_dim)
            sigma_hat_full   = torch.bmm(lower_triangular, lower_triangular.permute(0,2,1))
            epsilon          = torch.randn((mu_hat.shape[0], mu_hat.shape[1], 1)).normal_(0, np.sqrt(dt)).to(z_init.device) 
            z_n  = z_init + mu_hat * dt + torch.bmm(sigma_hat_full, epsilon).squeeze(2)

        return z_n

    def step(self, frames, ts, dt, mu, sigma, detach_ac=False, plus_one=False):
        '''
        The main function for training, predicts the next step

        frames: input frames, tensor sized (batch_size, n_channels, w, w)
        NOTE the batch size is also acting as the time index, it is assumed to be in order
        ts: time step
        mu: mu function
        sigma: sigma function
        detach_ac: when training if detaching the autoencoder should occur
        plus_one : if predicting the next step should occur
        '''

        # Get the parameters for the latent distributions
        q_mu, q_sigma, det = self.encode(frames)

        # get a sample from our estimated distribution
        if detach_ac:
            inc, z, q_sigma_full = self.get_increments(q_mu.detach(), q_sigma.detach())
        else:
            inc, z, q_sigma_full = self.get_increments(q_mu, q_sigma)

        # now we want to minimize the kl divergence with the
        # parameters of the SDE
        if self.loss == 'mc':
            net_inputs = torch.cat((ts[:-1].unsqueeze(1), z[:-1,:]), dim=1)
        else:
            net_inputs = torch.cat((ts[:-1].unsqueeze(1), q_mu[:-1,:]), dim=1)
        mu_hat, sigma_hat = utils.sample_mu_sigma(mu, sigma, net_inputs)

        # helpers to go from cholesky vector to full matrix
        if self.sigma_type =='diag':
            # if diagonal, keep it the same
            lower_triangular = sigma_hat
            sigma_hat_full = torch.diagonal(sigma_hat, dim1=-2, dim2=-1)
        else:
            lower_triangular = triangle_vec_to_lower(sigma_hat, self.latent_dim)
            sigma_hat_full = torch.bmm(lower_triangular, lower_triangular.permute(0,2,1))

        # Calculate losses
        if self.loss == 'mc':
            kl_loss = losses.kl_div_cholesky(mu_hat, lower_triangular, inc, dt)

        elif self.loss == 'exact':
            # since we assume gaussian parameterization
            # difference betwen mu is a new gaussian
            q_mu_inc = q_mu[1:] - q_mu[:-1]

            if self.sigma_type == 'diag': 
                sig_q = q_sigma
            else:
                lower_tri_q = triangle_vec_to_lower(q_sigma, self.latent_dim)
                sig_q       = torch.bmm(lower_tri_q, lower_tri_q.permute(0,2,1))

            # likewise, sum of variances is also gaussian
            sig_q_inc = sig_q[1:] + sig_q[:-1]

            # minimize the KL between the distribution from the encoder and the latent SDE
            # specifically, the increments of our multivariate gaussian should match the mu_hat 
            if detach_ac:
                kl_loss   = losses.kl_div_exact(q_mu_inc.detach(), sig_q_inc.detach(), mu_hat, lower_triangular, dt)
            else:
                kl_loss   = losses.kl_div_exact(q_mu_inc, sig_q_inc, mu_hat, lower_triangular, dt)

        elif self.loss == 'vae':
            # from appendix b of autoencoding variational bayes paper
            # note: q_sigma is assumed to be vector of diagonals
            kl_loss = -0.5 * torch.sum(1 + torch.log(q_sigma**2) - q_mu ** 2 - q_sigma ** 2)

        # add the prior 
        if self.pz:
            kl_loss += self.pz.log_prob(z).mean()

        # pass to the decoder
        conditional_frame = frames[0].unsqueeze(0).repeat(q_mu.size(0)-1,1,1,1)

        if plus_one:
            # We will use the mean as the current state
            # Then, we sample the next state according to the SDE
            z_step = self.get_next_z(q_mu, ts, dt, mu, sigma)
            if det is not None:
                det_samp = det[:-1] + torch.randn_like(det[:-1]).normal_(0,self.sig_det)
                decode_vec = torch.cat((z_step[:-1], det_samp), dim=1)
            else:
                decode_vec = z_step[:-1]
        else:
            if det is not None:

                det_samp = det + torch.randn_like(det).normal_(0,self.sig_det)
                decode_vec = torch.cat((z, det_samp),dim=1)
            else:
                decode_vec = z

        # after sampling the latent space reconstruct the image
        frames_hat = self.decode(decode_vec,x=conditional_frame)

        # reconstruction loss
        if plus_one:
            l2_loss = 0.5 * F.mse_loss(frames_hat, frames[1:]) / self.sig_x ** 2
        else:
            l2_loss = 0.5 * F.mse_loss(frames_hat, frames) / self.sig_x ** 2

        if det is not None:
            kl_loss += 0.5 * torch.sum(det**2) / self.sig_det **2 

        return kl_loss, l2_loss, frames_hat, mu_hat, q_mu, sigma_hat_full, q_sigma_full, inc, z

    def encode(self, x):
        '''
        Takes in a frame (x)
        Outputs mu and sigma for the frame
        '''
        latent = self.encoder(x)
        mu = self.mu_linear(latent)
        sigma = self.sigma_linear(latent)

        if self.flow:
            z = self.flow(z)

        det = None
        if self.add_det:
            det = self.det(latent)

        return mu, sigma, det

    def decode(self, z, x=None):
        up_to_last = self.decoder(z)
        if self.use_skip:
            x = self.decoder_last(torch.cat((up_to_last, x),dim=1))
        else:
            x = self.decoder_last(up_to_last)
        return x

    def forward(self, x, t):
        latent = self.encoder(x)
        x = self.decoder(latent)
        return x, latent
