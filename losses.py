####
## This file contains the various losses used in the code.
####

import torch
import numpy as np

def kl_div_cholesky(mu_hat, lower_triangular, inc, dt):

    # get size of latent space
    n_var = mu_hat.shape[1]

    if mu_hat.shape == lower_triangular.shape:

        inc_diff = ( inc - mu_hat * dt )
        kl = ( (inc_diff ** 2) / lower_triangular.abs() / dt ).mean() + (n_var * np.log(dt) + lower_triangular.abs().prod(1).log()).mean()
    else:
        # we're using cholesky solve to invert the matrix
        # so we want to solve AX= B where B is the identity matrix
        # in this case, A is our cholesky factorization
        B = torch.eye(n_var).unsqueeze(0).repeat(mu_hat.shape[0],1,1).to(mu_hat.device)
        sigma_inv = torch.cholesky_solve(B, lower_triangular)

        # difference between the increments and the mean
        inc_diff = ( inc - mu_hat * dt ).unsqueeze(2)

        # full term should be (x - \mu)T \sigma^-1 (x - \mu) + log(|\sigma|)
        kl = torch.bmm( inc_diff.permute(0,2,1), torch.bmm(1 / dt * sigma_inv, inc_diff) ).mean(0) \
                 + n_var * np.log(dt) + (lower_triangular.det()).log().mean(0)
    return kl


def kl_div_exact(q_mu, q_sigma, mu_hat, lower_triangular, dt):

    # get size of latent space
    n_var = mu_hat.shape[1]

    if q_mu.shape == q_sigma.shape:

        # this time we're using q_sigma as a diagonal matrix
        #mu_diff   = (mu_hat*dt - q_mu) 
        mu_diff   = (mu_hat - q_mu) 

        det_sig   = lower_triangular.abs().prod(1)
        det_sig_q = q_sigma.abs().prod(1)

        trace = (q_sigma.abs() / lower_triangular.abs()).sum(1)

        kl = 1/2 * ( trace + ( mu_diff ** 2 / lower_triangular.abs() ).sum(1)  \
                + det_sig.log() - det_sig_q.log() ).mean()

    else:

        # we're using cholesky solve to invert the matrix
        # so we want to solve AX= B where B is the identity matrix
        # in this case, A is our cholesky factorization
        B = torch.eye(n_var).unsqueeze(0).repeat(mu_hat.shape[0],1,1).to(mu_hat.device)
        sigma_inv = torch.cholesky_solve(B, lower_triangular)

        mu_diff = (mu_hat - q_mu).unsqueeze(2)

        det_sig = lower_triangular.det() ** 2
        det_sig_q = q_sigma.det() ** 2

        trace = (torch.diagonal(sigma_inv, dim1=-2, dim2=-1) * torch.diagonal(q_sigma, dim1=-2, dim2=-1) ** 2).sum(1)

        kl = 1/2 * ( trace + torch.bmm( mu_diff.permute(0,2,1), torch.bmm(sigma_inv, mu_diff) ) * dt \
                + det_sig.log() - det_sig_q.log() ).mean()

    return kl

def wass_dist(q_mu, q_sigma, mu_hat, sigma_hat):
    
    mu_part= (q_mu - mu_hat).norm(2)**2 
    #sqrt_q_sigma = q_sigma.cholesky()

    sigma_part = (q_sigma - sigma_hat).norm()**2

    w = mu_part + sigma_part

    return w
