####
## This function 
####
import torch
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import nets
import imageio
import numpy as np
from scipy.linalg import lstsq
import os
import pickle
import torch.nn as nn
from tqdm import tqdm

def triangle_vec_to_lower(vec,N):

    # helper method for converting vector to lower triangular matrix
    tri_inds = torch.tril_indices(N,N)

    lower = torch.zeros(vec.shape[0],N,N).to(vec.device)
    lower[:,tri_inds[0,:],tri_inds[1,:]] = vec

    return lower

def calc_affine(xt, 
        xt_hat, 
        savepath : str, 
        suffix : str ='train', 
        affine : bool = False, 
        gbm : bool  = False):

    '''
    Function to calculate the mapping between the original coordinates (xt) and the estimated (xt_hat).

    xt : array_like (len x dim) specifies original positions of latent SDE
    xt_hat : array_like (len x dim) specifies estimated positions of the latent SDE
    savepath : str, where to save the figure plotting the overlay of the xt_hat transformed ot the space of xt
    suffix : str, what to end the saved file with
    affine : bool, if to use an affine transformation (e.g. add scaling) or scaling pre applied
    gbm : bool, if to take a transformation of the coordinates 
    '''

    if gbm:
        # Take the log to get to a random walk 
        xt = np.log(xt)

    if affine:
        # calculate the best orthogonal map using least squares
        # augment the matrix
        xt_hat_aug = np.concatenate((xt_hat, np.ones((xt_hat.shape[0],1))),axis=1)

        # calc affine map 
        x, residues, rank, s = lstsq(xt_hat_aug, xt)
        bias = x[-1,:]
        mat  = x[:-1]
        u, s2, vh = np.linalg.svd(mat)

        Q = mat

        xt_hat_aug_t = (  ( xt_hat ) @ Q + bias ) # from latent to original 


        back = ( xt - bias ) @ np.linalg.inv(Q)
        mse = ( ( xt_hat_aug_t - xt ) ** 2 ).mean()
        rel_error = np.abs( (xt_hat_aug_t - xt ) / xt).mean() 

        if savepath:

            all_digits_plots = [] 
            all_digits_back  = [] 
            all_digits_titles = [r'$x_1$',r'$y_1$',r'$x_2$',r'$y_2$']
            all_labels = []

            for i in range(xt_hat_aug_t.shape[1]):
                all_digits_plots.append(np.stack((xt_hat_aug_t[:,i], xt[:,i]),axis=1))
                all_digits_back.append(np.stack((back[:,i], xt_hat[:,i]), axis=1))
                all_labels.append(['Estimated', 'Truth'])
            plot_subplots(all_digits_plots, all_digits_titles, os.path.join(savepath, 'transformed_all{}.pdf'.format(suffix)),
                    labels=all_labels,plot_type='plot',axis=True)
            plot_subplots(all_digits_back, all_digits_titles, os.path.join(savepath, 'transformed_back{}.pdf'.format(suffix)),
                    labels=all_labels,plot_type='plot',axis=True)

        return xt_hat_aug_t, Q, bias, mse, rel_error
    else:

        xtt     = xt.T
        xt_hatt = xt_hat.T

        K = xtt.shape[1]
        one = np.ones((K,1))

        bY = xtt.mean(1,keepdims=True)
        bX = xt_hatt.mean(1,keepdims=True)

        cY = xtt - bY 
        cX = xt_hatt - bX 

        u, s, vh = np.linalg.svd( cY @ cX.T )

        Q = (u @ vh)

        bias = ( bY - Q @ bX )

        xt_hat_aug_t = ( Q @ ( xt_hatt ) + bias ).T

        emp_scale = (xt_hat_aug_t[1:] / xt[1:]).mean()

        mse = ( ( xt_hat_aug_t - xt ) ** 2 ).mean()
        rel_error = np.abs( (xt_hat_aug_t - (xt+1e-5) ) / (xt+1e-5)).mean() 

        if savepath:
            all_digits_plots = [] 
            all_digits_titles = [r'$x_1$',r'$y_1$',r'$x_2$',r'$y_2$']
            all_digits_titles = [r'${}_{}$'.format(v,i) for i in range(xt_hat_aug_t.shape[1]) for v in ['x','y']]
            all_labels = []

            for i in range(xt_hat_aug_t.shape[1]):
                all_digits_plots.append(np.stack((xt_hat_aug_t[:,i], xt[:,i]),axis=1))
                all_labels.append(['Estimated', 'Truth'])
            plot_subplots(all_digits_plots, all_digits_titles, os.path.join(savepath, 'transformed_all{}.pdf'.format(suffix)),
                    labels=all_labels,plot_type='plot',axis=True)

        return xt_hat_aug_t, Q.T, bias.T, mse, rel_error

def get_DNA_centers(frames):
    centers = np.array([cv2.minMaxLoc(maximum_filter(skimage.filters.gaussian(x[0], sigma=3), size=(4,4)))[-1] for x in frames])
    return centers

def plot_subplots(tensors, titles, file_name, labels=None, plot_type='imshow', n_rows=1, axis=False, figsize=(20,10)):
    '''
    Helper method to plot multiple plots on the same figure
    '''

    fig = plt.figure(figsize=figsize)
    if type(plot_type) == str:
        plt_fn = getattr(plt, plot_type)

    for idx, tensor in enumerate(tensors):
        if type(plot_type) == list:
            plt_fn = getattr(plt, plot_type[idx])

        plt.subplot(n_rows, len(tensors), idx+1) 
        if type(tensor) == list:
            if plot_type == 'plot':
                lines = plt_fn(*tensor, linewidth=3)
            else:
                lines = plt_fn(*tensor)
        else:
            if plot_type == 'plot':
                lines = plt_fn(tensor, linewidth=3)
            else:
                lines = plt_fn(tensor)

        plt.title(titles[idx], fontsize=40)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        if not axis:
            plt.axis('off')
        if idx == len(tensors) - 1:
            if labels:
                plt.legend(lines, labels[idx], prop={'size':30})
    plt.tight_layout()
    if plot_type == 'imshow':
        fig.subplots_adjust(left=0.01, bottom=0.01, right=0.99, top=0.99)
    plt.savefig(file_name)
    plt.close('all')

def sample_mu_sigma(mu, sigma, net_inputs):
    '''
    Helper method to evaluate mu and sigma
    mu : matrix or nn.Module describing the drift
    sigma : matrix or nn.Module describing the diffusion
    net_inputs : tensor with size (t x dims) 
    '''

    if type(mu) == nets.MLP:
        width = mu.first_layer.linear.weight.shape[1]
        if width == net_inputs.shape[1]:
            mu_hat     = mu(net_inputs)
        else:
            mu_hat     = mu(net_inputs[:,1:])

    elif type(mu) == nets.Lin or type(mu) == nets.Well:
        mu_hat     = mu(net_inputs[:,1:])
    else:
        mu_hat = mu.unsqueeze(0).repeat(net_inputs.shape[0],1)
        
    sigma_hat = 0 

    if sigma is not None:
        if type(sigma) == nets.MLP:
            sigma_hat  = sigma(net_inputs)
        else:
            sigma_hat = sigma.unsqueeze(0).repeat(net_inputs.shape[0],1)

    return mu_hat, sigma_hat

def compare_mu2(mu_hat, xt_, ts, Q, b, dt, dataset, filename:str, N:int=50, affine:bool=False, loss_type:str='exact', gbm:bool=False, oracle:nn.Module=None):
    '''
    Compute errors between estimated drift and true drift
    Plots the estimated and true drift

    xt_: the latent xt that we're interested in
    ts: time stamp
    mu_hat: the estimated mu_hat
    Q: the orthogonal map from z to X
    b: bias for orthogonal map
    dataset: dataset for the experiment
    '''

    # get the useful variables for the rest of the routine
    N = xt_.shape[0]
    d = xt_.shape[1]

    scale = (dataset.xt_orig.max() - dataset.xt_orig.min())# ** (2/d)

    xt_plot = torch.zeros(N, d)

    # sample the original SDE used to generate the videos
    # this is only to plot
    for i in range(d):
        #lower = dataset.xt_orig[:,i].min().item() 
        #upper = dataset.xt_orig[:,i].max().item() 

        lower = dataset.xt_orig[:,i].mean().item() - dataset.xt_orig[:,i].std().item()
        upper = dataset.xt_orig[:,i].mean().item() + dataset.xt_orig[:,i].std().item() 

        xt_plot[:,i] = (torch.linspace(lower, upper, N).to(ts.device))

    # get the original sequence
    xt_calc = torch.Tensor(dataset.xt_orig[:N]).to(ts.device)
    
    # turn time to a tensor
    t_plot = torch.Tensor(dataset.ts[:N]).to(ts.device)

    q_max = xt_.max().detach().cpu().numpy()
    q_min = xt_.min().detach().cpu().numpy()
    mm = q_max - q_min

    # map it to the autoencoder latent space
    if affine:
        xt_latent_approx_plot = ( xt_plot.cpu().numpy() - b ) @ np.linalg.inv(Q)
        xt_latent_approx_calc = ( xt_calc.cpu().numpy() - b ) @ np.linalg.inv(Q) 
    else:
        if loss_type == 'exact':
            xt_latent_approx_plot = ( xt_plot.cpu().numpy() - b ) @ Q.T * mm / scale + q_min 
            xt_latent_approx_calc = ( xt_calc.cpu().numpy() - b ) @ Q.T * mm / scale + q_min 
        else:
            xt_latent_approx_plot = ( xt_plot.cpu().numpy() - b ) @ Q.T 
            xt_latent_approx_calc = ( xt_calc.cpu().numpy() - b ) @ Q.T 

    # sample the real mu function with the transformed domain
    mu_plot = dataset.mu(t_plot.cpu(), * [xt_plot[:,i].cpu() for i in range(d)])
    mu_calc = dataset.mu(t_plot.cpu(), * [xt_calc[:,i].cpu() for i in range(d)])

    if gbm:
        mu_plot = [mp / xt_plot[:,0] - 1 for mp in mu_plot]
        mu_calc = [mc / xt_calc[:,0] - 1 for mc in mu_calc]
        #mu_calc = np.stack(mu_calc,1) / xt_calc

    # turn the latent estimation into a tensor
    xt_latent_approx_plot = torch.Tensor( xt_latent_approx_plot ).to(xt_.device)
    xt_latent_approx_calc = torch.Tensor( xt_latent_approx_calc ).to(xt_.device)

    # setup the tensor for sampling the learned mu
    xt_dom_plot = torch.cat((t_plot.unsqueeze(1), xt_latent_approx_plot), dim=1)
    xt_dom_calc = torch.cat((t_plot.unsqueeze(1), xt_latent_approx_calc), dim=1)


    # scale the latent data back to the original space    
    mu_approx_plot = (( (sample_mu_sigma(mu_hat, None, xt_dom_plot )[0].detach().cpu().numpy()) )[:,:Q.shape[0]] @ Q ) *scale  
    mu_approx_calc = (( (sample_mu_sigma(mu_hat, None, xt_dom_calc )[0].detach().cpu().numpy()) )[:,:Q.shape[0]] @ Q ) *scale 

    # housekeeping things for sympy integration
    for idx, muv in enumerate(mu_plot):
        if type(muv) == float or type(muv) == int:
            mu_plot[idx] = muv * torch.ones(N)
            mu_calc[idx] = muv * torch.ones(N)

    all_comp_plots  = [] 
    all_comp_titles = [r'$\mu_0(x)$',r'$\mu_0(y)$',r'$\mu_1(x)$',r'$\mu_1(y)$']
    all_comp_titles = [r'$\mu_{}({})$'.format(i,v) for i in range(d) for v in ['x','y']]
    all_labels      = []

    # make the sympy output into a tensor
    mu_plot = ( torch.stack(mu_plot, 1) ).cpu().numpy()
    mu_calc = ( torch.stack(mu_calc, 1) ).cpu().numpy()
    
    if d > 4:
        figsize = (40, 10)
    else:
        figsize = (20, 10)

    # plot the results
    if oracle is not None:
        mu_oracle_plot = oracle(xt_plot).cpu().detach()
        for i in range(mu_approx_plot.shape[1]):
            all_comp_plots.append([xt_plot[:,i].cpu(), np.stack((mu_approx_plot[:,i], mu_plot[:,i], mu_oracle_plot[:,i]), axis=1)])
            all_labels.append(['Estimated', 'Truth', 'Oracle'])

    else:
        for i in range(mu_approx_plot.shape[1]):
            all_comp_plots.append([xt_plot[:,i].cpu(), np.stack((mu_approx_plot[:,i], mu_plot[:,i]), axis=1)])
            all_labels.append(['Estimated', 'Truth'])
            
    if filename:

        plot_subplots(all_comp_plots, 
                all_comp_titles, 
                filename, 
                labels=all_labels, 
                plot_type='plot', 
                figsize=figsize,
                axis=True)

    # calc metrics!
    mse      = ( (mu_calc - mu_approx_calc) ** 2 ).mean()
    mse_rel  = np.abs((mu_calc - mu_approx_calc) / (mu_calc + 1e-9)).mean()
    mse_crlb = ( (mu_calc - mu_approx_calc) ** 2 ).mean()

    return mse, mse_rel, mse_crlb, all_comp_plots, all_comp_titles


def plot_mu_hat(mu, sigma, xt, ts, filename, N=50):

    xt_plot = []
    xt_plot_mean= []

    for i in range(xt.shape[1]):

        lower = xt[:,i].min().item() 
        upper = xt[:,i].max().item() 
        mean = xt[:,i].mean().item()
        xt_plot.append(torch.linspace(lower, upper, N).to(ts.device))
        xt_plot_mean.append(mean * torch.ones(N).to(ts.device))

    t_plot  = torch.linspace(ts.min().item(), ts.max().item(), N).to(ts.device)

    plot_data  = torch.stack([t_plot, *xt_plot], dim = 1)
    mu_hat, _ = sample_mu_sigma(mu, sigma, plot_data)

    plot_subplots([mu_hat.detach().cpu().numpy()],
            ['mu_hat'], 
            filename,
            plot_type='plot',
            axis=True)

    return mu_hat, plot_data

def plot_mu_points(xt, ts, dataset, N=50):

    xt_plot = []
    xt_plot_mean= []

    for i in range(xt.shape[1]):

        lower = xt[:,i].min().item() 
        upper = xt[:,i].max().item() 

        xt_plot.append(torch.linspace(lower, upper, N))

    t_plot  = torch.linspace(ts.min().item(), ts.max().item(), N)
    mu_val = dataset.mu(t_plot,*xt_plot)

    # housekeeping things for sympy integration
    for idx, muv in enumerate(mu_val):
        if type(muv) == float or type(muv) == int:
            mu_val[idx] = muv * np.ones(N)

    return np.stack(mu_val, 1), torch.stack((t_plot, *xt_plot),1)

def plot_mu(dataset, batch, batch_size, N=50):

    xt_plot = []
    xt_plot_mean= []

    for i in range(dataset.xt_orig.shape[1]):

        lower = dataset.xt_orig[batch*batch_size:(batch+1)*batch_size,i].min()
        upper = dataset.xt_orig[batch*batch_size:(batch+1)*batch_size,i].max()

        xt_plot.append(torch.linspace(lower, upper, N))

    t_plot  = torch.linspace(dataset.ts.min().item(), dataset.ts.max().item(), N)
    mu_val = dataset.mu(t_plot,*xt_plot)

    # housekeeping things for sympy integration
    for idx, muv in enumerate(mu_val):
        if type(muv) == float or type(muv) == int:
            mu_val[idx] = muv * np.ones(N)

    return np.stack(mu_val, 1), torch.stack((t_plot, *xt_plot),1)

    
def save_gif(frames, path):
    def scale(x):
        return (x - x.min()) / (x.max() - x.min())

    f_split = np.split(frames.permute(0,2,3,1), frames.shape[0])
    imageio.mimsave(path, [(255*scale(x.squeeze(0))).type(torch.ByteTensor) for x in f_split])

def solve_oracle_problem(xt, t, niter=1000):

    xt = torch.tensor(xt).float().to(t.device)

    if len(xt.shape) == 2:
        xt = xt.unsqueeze(0)

    inc = xt[:,1:] - xt[:,:-1]
    d   = xt.shape[-1]
    b   = xt.shape[0]

    dt = t[0,1] - t[0,0]

    mu = nets.MLP(d, 64, 4, d, activation='nn.Softplus').to(t.device)
    mu.train()

    p = nn.Parameter(torch.randn(3, 1))

    #mu = lambda x: torch.stack((x**2, x, torch.ones_like(x)),2) @ p 

    opt = torch.optim.Adam(mu.parameters(), lr=1e-4)
    #opt = torch.optim.Adam([p], lr=1e-4)

    for iter in tqdm(range(niter)):
        for bidx in range(b):

            opt.zero_grad()

            m = mu(xt[bidx, :-1])

            diff = -( ( inc[bidx]  - m * dt ) / dt.sqrt() ) ** 2

            loss = -diff.mean()

            loss.backward()
            opt.step()

            if iter % 100 == 0 and bidx == 0:
                print('loss : {}' .format(loss))

    return mu
