import os
import io
import numpy as np
from PIL import Image

from skimage.transform import resize
from skimage.io import imread 
import skimage.filters

from scipy.ndimage import maximum_filter

import cv2

import pandas as pd

import torch
import torchvision.transforms.functional as TF
import torchvision.datasets as datasets

import sim_process
import sympy

import matplotlib.pyplot as plt
import pickle



class SDEImageDataset(torch.utils.data.Dataset):

    def __init__(self, fcn_mu: str, fcn_sigma: str, 
            normalize:  bool = True, 
            path:       str = 'data', 
            im_type:    str = 'balls', 
            num_traj:   int = 1,
            add_noise:  bool = False,
            **params):
        super(SDEImageDataset).__init__()
        '''
        SDEImageDataset provides a dataset for videos of yellow balls
        moving according to an ito process given by fcn_mu and fcn_sigma

        fcn_mu    : string that describes the drift
        fcn_sigma : string that describes the diffusion
        normalize : boolean to normalize the images
        im_type   : either 'wass', 'mnist', or 'balls'. defaults to 'balls'
        num_traj  : number of times to simulate the process
        '''

        self.mu        = fcn_mu
        self.sigma     = fcn_sigma

        self.path      = path
        self.normalize = normalize

        self.num_traj = num_traj
        self.params   = params
        self.im_type  = im_type

        try:
            # if the data exists, load it  
            with open(path,'rb') as f:
                data = pickle.load(f)

            self.num_traj = data['num_traj']
            self.imgs     = data['imgs']
            self.params   = data['params']
            self.scale    = data['scale']
            
            self.xt_orig  = data['xt_orig']
            self.ts       = data['ts']
            self.xt       = data['xt']
            self.frames   = data['frames']
            try:
                self.orig_frames = data['original_frames']

            except KeyError:
                print('No original data found, assuming data not corrupt with noise')

            if 'dna' in path:
                self.frames = (self.frames - self.frames.mean(0)) / (self.frames.std(0))

        except FileNotFoundError:
            # otherwise, generate new data
            data = {}

            if im_type == 'mnist':
                self.imgs = get_digits(num_dig=2)
            elif im_type == 'wass':
                self.imgs = get_coil(num_img=2)
            elif im_type == 'ballwass':
                self.imgs = get_coil(num_img=2)
            else:
                self.imgs = None

            self.saved_frames = []
            self.saved_traj   = []

            all_xt = []
            all_ts = []
            all_frames = []


            # for each run simulate a stochastic process
            for i in range(self.num_traj):

                if 'dna' in im_type:
                    print(im_type)
                    try:
                        xt, ts, frames = get_dna_movie_ind(im_type[-1],i)
                    except TypeError:
                        print('high var')
                        continue
                    all_frames.append(frames)

                else:
                    xt, ts = sim_process.sim_process_multi(
                            self.mu, 
                            self.sigma, 
                            **params)

                if 'gbm' in path:
                    while True:
                        xt, ts = sim_process.sim_process_multi(
                                self.mu, 
                                self.sigma, 
                                **params)
                        if (xt > 3).sum() == 0:
                            print('Less than three')
                            print(xt.max())
                            break

                    xt = np.exp(xt)

                if im_type == 'wass':
                    all_xt.append(xt[:,0].reshape(-1,1))
                else:
                    all_xt.append(xt)
                all_ts.append(ts)

            self.num_traj = len(all_frames)

            # combine all runs
            self.xt_orig = np.concatenate(all_xt)
            self.ts      = np.concatenate(all_ts)


            # scale the process
            xt_scaled    = ( self.xt_orig - self.xt_orig.min() ) / (self.xt_orig.max() - self.xt_orig.min())
            self.xt = xt_scaled

            # store the scaling factor
            self.scale = self.xt_orig.max() - self.xt_orig.min()
            
            # generate the movie
            if im_type == 'wass':
                print('frames')
                self.frames = sim_process.gen_movie_fmnist_given(self.xt, self.ts, 
                        imgs=self.imgs, 
                        **self.params)
            elif im_type == 'mnist':
                self.frames = sim_process.gen_movie_mnist_given(self.xt, self.ts, self.imgs, **params)
            elif im_type == 'ballwass':
                bg_frames = sim_process.gen_movie_fmnist_given(np.expand_dims(self.xt[:,0],1), self.ts, 
                        imgs=self.imgs, 
                        **self.params)
                self.frames = sim_process.gen_movie_given(self.xt[:,1:], self.ts, None, bg_frames=bg_frames,**params)
            elif 'dna' in im_type:
                self.frames = torch.cat(all_frames)
            elif im_type == 'Mball':
                n = self.xt.shape[1]
                assert n % 2 == 0, 'must be even number of trajectories.'
                for ball_num in range(n // 2):
                    if ball_num == 0:
                        self.frames = sim_process.gen_movie_given_c(self.xt[:,ball_num:ball_num+2], self.ts, self.imgs, **params)
                    else:
                        self.frames += sim_process.gen_movie_given_c(self.xt[:,ball_num:ball_num+2], self.ts, self.imgs, **params)
            else: 
                print('ball frames')
                self.frames = sim_process.gen_movie_given(self.xt, self.ts, self.imgs, **params)

            if add_noise:
                data['original_frames'] = self.frames.copy()
                if add_noise == 'heavy':
                    self.frames = self.frames + np.random.standard_t(3, self.frames.shape)
                else:
                    if 'mnist' in im_type:
                        self.frames = self.frames + np.random.randn(*self.frames.shape) * np.sqrt(0.25) #* 2 #* np.sqrt(3)
                    else:
                        self.frames = self.frames + np.random.randn(*self.frames.shape) * 2

            data['xt_orig'] = self.xt_orig
            data['ts']      = self.ts 
            data['xt']      = self.xt
            data['frames']  = self.frames
            data['scale']   = self.scale


            data['num_traj'] = self.num_traj
            data['imgs']     = self.imgs
            data['params']   = self.params


            with open(path, 'wb') as f:
                pickle.dump(data, f)

            if 'dna' in path:
                self.frames = (self.frames - self.frames.mean(0)) / (self.frames.std(0))
        try:
            comp_mu = params['real_mu'] 
            x = sympy.symbols([x for x in ['t','x','y','z']])
            comp_mu = sympy.sympify(comp_mu)
            comp_mu = sympy.lambdify(x, comp_mu)
            self.mu = comp_mu
            #self.xt_orig[:,-1]  = 2 * np.sqrt(self.xt_orig[:,-1])
            #A = np.linalg.inv(np.array([[1,2],[2.5,3]]))
            #self.xt_orig[:,:-1] = self.xt_orig[:,:-1] @ A
        except KeyError:
            print('Using original mu as computed mu')

        if 'dna' in im_type:
            emp_mean = (self.xt_orig[1:] - self.xt_orig[:-1]).mean(0)
            emp_cov  = np.cov(self.xt_orig[1:] - self.xt_orig[:-1], rowvar=False)
            print('Empirical Mean')
            print(emp_mean)
            print('Empirical Covariance')
            print(emp_cov)

    def __len__(self):
        return self.frames.shape[0] #self.params['n_points'] * self.num_traj

    def __getitem__(self, index):
        frame = torch.Tensor(self.frames[index,:,:,:])

        if self.normalize:
            TF.normalize(frame,[0,0,0],[1,1,1],inplace=True)
            
        return frame, self.ts[index]

class BallDataset(SDEImageDataset):

    def __init__(self, fcn_mu: str, fcn_sigma: str, 
            normalize: bool = True,
            path:      str = 'data', 
            num_traj:  int = 1, **params):

        mu_s      = sympy.sympify(fcn_mu)
        sigma_s   = sympy.sympify(fcn_sigma)
        x         = sympy.symbols([x for x in ['t','x','y']])

        mu        = sympy.lambdify(x,mu_s)
        sigma     = sympy.lambdify(x,sigma_s)

        super(BallDataset, self).__init__(mu, sigma, normalize, path, 'ball', num_traj, **params)

class MBallDataset(SDEImageDataset):

    def __init__(self, fcn_mu: str, fcn_sigma: str, 
            n: int = 5,
            normalize: bool = True,
            path:      str = 'data', 
            num_traj:  int = 1, **params):

        mu_s      = sympy.sympify(fcn_mu)
        sigma_s   = sympy.sympify(fcn_sigma)
        var = ['{}{}'.format(v,i) for i in range(n) for v in ['x','y']]
        var.insert(0,'t')

        x         = sympy.symbols([x for x in var])
        mu        = sympy.lambdify(x,mu_s)
        sigma     = sympy.lambdify(x,sigma_s)

        super(MBallDataset, self).__init__(mu, sigma, normalize, path, 'Mball', num_traj, **params)

class SMNISTDynamicDataset(SDEImageDataset):

    def __init__(self, fcn_mu: str, fcn_sigma: str, 
            normalize: bool = True,
            path:      str = 'data', 
            num_traj:  int = 10, 
            **params):

        mu_s      = sympy.sympify(fcn_mu)
        sigma_s   = sympy.sympify(fcn_sigma)
        x         = sympy.symbols([x for x in ['t','x','y','z','a']])
        mu        = sympy.lambdify(x,mu_s)
        sigma     = sympy.lambdify(x,sigma_s)
        super(SMNISTDynamicDataset, self).__init__(mu, sigma, normalize, path, 'mnist', num_traj, **params)

class BallWassDataset(SDEImageDataset):

    def __init__(self, fcn_mu: str, fcn_sigma: str, 
            normalize: bool = True,
            path:      str = 'data', 
            num_traj:  int = 1, 
            **params):

        mu_s      = sympy.sympify(fcn_mu)
        sigma_s   = sympy.sympify(fcn_sigma)
        x         = sympy.symbols([x for x in ['t','x','y','z']])
        mu        = sympy.lambdify(x,mu_s)
        sigma     = sympy.lambdify(x,sigma_s)
        super(BallWassDataset, self).__init__(mu, sigma, normalize, path, 'ballwass', num_traj, **params)

class DNADataset(SDEImageDataset):

    def __init__(self, fcn_mu: str, fcn_sigma: str, 
            name: str, 
            normalize: bool = True,
            path:      str = 'data', 
            #num_traj:  int = 60, 
            **params):

        mu_s      = sympy.sympify(fcn_mu)
        sigma_s   = sympy.sympify(fcn_sigma)
        #x         = sympy.symbols([x for x in ['t','x','y','z']])
        x         = sympy.symbols([x for x in ['t','x','y']])
        mu        = sympy.lambdify(x,mu_s)
        sigma     = sympy.lambdify(x,sigma_s)
        super(DNADataset, self).__init__(mu, sigma, normalize, path, name, **params)

class COILDataset(SDEImageDataset):

    def __init__(self, fcn_mu: str, fcn_sigma: str,
            load_path:      str, 
            normalize: bool = True, 
            **params):

        mu_s      = sympy.sympify(fcn_mu)
        sigma_s   = sympy.sympify(fcn_sigma)
        x         = sympy.symbols([x for x in ['t','x']])

        mu    = sympy.lambdify(x,mu_s)
        sigma = sympy.lambdify(x,sigma_s)

        num_traj = 1
        super(COILDataset, self).__init__(mu, sigma, normalize, load_path, 'wass', num_traj, **params)

            
class VectorDataset(torch.utils.data.Dataset):

    def __init__(self, fcn_mu, fcn_sigma, **params):
        super(VectorDataset, self).__init__()

        mu_s      = sympy.sympify(fcn_mu)
        sigma_s   = sympy.sympify(fcn_sigma)
        x         = sympy.symbols([x for x in ['t','x','y','z']])

        self.mu    = sympy.lambdify(x,mu_s)
        self.sigma = sympy.lambdify(x,sigma_s)
        xt, ts = sim_process.sim_process_multi(
                self.mu, 
                self.sigma, 
                **params)

        self.A = torch.randn(len(x)-1, params['nc'])

        self.frames = torch.tensor(xt).float() @ self.A 
        self.xt_orig = xt
        self.xt = xt 
        self.ts = ts

    def __len__(self):
        return self.frames.shape[0]

    def __getitem__(self, idx):
        return self.frames[idx, :], self.ts[idx]


def get_digits(path: str,train=True, num_dig = 2, im_size = (16, 16)):

    from skimage.transform import resize

    digits = datasets.MNIST(path, train=train, download=False)
    data = [np.array(img, dtype=np.uint8) for i, (img, label) in enumerate(digits)]
    print(data[0].shape)

    images = [data[i] for i in np.random.randint(0,len(data),num_dig)]
    images = [resize(image, im_size) for image in images]

    return images

def get_dna_movie_ind(base_ind, ind, path='data/DNA/'):

    '''
    Returns the movie of the DNA for the particular index. 
    '''
    import fitsio

    w = 48 # bounding box

    filename = os.path.join(path, '{}V-{}'.format(base_ind, ind))
    ref_filename = os.path.join(path, '{}V-0'.format(base_ind))

    data, h = fitsio.read(filename, header=True)
    ref, h  = fitsio.read(ref_filename, header=True)

    datac = (data - data.mean(0)).copy() # create two normalizations, one for finding the center
    datac = ((datac - datac.min()) / (datac.max() - datac.min())).copy()

    data  = ((data - data.min()) / (data.max() - data.min())).copy() # the other for the network to see

    f_split  = np.split(data, data.shape[0])
    f_split  = [skimage.exposure.equalize_adapthist(x[0], clip_limit=0.03) for x in f_split] # equalize histogram for the network

    f_splitc = np.split(datac, datac.shape[0]) 

    # create a first pass at where the center is
    centers_1     = np.array([cv2.minMaxLoc(maximum_filter(skimage.filters.gaussian(x[0], sigma=3), size=(4,4)))[-1] for x in f_splitc])
    mean_center_1 = np.floor(centers_1.mean(0)).astype(np.int).copy() 
    std_center_1  = centers_1.std(0)

    # do a second pass constrained on the first pass
    centers     = np.array([cv2.minMaxLoc(maximum_filter(skimage.filters.gaussian(x[0,mean_center_1[1]-w:mean_center_1[1]+w, mean_center_1[0]-w:mean_center_1[0]+w], sigma=3), size=(3,3)))[-1] for x in f_splitc])
    mean_center = np.floor(centers.mean(0)).astype(np.int) 
    std_center  = centers.std(0)

    xt = (centers - mean_center) / w
    ts = torch.linspace(0, xt.shape[0] / 2, xt.shape[0])

    cropped = [x[mean_center_1[1]-w:mean_center_1[1]+w, mean_center_1[0]-w:mean_center_1[0]+w] for x in f_split]
    resized = [cv2.resize(x, dsize=(64,64), interpolation=cv2.INTER_LANCZOS4) for x in cropped]

    frames  = np.stack(resized)
    frames = torch.Tensor(frames).unsqueeze(1).repeat(1,3,1,1)
    return xt, ts, frames


def get_coil(path:str, num_img=2, im_size=(64,64)):
    obj_num = [4, 1, 13, 16] 
    obj_num = np.random.choice(range(1,21), num_img, replace=False)
    obj_ver = [1, 1, 1, 1] 

    from skimage.transform import resize
    im_strings = [os.path.join(path, 'obj{}__{}.png'.format(obj_num[i], obj_ver[i])) for i in range(num_img)]
    images = [resize(imread(im_path), im_size) for im_path in im_strings]

    return images

if __name__ == '__main__':

    ds = FMNISTWassDataset('[0,0,0,0]',list(5*np.eye(4).reshape(1,-1)[0]),x_init=np.random.randn(4),n_points=100)
    dl = torch.utils.data.DataLoader(ds, num_workers=0, batch_size = 100)
    for (frames, ts) in dl:
        import utils
        utils.save_gif(frames[:100].detach().cpu(), 'wass_test.gif')

    

