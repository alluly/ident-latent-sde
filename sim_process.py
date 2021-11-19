import matplotlib
matplotlib.use('agg')
import numpy as np
import sympy
from scipy.stats import gaussian_kde,norm
from scipy.optimize import curve_fit 
import matplotlib.pyplot as plt
import time
from tqdm import tqdm

def sim_process_multi(mu,sigma,**params):
    def dW(delta_t,n_vars,flight=False): 
        """Sample a random number at each call."""
        if flight:
            return np.random.levy(loc = np.zeros((n_vars,1)), scale = np.sqrt(delta_t))
        return np.random.normal(loc = np.zeros((n_vars,1)), scale = np.sqrt(delta_t))

    N      = params.setdefault('n_points', 100)
    t_init = params.setdefault('t0',     0)
    t_end  = params.setdefault('tn',     5)
    x_init = params.setdefault('x_init',     [0,0])

    n_vars = len(x_init)

    dt = (t_end - t_init) / N

    ts= np.arange(t_init, t_end, dt)

    xs = np.zeros( (N,n_vars) )
    xs[0,:] = np.array(x_init)

    for i in range(1, ts.size):

        t = (i-1) * dt

        x = np.array( xs[i-1, :] )

        xs[i,:] = x + np.array(mu(t,*x)) * dt + np.squeeze(np.array(sigma(t,*x)).reshape(-1,n_vars) @ dW(dt, n_vars))

    return xs, ts

def gen_movie_given_c(xt, ts, imgs, w=64, r=16, nc=3, bg_frames=None, **params):

    '''
    Generates a movie given an SDE, xt and time stamp, ts

    xt : N x 2 vector denoting the position of the object at N times
    ts : N x 1 vector denoting the time 
    w  : width of the image
    r  : radius of the ball  
    nc : number of channels

    returns the movie as a numpy array
    '''

    if imgs is not None:
        return gen_movie_mnist_given(xt, ts, imgs, **params)

    # generate the ball
    gX, gY = np.meshgrid(np.linspace(-1,1,16), np.linspace(-1,1,16))
    ball = (gX **2 + gY **2 < 1)

    cvar = np.random.rand(3)
    cvar = np.array([1,0.5,0])

    # setup the frames
    if bg_frames is None: 
        x_frames = np.zeros((len(ts), nc, w, w))
    else:
        x_frames = bg_frames.copy()


    # generate the movie 
    for idx in range(ts.size):

        idx1 = (xt[idx, :] * (w - r)).astype(int)

        x_frames[idx, 2, idx1[0]:idx1[0] + ball.shape[0], idx1[1]:idx1[1] + ball.shape[1]] += ball*cvar[0]
        x_frames[idx, 1, idx1[0]:idx1[0] + ball.shape[0], idx1[1]:idx1[1] + ball.shape[1]] += ball*cvar[1]
        x_frames[idx, 0, idx1[0]:idx1[0] + ball.shape[0], idx1[1]:idx1[1] + ball.shape[1]] += ball*cvar[2]

        x_frames[x_frames > 255] = 255

    return x_frames

def gen_movie_given(xt, ts, imgs, w=64, r=16, nc=3, bg_frames=None, **params):

    '''
    Generates a movie given an SDE, xt and time stamp, ts

    xt : N x 2 vector denoting the position of the object at N times
    ts : N x 1 vector denoting the time 
    w  : width of the image
    r  : radius of the ball  
    nc : number of channels

    returns the movie as a numpy array
    '''

    if imgs is not None:
        return gen_movie_mnist_given(xt, ts, imgs, **params)

    # generate the ball
    gX, gY = np.meshgrid(np.linspace(-1,1,16), np.linspace(-1,1,16))
    ball = (gX **2 + gY **2 < 1)

    # setup the frames
    if bg_frames is None: 
        x_frames = np.zeros((len(ts), nc, w, w))
    else:
        x_frames = bg_frames.copy()


    # generate the movie 
    for idx in range(ts.size):

        idx1 = (xt[idx, :] * (w - r)).astype(int)

        x_frames[idx, 1, idx1[0]:idx1[0] + ball.shape[0], idx1[1]:idx1[1] + ball.shape[1]] += ball
        x_frames[idx, 0, idx1[0]:idx1[0] + ball.shape[0], idx1[1]:idx1[1] + ball.shape[1]] += ball

        x_frames[x_frames > 255] = 255

    return x_frames

def gen_movie_mnist_given(xt, ts, imgs, w=64, r=16, nc=3, **params):

    # Get two images from mnist
    # Put them in the appropriate location from the SDE
    # Move them each frame

    assert xt.shape[1] == 2*len(imgs), 'xt should be 2 times the number of digits'

    x_frames = np.zeros((len(ts), nc, w, w))

    img1 = imgs[0]
    img2 = imgs[1]

    for idx in range(ts.size):

        idx1 = (xt[idx, :2] * (w - r)).astype(int)
        idx2 = (xt[idx, 2:] * (w - r)).astype(int)

        x_frames[idx, 0, idx1[0]:idx1[0] + img1.shape[0], idx1[1]:idx1[1] + img1.shape[1]] += img1
        x_frames[idx, 0, idx2[0]:idx2[0] + img2.shape[0], idx2[1]:idx2[1] + img2.shape[1]] += img2

        x_frames[x_frames > 255] = 255

    return x_frames

def gen_movie_fmnist_given(xt, ts, imgs, nc=3, **params):
    import ot
    reg = 0.004

    basis    = np.eye(len(imgs))

    imgs_np  = np.zeros((len(imgs), imgs[0].shape[0], imgs[0].shape[1]))
    x_frames = np.zeros((len(ts), nc, imgs[0].shape[0], imgs[0].shape[1]))

    for idx, img in enumerate(imgs):
        imgs_np[idx,:,:] = img / img.sum()


    for idx in tqdm(range(ts.size)):

        x = xt[idx,0]

        weights = (1 - x) * basis[0,:] + x * basis[1,:]

        frame = ot.bregman.convolutional_barycenter2d(imgs_np, reg, weights)
        x_frames[idx, :, :, :] = frame / frame.max()

    return x_frames 

def gen_movie(mu, sigma, w=64, r=16, nc=3, **params):

    xt_orig, ts = sim_process_multi(mu, sigma, **params)

    gX, gY= np.meshgrid(np.linspace(-1,1,16), np.linspace(-1,1,16))

    ball = (gX **2 + gY **2 < 1)

    xt = ( xt_orig - xt_orig.min()) / (xt_orig.max() - xt_orig.min()) 

    scale = (xt_orig.max() - xt_orig.min())
    #gX, gY = np.meshgrid(np.linspace(-1,1,w), np.linspace(-1,1,w))

    x_frames = np.zeros((len(ts), nc, w, w))

    for idx in range(ts.size):

        #x_frames[idx, :2, :, :] = ( (gX - xt[idx,0])**2 + (gY - xt[idx,1])**2 < r**2 )
        #x_frames[idx, :, :, 1] = np.abs(gX - xt[idx,0]) + np.abs(gY - xt[idx,1]) < r*2

        idx1 = (xt[idx, :] * (w - r)).astype(int)

        x_frames[idx, 1, idx1[0]:idx1[0] + ball.shape[0], idx1[1]:idx1[1] + ball.shape[1]] += ball
        x_frames[idx, 0, idx1[0]:idx1[0] + ball.shape[0], idx1[1]:idx1[1] + ball.shape[1]] += ball

        #x_frames[x_frames > 255] = 255
        #print(idx)

    return x_frames, ts, xt, xt_orig


def gen_movie_mnist(mu, sigma, imgs, w=64, r=16, nc=3, **params):

    # Get two images from mnist
    # Put them in the appropriate location from the SDE
    # Move them each frame

    xt, ts = sim_process_multi(mu, sigma, **params)

    assert xt.shape[1] == 2*len(imgs), 'xt should be 2 times the number of digits'

    xt = ( xt - xt.min()) / (xt.max() - xt.min()) 

    x_frames = np.zeros((len(ts), nc, w, w))

    img1 = imgs[0]
    img2 = imgs[1]

    for idx in range(ts.size):

        idx1 = (xt[idx, :2] * (w - r)).astype(int)
        idx2 = (xt[idx, 2:] * (w - r)).astype(int)

        x_frames[idx, 0, idx1[0]:idx1[0] + img1.shape[0], idx1[1]:idx1[1] + img1.shape[1]] += img1
        x_frames[idx, 0, idx2[0]:idx2[0] + img2.shape[0], idx2[1]:idx2[1] + img2.shape[1]] += img2

        x_frames[x_frames > 255] = 255


    return x_frames, ts, xt


def gen_movie_fmnist(mu, sigma, imgs, nc=3, **params):
    import ot
    reg = 0.004

    xt_orig, ts = sim_process_multi(mu, sigma, **params)

    xt = ( xt_orig - xt_orig.min()) / (xt_orig.max() - xt_orig.min()) 

    basis    = np.eye(len(imgs))

    imgs_np  = np.zeros((len(imgs), imgs[0].shape[0], imgs[0].shape[1]))
    x_frames = np.zeros((len(ts)+1, nc, imgs[0].shape[0], imgs[0].shape[1]))

    for idx, img in enumerate(imgs):
        imgs_np[idx,:,:] = img / img.sum()

    plot_example_wass(imgs_np)

    for idx in range(ts.size):

        x = xt[idx,0]
        #y = xt[idx,1]

        weights = (1 - x) * basis[0,:] + x * basis[1,:]

        frame = ot.bregman.convolutional_barycenter2d(imgs_np, reg, weights)
        x_frames[idx, :, :, :] = frame / frame.max()

    return x_frames, ts, xt[:,0].reshape(-1,1), xt_orig[:,0]

def plot_example_wass(imgs,n_examples=5):
    import ot

    reg = 0.004

    basis    = np.eye(imgs.shape[0])

    plt.subplot(1, n_examples + 1, 1)
    plt.imshow(imgs[0], cmap='gray')
    plt.axis('off')
    for i in range(n_examples):
        plt.subplot(1, n_examples + 1, i + 2)
        tx = float(i) / (n_examples - 1)
        weights = (1 - tx) * basis[0,:] + tx * basis[1,:]
        frame = ot.bregman.convolutional_barycenter2d(imgs, reg, weights)
        plt.imshow(frame, cmap='gray')
        plt.axis('off')
    plt.subplot(1, n_examples + 1, n_examples + 1)
    plt.imshow(imgs[1], cmap='gray')
    plt.axis('off')
    plt.savefig('wass_example.png')
    plt.close('all')
    print('output example')

if __name__ == '__main__':

    fcn_mu = '[0,0]'
    fcn_sigma = '[[5,0],[0,5]]'

    mu_s      = sympy.sympify(fcn_mu)
    sigma_s   = sympy.sympify(fcn_sigma)
    x         = sympy.symbols([x for x in ['t','x','y']])
    mu        = sympy.lambdify(x,mu_s)
    sigma     = sympy.lambdify(x,sigma_s)

    gen_movie(mu, sigma)
