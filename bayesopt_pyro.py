import warnings
from scipy.special import erfc
from scipy.optimize import minimize
warnings.filterwarnings("error");
import numpy as np

import os 
import sys
import matplotlib.pyplot as plt
import torch
import pyro
import pyro.contrib.gp as gp
import pyro.distributions as dist

smoke_test = ('CI' in os.environ);
assert pyro.__version__.startswith('0.4.1');
pyro.enable_validation(True);
pyro.set_rng_seed(0);

def plot_gp(gpr, X, X_train=None, Y_train=None, samples=[], init=0, points=0, optimum=0):
    X = X.flatten()
    with torch.no_grad():
        mu, cov = gpr(X, full_cov=True)
    
    uncertainty = 1.96 * cov.diag().sqrt();
    ub = mu + uncertainty;
    lb = mu-uncertainty;
    if points and optimum:
        print(points, optimum)
        plt.annotate('Points Sampled: {0}\nMaximum f(x): {1:0.2f}'.format(points, optimum), xy=(0.5,0.90), xycoords='axes fraction');


    plt.fill_between(X.numpy(), ub.numpy(), lb.numpy(), alpha=0.1)
    plt.plot(X, mu, label='Mean')
    for i, sample in enumerate(samples):
        plt.plot(X, sample, lw=1, ls='--', label=f'Sample {i+1}', zorder=1)
    if X_train is not None:
        
        plt.scatter(X_train[0:init].numpy(), Y_train[0:init].numpy(), color='black', marker='x', zorder=5)
        plt.plot(X_train[init:len(X_train)-1].numpy(), Y_train[init:len(Y_train)-1].numpy(), 'rx')
        plt.plot(X_train[len(X_train)-1].numpy(), Y_train[len(Y_train)-1].numpy(), 'ro')
        
    plt.legend()


def expected_improvement(X, gp, optimum):
    if isinstance(X, np.ndarray):
        X = torch.from_numpy(X).float();

    with torch.no_grad():
    	mu, cov = gp.forward(X, full_cov=True);

    mu = mu.numpy();
    cov = cov.numpy();
    optimum = optimum.numpy();

    delta = mu - optimum;
    delta_pos = np.where(delta < 0, 0, delta);
    sigma = np.sqrt(np.diag(cov))#.reshape(-1,1);
    u = np.divide(delta, sigma)
    phi = np.exp(-0.5 * u**2) / np.sqrt(2*np.pi)
    Phi = 0.5 * erfc(-u / np.sqrt(2))
    EI = delta_pos + sigma * phi - np.abs(delta) * Phi

    return EI;

def plot_approximation(gpr, X, Y, X_next=None, show_legend=False):
    if isinstance(X, np.ndarray):
        X = torch.from_numpy(X).float();

    with torch.no_grad():
        mu, cov = gpr.forward(X, full_cov=True);

    mu = mu.numpy();
    cov = cov.numpy();
    X = X.numpy();
    std = np.sqrt(np.diag(cov));
    plt.fill_between(X.ravel(), 
                     mu.ravel() + 1.96 * std, 
                     mu.ravel() - 1.96 * std, 
                     alpha=0.1) 
    plt.plot(X, Y, 'y--', lw=1, label='Noise-free objective')
    plt.plot(X, mu, 'b-', lw=1, label='Surrogate function')
    plt.plot(gpr.X.numpy(), gpr.y.numpy(), 'kx', mew=3, label='Noisy samples')
    if X_next:
        plt.axvline(x=X_next, ls='--', c='k', lw=1)
    if show_legend:
        plt.legend()

def plot_acquisition(X, Y, X_next, show_legend=False):
    plt.plot(X, Y, 'r-', lw=1, label='Acquisition function')
    plt.axvline(x=X_next, ls='--', c='k', lw=1, label='Next sampling location')
    if show_legend:
        plt.legend()

# Maximize input function using L-BFGS optimization.
def proposed_point(func, args, bounds):
    min_val = float('inf');
    min_x = None;
    for x0 in np.random.uniform(bounds[0], bounds[1], 50):
        res = minimize(func, args=args, x0=x0, bounds=[bounds]);

        y0 = func(res.x, *args)[0];

        if y0 < min_val:
            min_val = y0;
            
            min_x = res.x[0];

    return min_x

def lower_bound(X, gp, optimum=None):
    if isinstance(X, np.ndarray):
        X = torch.from_numpy(X).float();

    with torch.no_grad():
        mu, cov = gpr.forward(X, full_cov=True);

    mu = mu.numpy();
    cov = cov.numpy();
    X = X.numpy();
    std = np.sqrt(np.diag(cov));

    return mu - 2*std;

def update_post(gpr, X, Y):
    X_sample = torch.cat((gpr.X, torch.tensor([X])), 0);
    y_sample = torch.cat((gpr.y, torch.tensor([Y])), 0);

    gpr.set_data(X_sample, y_sample);

    optimizer = torch.optim.Adam(gpr.parameters(), lr=0.005);
    gp.util.train(gpr, optimizer);

    return gpr

dom=(-1, 2)
noise= 0.4;
# Objective function
#obj_f = lambda x: np.sin(x);
#obj_f = lambda x, noise=0: x**3 + noise*dist.Normal(0, 1).sample(sample_shape= (1,) if not isinstance(x, torch.Tensor) else x.size());
#def obj_f(x, noise):
#    s = (1,) if not isinstance(x, torch.Tensor) else x.size();
#    return -(x-3)**2 + 10 + noise*dist.Normal(0, 1).sample(sample_shape=s);

#obj_f = lambda x, noise: -(x-3)**2 + 10 + noise*dist.Normal(0, 1).sample(sample_shape= (1,) if not isinstance(x, torch.Tensor) else x.size());
def obj_f(x, noise=0):
    if not isinstance(x, torch.Tensor):
        x = torch.tensor([x]);
    return -1.0 * torch.sin(3.0*x) - x**2.0 + 0.7*x + noise * dist.Normal(0, 1).sample(sample_shape= (1,) if not isinstance(x, torch.Tensor) else x.size())
#obj_f = lambda x, noise=0: -1.0 * torch.sin(3.0*x) - x**2.0 + 0.7*x + noise * dist.Normal(0, 1).sample(sample_shape= (1,) if not isinstance(x, torch.Tensor) else x.size());
#obj_f = lambda x: (x-1) * (x-2) * (x-3) + 5;

X = torch.arange(dom[0], dom[1], 0.01)#.reshape(-1, 1);
Y = obj_f(X, 0)

X_sample = dist.Uniform(dom[0], dom[1]).sample(sample_shape=(3,))
Y_sample = obj_f(X_sample, noise)

X_next = X_sample[-1];
X_sample = X_sample[:-1];

Y_next = Y_sample[-1];
Y_sample = Y_sample[:-1];

kernel = gp.kernels.RBF(input_dim=1, variance=torch.tensor(5.), lengthscale=torch.tensor(5.));
gpr = gp.models.GPRegression(X_sample, Y_sample, kernel, noise=torch.tensor([noise]))

#ac_func = lambda x, mu, cov, opt: x[np.argmax(np.diag(cov))][0]
ac_func = expected_improvement
#ac_func = lower_bound

optimum = torch.max(Y_sample);
optimum_x = X_sample[torch.argmax(Y_sample)];
n_iter = 10

plt.figure(figsize=(12, n_iter*3))
plt.subplots_adjust(hspace=0.4)

for i in range(n_iter):
    
    gpr = update_post(gpr, X_next, Y_next);

    min_acq = lambda X, gpr, optimum: -1 * ac_func(X, gpr, optimum)

    # Maximize the acquisition function to get next sampling point
    X_next = proposed_point(func=min_acq, args=(gpr, optimum), bounds=dom)
    
    # Obtain next noisy sample from the objective function
    Y_next = obj_f(X_next, noise)
    if Y_next > optimum:
        optimum = Y_next
        optimum_x = X_next;
    
    #plt.pause(0.01);
    #plt.clf();
    # Plot samples, surrogate function, noise-free objective and next sampling location
    plt.subplot(n_iter, 2, 2*i + 1)
    plot_approximation(gpr, X, Y, X_next, show_legend=i==0)
    plt.title(f'Iteration {i+1}')
    #plot_gp(mu_s, cov_s, X, X_train=gp.X_train, Y_train=gp.Y_train, init=len(init_sample), points=n+i, optimum=optimum)
    
    #print(X.shape)
    #print(ac_func(X, gpr, noise, optimum).shape)
    plt.subplot(n_iter, 2, 2*i+2)
    plot_acquisition(X, ac_func(X, gpr, optimum), X_next, show_legend=i==0)
    
    #plt.show(block=False)

    

plt.savefig('plot.png');

