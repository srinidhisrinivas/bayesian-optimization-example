import warnings
from scipy.special import erfc
from scipy.optimize import minimize
warnings.filterwarnings("error");

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
        mu, cov = gpr.forward(X, full_cov=True)
    
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

dom = (0, 6);

noise= 0.4;
# Objective function
#obj_f = lambda x: np.sin(x);
#obj_f = lambda x: x**3;
obj_f = lambda x, noise: -(x-3)**2 + 10 + noise*dist.Normal(0, 1).sample(sample_shape=x.size());
#obj_f = lambda x, noise=0: -np.sin(3*x) - x**2 + 0.7*x + noise * np.random.randn(*x.shape);
#obj_f = lambda x: (x-1) * (x-2) * (x-3) + 5;

X = torch.arange(dom[0], dom[1], 0.01).reshape(-1, 1);
Y = obj_f(X, 0)

init_sample = dist.Uniform(dom[0], dom[1]).sample(sample_shape=(8,))
y_init = obj_f(init_sample, noise)

kernel = gp.kernels.RBF(input_dim=1, variance=torch.tensor(5.), lengthscale=torch.tensor(5.));
gpr = gp.models.GPRegression(init_sample, y_init, kernel, noise=torch.tensor([noise]))
#gpr = gp.models.GPRegression(kernel, noise=torch.tensor([noise]))

optimizer = torch.optim.Adam(gpr.parameters(), lr=0.005);
gp.util.train(gpr, optimizer);
"""
loss_fn = pyro.infer.Trace_ELBO().differentiable_loss
losses=[]
num_steps = 10 if not smoke_test else 2;
for i in range(num_steps):
    optimizer.zero_grad();
    loss = loss_fn(gpr.model, gpr.guide)
    loss.backward()
    optimizer.step();
    losses.append(loss.item())
"""
#plt.plot(losses);
plt.figure();
plot_gp(gpr, X, X_train=init_sample, Y_train=y_init);

plt.show()

