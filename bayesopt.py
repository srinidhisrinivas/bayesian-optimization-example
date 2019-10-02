import numpy as np
np.set_printoptions(precision=3)
import warnings
from scipy.special import erfc
warnings.filterwarnings("error");

def kernel(X1, X2, l=1.0, sigma_f=1.0):
    ''' Isotropic squared exponential kernel. Computes a covariance matrix from points in X1 and X2. Args: X1: Array of m points (m x d). X2: Array of n points (n x d). Returns: Covariance matrix (m x n). '''
    sqdist = np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)
    #print(np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X**2, 1));
    return sigma_f**2 * np.exp(-0.5 / l**2 * sqdist)

def periodic_kernel(X1, X2, l=1.0, p=1.0, sigma_f=1.0):
    
    dist = np.absolute(X1.reshape(-1, 1) - X2.reshape(1, -1)) * np.pi / p;
    inner = -2 * np.sin(dist)**2 / l**2;
    return sigma_f**2 * np.exp(inner);

import matplotlib.pyplot as plt

from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

def plot_gp(mu, cov, X, X_train=None, Y_train=None, samples=[], init=0, points=0, optimum=0):
    X = X.ravel()
    mu = mu.ravel()
    uncertainty = 1.96 * np.sqrt(np.diag(cov))

    if points and optimum:
        plt.annotate('Points Sampled: {0}\nMaximum f(x): {1:0.2f}'.format(points, optimum), xy=(0.5,0.90), xycoords='axes fraction');
        #plt.text(0.05, 0.95, 'Points Sampled: {}'.format(points), fontsize=10, verticalalignment='top', transform=plt.axis.transAxes);
    #plt.ylim([-2, 2])
    #plt.xlim([-6, 6]);

    plt.fill_between(X, mu + uncertainty, mu - uncertainty, alpha=0.1)
    plt.plot(X, mu, label='Mean')
    for i, sample in enumerate(samples):
        plt.plot(X, sample, lw=1, ls='--', label=f'Sample {i+1}', zorder=1)
    if X_train is not None:
        
        plt.scatter(X_train[0:init], Y_train[0:init], color='black', marker='x', zorder=5)
        plt.plot(X_train[init:len(X_train)-1], Y_train[init:len(Y_train)-1], 'rx')
        plt.plot(X_train[len(X_train)-1], Y_train[len(Y_train)-1], 'ro')
        
    plt.legend()

def plot_gp_2D(gx, gy, mu, X_train, Y_train, title, i):
    ax = plt.gcf().add_subplot(1, 2, i, projection='3d')
    ax.plot_surface(gx, gy, mu.reshape(gx.shape), cmap=cm.coolwarm, linewidth=0, alpha=0.2, antialiased=False)
    ax.scatter(X_train[:,0], X_train[:,1], Y_train, c=Y_train, cmap=cm.coolwarm)
    ax.set_title(title)

def cube(x):
    return x**3;

class GP:
    def __init__(self):
        #self.mean_func = zero_mean();
        self.X_train = np.array([]).reshape(-1,1);
        self.Y_train = np.array([]).reshape(-1,1);
        pass;

    def zero_mean(self, x):
        return np.zeros(x.shape);

    def update_post(self, x, y):
        if np.isscalar(x):
            x = np.array([x]).reshape(-1,1);

        if np.isscalar(y):
            y = np.array([y]).reshape(-1, 1)

        self.X_train = np.append(self.X_train, x.reshape(-1, 1), axis=0)
        self.Y_train = np.append(self.Y_train, y.reshape(-1, 1), axis=0)

    def posterior_predictive(self, X_s, l=1.0, sigma_f=1.0, sigma_y = 1e-8):
        ''' Computes the suffifient statistics of the GP posterior predictive distribution from m training data X_train and Y_train and n new inputs X_s. Args: X_s: New input locations (n x d). X_train: Training locations (m x d). Y_train: Training targets (m x 1). l: Kernel length parameter. sigma_f: Kernel vertical variation parameter. sigma_y: Noise parameter. Returns: Posterior mean vector (n x d) and covariance matrix (n x n). '''
        K = kernel(self.X_train, self.X_train, l=l, sigma_f=sigma_f) + sigma_y**2 * np.eye(len(self.X_train))
        print(K)
        K_s = kernel(self.X_train, X_s, l=l, sigma_f=sigma_f)
        K_ss = kernel(X_s, X_s, l=l, sigma_f=sigma_f) + 1e-8 * np.eye(len(X_s))
        K_inv = np.linalg.inv(K)
        
        # Equation (4)
        mu_s = K_s.T.dot(K_inv).dot(self.Y_train)

        print(K_inv.shape)
        print(K_ss.shape)
        print(K_s.shape)
        # Equation (5)
        cov_s = K_ss - K_s.T.dot(K_inv).dot(K_s)
        #raise np.linalg.LinAlgError('Lol')
        return mu_s, cov_s


budget = 30;
# Feasible region 
dom = (0, 3);

# Objective function
#obj_f = lambda x: np.sin(x);
#obj_f = lambda x: x**3;
#obj_f = lambda x: -(x-3)**2 + 10;
obj_f = lambda x: (x-1) * (x-2) * (x-3) + 5;

def expected_improvement(X, mu, cov, optimum):
    delta = mu - optimum;
    delta_pos = np.where(delta < 0, 0, delta);
    sigma = np.sqrt(np.diag(cov)).reshape(-1,1);
    u = np.divide(delta, sigma)

    phi = np.exp(-0.5 * u**2) / np.sqrt(2*np.pi)
    Phi = 0.5 * erfc(-u / np.sqrt(2))
    EI = delta_pos + sigma * phi - np.abs(delta) * Phi
    return X[np.argmax(EI)][0];

ac_func = lambda x, mu, cov, opt: x[np.argmax(np.diag(cov))][0]
#ac_func = expected_improvement
#init_sample = np.linspace(dom[0], dom[1], 1);
init_sample = np.random.uniform(low=dom[0], high=dom[1], size=2)
#init_sample = init_sample[:-1]
#init_sample = np.array([-4, -3, -2, -1, 1]).reshape(-1, 1);
#print(init_sample, obj_f(init_sample));

gp = GP();
gp.update_post(init_sample, obj_f(init_sample));

n = n0 = init_sample.shape[0];
X = np.arange(dom[0], dom[1], 0.2).reshape(-1, 1);
mu_s, cov_s = gp.posterior_predictive(X)

sampled_x = init_sample

optimum = np.max(obj_f(init_sample));
optimum_x = init_sample[np.argmax(obj_f(init_sample))];

while n < budget:
    
    try:
        samples = np.random.multivariate_normal(mu_s.ravel(), cov_s, 3)
    except RuntimeWarning:
        print('Covariance matrix not positive semi-definite\n')
        #print(cov_s)
        break;
    plt.pause(1)
    plt.clf()
    plot_gp(mu_s, cov_s, X, X_train=gp.X_train, Y_train=gp.Y_train, samples=samples, init=len(init_sample), points=n, optimum=optimum)
    plt.show(block=False)

    sample = ac_func(X, mu_s, cov_s, optimum);
    sampled_x = np.append(sampled_x, sample)
    y_sample = obj_f(sample);
    if optimum < y_sample:
        optimum = y_sample;
        optimum_x = sample;

    gp.update_post(sample, y_sample);
    try:
        mu_s, cov_s = gp.posterior_predictive(X);
    except np.linalg.LinAlgError:
        print('Singular Matrix');
        break;

    n += 1;

print(sampled_x)
print('\n')
diff_matrix = []
for i in range(len(sampled_x)):
    row = []
    for j in range(len(sampled_x)):
        row.append(sampled_x[i] - sampled_x[j]);
    diff_matrix.append(row);

print(np.absolute(np.array(diff_matrix)))
# Finite number of points

#X = np.arange(-5, 5, 0.2).reshape(-1, 1)
#mu_s, cov_s = gp.posterior_predictive(X)
#print(cov_s)

#samples = np.random.multivariate_normal(mu_s.ravel(), cov_s, 3)
#plot_gp(mu_s, cov_s, X, X_train=gp.X_train, Y_train=gp.Y_train, samples=samples, init=5)

#plt.savefig('./gp_plots/2.png');
#plt.close()
plt.show()
"""
noise = 0.4

# Noisy training data
X_train = np.arange(-3, 4, 1).reshape(-1, 1)
Y_train = np.sin(X_train) + noise * np.random.randn(*X_train.shape)

gp.update_post(X_train, Y_train);
# Compute mean and covariance of the posterior predictive distribution
mu_s, cov_s = gp.posterior_predictive(X, sigma_y=noise)

samples = np.random.multivariate_normal(mu_s.ravel(), cov_s, 3)
plot_gp(mu_s, cov_s, X, X_train=X_train, Y_train=Y_train, samples=samples)
#plt.savefig('./gp_plots/3.png');
#plt.close()
plt.show()
"""