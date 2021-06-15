import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from scipy  import stats
from random import seed, shuffle

#from scipy.stats import norm
from sklearn.neighbors   import KernelDensity
from sklearn.utils.fixes import parse_version

# `normed` is being deprecated in favor of `density` in histograms
if parse_version(matplotlib.__version__) >= parse_version('2.1'):
    density_param = {'density': True}
else:
    density_param = {'normed': True}


ECO_SEED = 23

def de(sample, lib = 'scipy'):
  if(lib == 'scipy'):
    method = 'silverman'
    kernel = stats.gaussian_kde(sample, bw_method = method)
  elif(lib == 'scikit'):
    bw = 0.75
    kernel = KernelDensity(kernel='gaussian', bandwidth = bw).fit(sample)
  else:
    raise ValueError

  return kernel

def generateUnivariateData(ss):
  """
  Measurement model, returns mixed measurements
  """
  share = 0.3
  (mu1, mu2) = (0.0, 5.0)
  (sd1, sd2) = (1.0, 1.0)
  oracle = lambda e: (share * stats.norm(mu1, sd1).pdf(e) + (1 - share) * stats.norm(mu2, sd2).pdf(e))

  latent1 = np.random.normal(size = int(     share  * ss), loc = mu1, scale = sd1)
  latent2 = np.random.normal(size = int((1 - share) * ss), loc = mu2, scale = sd2)

  tmp = latent1.tolist() + latent2.tolist()
  shuffle(tmp)
  measurements = np.array(tmp)

  return (measurements, oracle)

def generateBivariateData(ss):
  """
  Measurement model, returns two coupled measurements
  """
  (mu1, mu2) = (0.0, 5.0)
  (sd1, sd2) = (1.0, 1.0)

  latent1 = np.random.normal(size = ss, loc = mu1, scale = sd1)
  latent2 = np.random.normal(size = ss, loc = mu2, scale = sd2)

  measurement1 = latent1 + latent2
  measurement2 = latent1 - latent2
  return (measurement1, measurement2)

def univariateTest(data, lib):

  # generates mocked bivariate data
  (sample, oracle) = data
  xmin = sample.min()
  xmax = sample.max()

  # estimates the density function from the measurements
  X = np.linspace(xmin, xmax, 100)
  kernel = de(sample)
  Y = kernel(X)

  # plots the results
  fig, ax = plt.subplots()
  ax.plot(X, Y,         'b.', label = 'Estimated', markersize=2)
  ax.plot(X, oracle(X), 'k-', label = 'Oracle', alpha = 0.2)
  plt.legend()

  plt.show()

  return None

def bivariateTest(data, lib):

  # generates mocked bivariate data
  (m1, m2) = data
  xmin = m1.min()
  xmax = m1.max()
  ymin = m2.min()
  ymax = m2.max()

  # estimates the density function from the measurements
  X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
  positions = np.vstack([X.ravel(), Y.ravel()])
  sample = np.vstack([m1, m2])
  kernel = de(sample)
  Z = np.reshape(kernel(positions).T, X.shape) #xxx try np.exp(kernel(positions)).T

  # plots the results
  fig, ax = plt.subplots()
  im = ax.imshow(np.rot90(Z), cmap=plt.cm.gist_earth_r, extent=[xmin, xmax, ymin, ymax])
  cax = plt.axes([0.84, 0.11, 0.03, 0.77])
  plt.colorbar(im, cax=cax)
  ax.plot(m1, m2, 'm.', markersize=1, alpha=.15)
  ax.set_xlim([xmin, xmax])
  ax.set_ylim([ymin, ymax])
  plt.show()

  return None

def main(ss, nd, lib):

  # locks the random number generators
  np.random.seed(ECO_SEED)
  seed(ECO_SEED)

  # performs and illustrates the density estimation
  if(nd == 1):
    data = generateUnivariateData(ss)
    univariateTest(data, lib)

  elif(nd == 2):
    data = generateBivariateData(ss)
    bivariateTest(data, lib)

  else:
    raise ValueError

  return None

if(__name__ == '__main__'):

  ss  = int(sys.argv[1]) # number of measurements to be generated
  nd  = int(sys.argv[2]) # number of dimensions of each measurement (1 or 2)
  lib = sys.argv[3]      # name of KDE library to use

  main(ss, nd, lib)
