import numpy as np
import matplotlib.pyplot as plt
import bootstrapped.bootstrap as bs
import bootstrapped.stats_functions as bs_stats

from random import seed, sample
from scipy  import stats
from sharedDefs import tsprint
from scipy.spatial import ConvexHull, Delaunay, convex_hull_plot_2d
from sklearn.decomposition import PCA

ECO_SEED = 23

def in_hull(Q, hull):
  if not isinstance(hull, Delaunay):
    vertices = [hull.points[i] for i in hull.vertices]
    hull = Delaunay(vertices)
  return hull.find_simplex(Q)>=0

def distance(v, w):
  return np.linalg.norm(v - w)

def estimateDistanceDistrib(Q, hull, interior, samplingProb = 1.0):

  (sample_int, sample_ext) = ([], [])
  for (isInterior, v) in zip(interior, Q):
    if(np.random.rand() <= samplingProb):
      val = max([distance(v, hull.points[i]) for i in hull.vertices])
      (sample_int if isInterior else sample_ext).append(val)

  ss_int = len(sample_int)
  ss_ext = len(sample_ext)
  ci_int = bs.bootstrap(np.array(sample_int), stat_func=bs_stats.mean)
  ci_ext = bs.bootstrap(np.array(sample_ext), stat_func=bs_stats.mean)
  print('-- mu internal distrubution: {0} (out of {1} samples)'.format(ci_int.value, ss_int))
  print('               95% interval: [{0}, {1}]'.format(ci_int.lower_bound, ci_int.upper_bound))
  print('-- mu external distrubution: {0} (out of {1} samples)'.format(ci_ext.value, ss_ext))
  print('               95% interval: [{0}, {1}]'.format(ci_ext.lower_bound, ci_ext.upper_bound))

  stats = {'int.sample': sample_int, 'int.ci': ci_int,
           'ext.sample': sample_ext, 'ext.ci': ci_ext}

  return stats

def plot(hull, Q_, interior, distStats):

  # unpacks parameters
  (unitsizew, unitsizeh) = (1.940, 1.916)
  (nrows, ncols) = (5, 6)
  nd = Q_.shape[1]

  fig = plt.figure(figsize=(ncols * unitsizew, nrows * unitsizeh))
  gs = fig.add_gridspec(nrows, ncols)
  plt.xkcd()

  panel1 = fig.add_subplot(gs[0:3, :])
  panel1.set_title('Item space')
  panel1.autoscale()
  panel1.axis('off')

  panel2 = fig.add_subplot(gs[3:, :])
  panel2.set_title('Distance distributions')

  # ensures items are mapped to 2D points
  if(nd == 2):
    V = hull.points
    Q = Q_
  else:
    pca = PCA(n_components=2)
    V = pca.fit_transform(hull.points)
    Q = pca.transform(Q_)

  # plots the popular items and the hull
  panel1.plot(V[:,0], V[:,1], 'bo')
  if(nd == 2):
    for simplex in hull.simplices:
      panel1.plot(V[simplex, 0], V[simplex, 1], 'b:')

  # plots the points in Q
  for (isInterior, v) in zip(interior, Q):
    panel1.plot(v[0], v[1], 'b+' if isInterior else 'r+')

  #  plots distance distributions
  # instead of histograms, induces gaussian models that fit the samples
  sample_int = distStats['int.sample']
  sample_ext = distStats['ext.sample']
  mu_int     = distStats['int.ci'].value
  mu_ext     = distStats['ext.ci'].value
  method = 'silverman'
  try:
    kde1 = stats.gaussian_kde(sample_int, method)
    kde1_pattern = 'b-'
  except np.linalg.LinAlgError:
    # if singular matrix, just black out; not a good solution, so ...
    kde1 = lambda e: [0 for _ in e]
    kde1_pattern = 'b:'

  try:
    kde2 = stats.gaussian_kde(sample_ext, method)
    kde2_pattern = 'r-'
  except np.linalg.LinAlgError:
    kde2 = lambda e: [0 for _ in e]
    kde2_pattern = 'r:'

  x_lb = min(sample_int + sample_ext)
  x_ub = max(sample_int + sample_ext)
  x_grades = 200
  x_eval = np.linspace(x_lb, x_ub, num=x_grades)
  y_int = kde1(x_eval)
  y_ext = kde2(x_eval)
  y_max = max(y_int + y_ext)
  panel2.plot(x_eval, y_int, kde1_pattern, label='Interior')
  panel2.plot(x_eval, y_ext, kde2_pattern, label='Exterior')
  panel2.axvline(mu_int, 0.0, max(y_int), color='b', linestyle=':')
  panel2.axvline(mu_ext, 0.0, max(y_ext), color='r', linestyle=':')

  panel2.fill_betweenx((0, y_max), distStats['int.ci'].lower_bound,
                                   distStats['int.ci'].upper_bound, alpha=.13, color='g')

  panel2.fill_betweenx((0, y_max), distStats['ext.ci'].lower_bound,
                                   distStats['ext.ci'].upper_bound, alpha=.13, color='g')

  plt.show()

  return None

def loadDataset(n, ss, d):

  U = [np.random.random(d) for _ in range(n)]             # Universe set
  idx = sample(list(range(n)), ss)                        # sample of points in U
  P = np.vstack([U[i] for i in idx])                      # set P representing popular items
  Q = np.vstack([U[i] for i in range(n) if i not in idx]) # set Q (complement of P)

  return (P, Q)

def main():

  # initialises random seed
  np.random.seed(ECO_SEED)
  seed(ECO_SEED)

  # determines the values of the model parameters
  (n, ss, d) = (2400, 800, 8)
  samplingProb = 1.00

  # creates/loads a dataset U and splits it into P (popular items) and Q (the complement of P) partitions
  tsprint('Creating dataset')
  (P, Q) = loadDataset(n, ss, d)

  # computes the convex hull of P
  tsprint('Computing the convex hull')
  hull = ConvexHull(P)

  # determines which points in Q are interior/exterior to the hull
  tsprint('Determining which points are interior (or exterior) to the hull')
  interior = in_hull(Q, hull)

  # estimates the distribution of max-distances from items interior/exterior to the hull
  #                                               to items representing a hull vertex
  tsprint('Estimating distance distributions')
  stats = estimateDistanceDistrib(Q, hull, interior, samplingProb)

  # presents the results
  #tsprint('Plotting the results')
  #plot(hull, Q, interior, stats)

if __name__ == "__main__":

  main()
