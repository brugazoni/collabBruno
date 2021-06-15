import sys
import numpy as np

from random     import seed
from sharedDefs import ECO_SEED
from sharedDefs import tsprint, estimateHullDistribs, plotHull, distance
from sharedDefs import in_hull, in_hull_tri, in_hull_lp

def testNils1(nd):
  """
  checking if methods from Juh_ and Niels attain similar results
  """

  qs = 10000 # size of partition q
  ps = 600   # size of partition P
  tsprint('Generating the dataset of P:{0}+Q:{1} {2}-dimensional vectors'.format(ps, qs, nd))
  (P, Q) = generateDataset(qs, ps, nd) # -- P is the set of vertices; Q is the set of points of interest

  tsprint("Applying Juh_'s method")
  (res1, _) = in_hull_tri(Q, P) # Juh_'s method

  tsprint("Applying Niels' method")
  (res2, _) = in_hull_lp(Q, P)  # Nils' method

  discrepancies = sum([1 if res1[i] != res2[i] else 0 for i in range(len(Q))])

  tsprint('{0} discrepancy(ies) found.'.format(discrepancies))

def testNils2(nd):
  """
  checking how running time compares
  """
  # modified from answers given by Juh_ and Niels:
  # https://stackoverflow.com/questions/16750618/whats-an-efficient-way-to-find-if-a-point-lies-in-the-convex-hull-of-a-point-cl
  qs = 3     # size of partition q
  ps = 600   # size of partition P

  tsprint('Generating the dataset of P:{0}+Q:{1} {2}-dimensional vectors'.format(ps, qs, nd))
  P  = np.random.rand(ps, nd) # point cloud
  q1 = np.random.rand(nd)     # point of interest (interior or exterior)
  q2 = (P[0] + P[1])/2        # point of interest (interior)
  q3 = -q2                    # point of interest (exterior?)
  Q = [q1, q2, q3]

  tsprint('Determining which points are interior (or exterior) to the hull induced from P')
  (interior, summary, hull) = in_hull(Q, P)
  tsprint(interior)

def generateDataset(n, ss, d):

  U = [np.random.random(d) for _ in range(n)]              # Universe set

  pivot = U[np.random.randint(0, n - 1)]
  L = sorted([(i, distance(U[i], pivot)) for i in range(n)], key = lambda e: e[1])
  idxs, _ = zip(*L[0:2*ss])
  idxs = [idxs[2*i] for i in range(ss)]

  P = np.vstack([U[i] for i in idxs])                      # set P of popular items
  Q = np.vstack([U[i] for i in range(n) if i not in idxs]) # set Q (complement of P)

  return (P, Q)

def main(nd):

  # initialises random seed
  np.random.seed(ECO_SEED)
  seed(ECO_SEED)

  # determines the values of the model parameters
  qs = 10000 # size of partition q
  ps = 600   # size of partition P
  samplingProb = 1.00

  # creates/loads a dataset U and splits it into P (popular items) and Q (the complement of P) partitions
  tsprint('Generating the dataset of P:{0}+Q:{1} {2}-dimensional vectors'.format(ps, qs, nd))
  (P, Q) = generateDataset(qs+ps, ps, nd)

  # determines which points in Q are interior/exterior to the hull
  tsprint('Determining which points are interior (or exterior) to the hull')
  (interior, _, hull, vertices) = in_hull(Q, P)

  # estimates the distribution of max-distances from items interior/exterior to the hull
  #                                               to items representing a hull vertex
  tsprint('Estimating distance distributions')
  stats = estimateHullDistribs(Q, vertices, interior, samplingProb)

  # presents the results
  tsprint('Plotting the results')
  plotHull(hull, Q, interior, stats, 'test_{0}'.format(nd))

if __name__ == "__main__":

  nd = int(sys.argv[1]) # number of dimensions

  testNils1(nd)
  #testNils2(nd)
  #main(nd)
