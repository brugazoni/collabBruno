import numpy as np

from random        import seed
from sharedDefs    import ECO_SEED
from sharedDefs    import tsprint, in_hull, estimateDistanceDistrib, plotHull, distance
from scipy.spatial import ConvexHull

def generateDataset(n, ss, d):

  U = [np.random.random(d) for _ in range(n)]              # Universe set

  pivot = U[np.random.randint(0, n - 1)]
  L = sorted([(i, distance(U[i], pivot)) for i in range(n)], key = lambda e: e[1])
  idxs, _ = zip(*L[0:2*ss])
  idxs = [idxs[2*i] for i in range(ss)]

  P = np.vstack([U[i] for i in idxs])                      # set P of popular items
  Q = np.vstack([U[i] for i in range(n) if i not in idxs]) # set Q (complement of P)

  return (P, Q)

def main():

  # initialises random seed
  np.random.seed(ECO_SEED)
  seed(ECO_SEED)

  # determines the values of the model parameters
  (n, ss, d) = (1000, 100, 2)
  samplingProb = 1.00

  # creates/loads a dataset U and splits it into P (popular items) and Q (the complement of P) partitions
  tsprint('Creating dataset')
  (P, Q) = generateDataset(n, ss, d)

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
  tsprint('Plotting the results')
  plotHull(hull, Q, interior, stats)

if __name__ == "__main__":

  main()
