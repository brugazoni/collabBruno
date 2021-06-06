"""
  `estimate.py`
  In `P01`/`simulate.py', the simulated user study is predicated on a "surprise threshold",
  which is an idealised construct that corresponds to the degree of surprise above which:
  (1) longer explanations are systematically preferred by participants, or
  (2) participants systematically explore a larger share of the available explanations

  This script aims to estimate this threshold using datasets collected from the Spotify platform.
  The process is itself predicated on a number of conventions and premises, which we make
  explicit in the following:

  C1. Adopt the definition of `popularity` advanced by Kaminskas and Bridge [1] as a convention.
      Extend the definition to allow for different levels of recency;
  C2. Adopt the definition of `surprise`   advanced by Kaminskas and Bridge [2] as a convention.
      Select the model in Equation 5, with Euclidean distance;
  C3. Adopt the definition of `relevant feature` advanced by Tversky [3] as a convention.
      Note that this definition also describes how items may be compared accordin to their features.

  (the term 'item' refers to a music track served by the Spotify streaming platform)

  P0. xxx high variability of behaviour. no hope for individual level regularity; seeking for population-level regularity
  P1. Assume that the available item features are relevant to the population of users {C3};
  P2. Assume that the set of items that are popular in a given period of time are highly similar
      (among themselves) than if compared to non-popular items [4] {C1};
  P3. Assume that every user has at least one popular item in their profile {testable};
  P4. Assume that we have a representative sample of top N most popular items in a target region
      (eg Brazil) nowadays;
  P5. Assume that every user has at least one popular item in their profile {testable};
  P6. xxx S_hat homomorphic/isomorphic to S

  Rationale
  A1. From [P1] and [P2], we conclude that there is a region in item space within which most of
      the popular items are confined {testable; assess average pop(i) for i interior/exterior to hull};

  Evidence

  Weaknesses

  [4]. gives us transitivity to the idea of convex hull in item space; Bourdier


  syntax .: python simulate.py <number of agents> <number of queries> [<mock>]

  example : python simulate.py 15 6 mock

"""

import numpy as np
import sharedDefs as ud

from random     import seed, sample
from os.path    import join
from sharedDefs import ECO_SEED
from sharedDefs import getMountedOn, deserialise, tsprint, serialise, saveLog, dict2text, saveAsText
from sharedDefs import in_hull, estimateDistanceDistrib, plotHull, distance, buildDataset
from sharedDefs import headerfy

from scipy.spatial   import ConvexHull
from sklearn.cluster import KMeans

def main():

  ud.LogBuffer = []

  # initialises random seed
  #np.random.seed(ECO_SEED)
  #seed(ECO_SEED)

  # determines the values of the model parameters
  sourcepath=[getMountedOn(), 'Task Stage', 'Task - collabBruno', 'collabBruno', 'results',  'spotify']
  targetpath=[getMountedOn(), 'Task Stage', 'Task - collabBruno', 'collabBruno', 'results',  'spotify']

  ignoreLinks = ['43F1e4glaVnVe3DXOU3Xum', '4BQMlG5VKApk9AdIsjxDqx',  '05KOgYg8PGeJyyWBPi5ja8',
                 '0wfbD5rAksdXUzRvMfM3x5', '3DXncPQOG4VBw3QHh3S817', '72Q0FQQo32KJloivv5xge2',
                 '5JDdzBawkk5UayooiFl5lM', '12PQ6KbFqC7xCsqDV8bmpb', '3l8yQMdniG6Os8gUBNXV57',
                 '5mAxA6Q1SIym6dPNiFLUyd']

  featureFields = ['acousticness', 'danceability', 'duration_ms', 'energy', 'explicit',
                   'instrumentalness', 'key', 'liveness', 'loudness', 'mode', 'speechiness', 'tempo',
                   'popularity', 'valence', 'release_date']

  samplingProbs = (1.0, 1/58)
  n_components  = 2

  # loads preprocessed data
  tsprint('Loading preprocessed data')
  features = deserialise(join(*sourcepath, 'features'))
  url2id   = deserialise(join(*sourcepath, 'url2id'))

  # remove links that failed manual inspection
  for urlID in ignoreLinks:
    url2id.pop(urlID)

  # builds the dataset that will be employed by the estimation processes
  tsprint('Building the dataset')
  (P, Q, samples) = buildDataset(url2id, features, samplingProbs, n_components)
  (allPopIDs, allItemIDs, popIDs, itemIDs) = samples
  tsprint('-- {0:5d} out of {1:6d} popular items included in the sample P'.format(len(popIDs),  len(allPopIDs)))
  tsprint('-- {0:5d} out of {1:6d} regular items included in the sample Q'.format(len(itemIDs), len(allItemIDs)))

  # computes the convex hull of P
  tsprint('Computing the convex hull around P')
  hull = ConvexHull(P)

  # determines which points in Q are interior/exterior to the hull
  tsprint('Determining which points are interior (or exterior) to the hull')
  interior, summary = in_hull(Q, hull)
  tsprint('-- {0:5d} items in sample Q are interior to the hull induced from P'.format(summary['interior']))
  tsprint('-- {0:5d} items in sample Q are exterior to the hull induced from P'.format(summary['exterior']))

  # estimates the distribution of max-distances from items interior/exterior to the hull
  #                                               to items representing hull vertices
  tsprint('Estimating surprise distributions for popular and regular items')
  stats = estimateDistanceDistrib(Q, hull, interior)

  # presents the results
  tsprint('Plotting the results')
  plotHull(hull, Q, interior, stats, join(*targetpath, 'panel'))

  # gathers evidence to some supporting premises

  # Evidence 1: 
  # -- It seems we have two dense regions regarding popular items. These are induced by the fact that
  #    one of the features of an item is the binary indicator "explicit", which together with 
  #    "liveness" and "speechiness" seems to split the two regions in the space
  X = np.array([features[itemID] for itemID in allPopIDs])
  kmeans = KMeans(n_clusters=2, random_state=ECO_SEED).fit(X)

  numOfFeatures = X.shape[1]
  mask = '{0}\t{1}\t' + '\t'.join(['{{{0}:6.3f}}'.format(i+2) for i in range(numOfFeatures)])
  header = headerfy(mask).format('ItemID', 'Cluster', *featureFields)
  content = [header]
  for itemID, cluster in zip(allPopIDs, kmeans.labels_):
    buffer = mask.format(itemID, cluster, *features[itemID])
    content.append(buffer)
  saveAsText('\n'.join(content), join(*targetpath, 'evidence1.csv'))






  # saves the results
  serialise(P,        join(*targetpath, 'P'))
  serialise(Q,        join(*targetpath, 'Q'))
  serialise(samples,  join(*targetpath, 'samples'))
  serialise(interior, join(*targetpath, 'interior'))
  serialise(hull,     join(*targetpath, 'hull'))

  saveLog(join(*targetpath, 'estimate.log'))

if __name__ == "__main__":

  main()
