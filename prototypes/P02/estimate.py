"""
  `estimate.py`

  syntax .: python estimate.py <config file>
  example : python estimate.py ../configs/general_T01_C0.cfg

  In `P01`/`simulate.py', the simulated user study is predicated on a "surprise threshold",
  which is an idealised construct that corresponds to the degree of surprise above which:
  (1) longer explanations are systematically preferred by participants, or
  (2) participants systematically explore a larger share of additional information (motivation)

  This script aims to estimate that threshold using datasets collected from the Spotify platform.
  The process is itself predicated on a number of conventions, premises and assumptions, which
  we make explicit in the following:

  Conventions:
  C1. Adopt a definition of `popularity` advanced by many authors [1][2] as a convention;
  C2. Adopt a definition of `surprise`   advanced by Kaminskas and Bridge [3] as a convention;
  C3. Adopt the definition of `relevant feature` from Tversky [4] as a convention.
      This definition describes how items are compared according to their features.

  Premises {testable}:
  (From now on, the term 'item' refers to a music track served by the Spotify streaming platform)

  P3. Assume that every user has at least one highly popular item in their profile;


  Assumptions:
  A0. Owing to individual differences, high variability of the behaviour of interest is expected.
      -- This means there is no hope for useful regularity at low cost in the individual level;
      -- Thus, we focus on seeking useful regularity in the population-level;
  A1. Assume that the available item features are relevant to the population of users [C3];
  A2. Assume that the set of items that are highly popular in a given period of time are highly
      similar (among themselves) than if compared to less-popular items [C1][C3][5];
  A3. Assume that we have a representative sample of top N most popular items in a target region
      (e.g., Brazil) nowadays;


  Rationale
  A1. From [P1] and [P2], we conclude that there is a region in item space within which most of
      the popular items are confined {testable; assess average pop(i) for i interior/exterior to hull};

  Evidence
  To complete

  Weaknesses
  To complete

  Bibliography

  [1] Vargas, S., and Castells, P. (2011). Rank and relevance in novelty and diversity metrics
      for recommender systems. In Proceedings of the 5th ACM Conference on Recommender Systems
      (pp. 109-116).

  [2] Kaminskas, M. and Bridge, D. (2016). Diversity, serendipity, novelty, and coverage: a
      survey and empirical analysis of beyond-accuracy objectives in recommender systems.
      ACM Transactions on Interactive Intelligent Systems (TiiS) 7.1: 1-42.

  [3] Kaminskas, M. and Bridge, D. (2014). Measuring surprise in recommender systems.
      Proceedings of the workshop on recommender systems evaluation: Dimensions and design
      (Workshop programme of the 8th ACM conference on recommender systems).

  [4] Tversky, A. (1977). Features of similarity. Psychological Review, 84(4), 327.

  [5] To be selected; Candidates:

      Bourdieu, P. (2008). Distinction: A social critique of the judgement of taste. Routledge.

      Bourdieu, P. (1985). The market of symbolic goods. Poetics, 14(1-2), 13-44.
      https://doi.org/10.1016/0304-422X(85)90003-8

      Prior, N. (2013), Bourdieu and the Sociology of Music Consumption: A Critical Assessment of
      Recent Developments. Sociology Compass, 7: 181-193. https://doi.org/10.1111/soc4.12020

      Also keep an eye on Jonathan Kropf:
      https://www.uni-kassel.de/fb05/fachgruppen-und-institute/soziologie/fachgebiete/soziologische-theorie/team/dr-jonathan-kropf

"""

import numpy as np
import sharedDefs as ud

from random     import seed, sample
from os.path    import join
from sharedDefs import ECO_SEED
from sharedDefs import getMountedOn, deserialise, tsprint, serialise, saveLog, dict2text, saveAsText
from sharedDefs import in_hull, estimateHullDistribs, plotHull, distance, buildDataset
from sharedDefs import headerfy

from scipy.spatial   import ConvexHull
from sklearn.cluster import KMeans

def main():

  ud.LogBuffer = []

  # initialises random seed
  np.random.seed(ECO_SEED)
  seed(ECO_SEED)

  # determines the values of the model parameters
  sourcepath=[getMountedOn(), 'Task Stage', 'Task - collabBruno', 'collabBruno', 'results',  'spotify']
  targetpath=[getMountedOn(), 'Task Stage', 'Task - collabBruno', 'collabBruno', 'results',  'spotify']

  featureFields = ['acousticness',     'danceability', 'duration_ms', 'energy',   'tempo',
                   'instrumentalness', 'release_date', 'liveness',    'loudness', 'mode',
                   'speechiness',      'explicit',     'popularity',  'valence',  'key']

  # use this with the tracks 600k
  ignoreLinks = ['43F1e4glaVnVe3DXOU3Xum', '4BQMlG5VKApk9AdIsjxDqx',  '05KOgYg8PGeJyyWBPi5ja8',
                 '0wfbD5rAksdXUzRvMfM3x5', '3DXncPQOG4VBw3QHh3S817', '72Q0FQQo32KJloivv5xge2',
                 '5JDdzBawkk5UayooiFl5lM', '12PQ6KbFqC7xCsqDV8bmpb', '3l8yQMdniG6Os8gUBNXV57',
                 '5mAxA6Q1SIym6dPNiFLUyd', '2Yia0Gh4n61fPAjrNE5i2t']

  # use this witht the tracks 163k
  #ignoreLinks = ['7KcGEssn7BnJdTgildK5y0', '2Yia0Gh4n61fPAjrNE5i2t', '3DXncPQOG4VBw3QHh3S817',
  #               '72Q0FQQo32KJloivv5xge2', '4mJDfMcT7odIUjWlb2WO4L', '2Ny6TS3vPVwx5PlJXXkLUW',
  #               '03hqMhmCZiNKMSPmVabPLP', '3tWa5qmAJ2OsHV6KxNZBxc', '65MOTX6uSFAnqXl7dlAuId',
  #               '5Htb1uFQ1KrkFXlefS8oGj', '3hYvGiw2Q8QE2YWdIJLBz0', '5mAxA6Q1SIym6dPNiFLUyd']



  minPopularity = 0.75

  samplingProbs = (1.0, 1/58)
  #dims = [2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15]
  #epss = [1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1]

  #dims = [2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13]
  #epss = [1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1]

  #dims = [2,  3,  4,  5,  6,  7]
  #epss = [1,  1,  1,  1,  1,  1]

  #dims = [7,  8,  9, 10]
  #epss = [1,  1,  1,  1]

  #dims = [7,   8,   9]
  #epss = [1.0, 1.0, 1.0]

  #dims = [2,  3,  4,  5]
  #epss = [1,  1,  1,  1]

  #dims = [2,   10,  11,  13]
  #epss = [1.0, 1.0, 1.0, 1.0]

  dims = [7]
  epss = [.95]

  # loads preprocessed data
  tsprint('Loading preprocessed data')
  features = deserialise(join(*sourcepath, 'features'))
  url2id   = deserialise(join(*sourcepath, 'url2id'))

  # remove links that failed manual inspection
  for urlID in ignoreLinks:
    url2id.pop(urlID)

  for (n_components, epsilon) in zip(dims, epss):

    # builds the dataset that will be employed by the estimation processes
    tsprint('Building the dataset')
    (P, Q, samples, ev) = buildDataset(url2id, features, featureFields, samplingProbs, n_components, epsilon, minPopularity)
    (allPopIDs, allRegIDs, popIDs, regIDs) = samples
    tsprint('-- {0:5d} out of {1:6d} popular items included in the sample P'.format(len(popIDs), len(allPopIDs)))
    tsprint('-- {0:5d} out of {1:6d} regular items included in the sample Q'.format(len(regIDs), len(allRegIDs)))

    # determines which points in Q are interior/exterior to the hull
    tsprint('Determining which points are interior (or exterior) to the hull induced from P')
    (interior, summary, hull) = in_hull(Q, P)
    tsprint('-- {0:5d} items in sample Q are interior to the hull induced from P'.format(summary['interior']))
    tsprint('-- {0:5d} items in sample Q are exterior to the hull induced from P'.format(summary['exterior']))

    # estimates the distribution of max-distances from items interior/exterior to the hull
    #                                               to items representing hull vertices
    tsprint('Estimating surprise distributions for popular and regular items')
    stats, rawData = estimateHullDistribs(hull, Q, interior, popIDs, regIDs, features, featureFields)

    # presents the results
    tsprint('Plotting the results')
    stats['explained_variance'] = sum(ev)
    plotHull(hull, Q, interior, stats, rawData, join(*targetpath, 'panel_{0}d'.format(n_components)))

  # saves the results
  serialise(P,        join(*targetpath, 'P'))
  serialise(Q,        join(*targetpath, 'Q'))
  serialise(samples,  join(*targetpath, 'samples'))
  serialise(interior, join(*targetpath, 'interior'))
  serialise(hull,     join(*targetpath, 'hull'))

  tsprint('Job completed.')
  saveLog(join(*targetpath, 'estimate.log'))

if __name__ == "__main__":

  main()
